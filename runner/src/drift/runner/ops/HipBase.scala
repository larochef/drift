package drift.runner.ops

import drift.runner.native.*
import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.*
import scala.collection.mutable

/** What every part of the GPU backend builds on: the HIP runtime, the kernel
  * modules and their functions, the launches, the buffers handed out and the
  * spares kept, scratch memory, and what the kernels that write a product's
  * input leave quantized for it.
  */
private[ops] trait HipBase extends Ops {

  /** The runtime the kernels are launched through. */
  protected def hip: HipRuntime

  /** What quantized weights meet in a matrix-vector product. */
  protected def inputs: MatVecInputs

  /** Every buffer handed out and not released yet, by its address. */
  protected val allocations = mutable.HashMap.empty[Long, MemorySegment]

  /** Released buffers kept for the next allocation of their size, by address
    * and oldest first, with the addresses of each size beside. A model
    * allocates the same few sizes over and over — a VAE's frames, a block's
    * work buffers — and each `hipMalloc` and `hipFree` is a call into the
    * driver, the second one waiting for the GPU: thousands a decode. Kept up to
    * `SpareBudget` bytes, the oldest given back first, and all of them when the
    * device has no room left.
    */
  protected val spare = mutable.LinkedHashMap.empty[Long, (MemorySegment, Long)]
  private val spareBySize =
    mutable.HashMap.empty[Long, mutable.ArrayDeque[Long]]
  private var spareBytes = 0L

  /** How much released memory is kept for reuse: 2 GiB, or
    * `-Ddrift.hip.spare=<MiB>` — 0 gives every buffer back at once.
    */
  private val spareBudget: Long =
    sys.props.get("drift.hip.spare").flatMap(_.toLongOption).getOrElse(2048L) *
      1024 * 1024

  protected def takeSpare(bytes: Long): Option[MemorySegment] =
    spareBySize.get(bytes).filter(_.nonEmpty).map { addresses =>
      // The newest of its size: the likeliest still in the driver's caches.
      val (pointer, _) = spare.remove(addresses.removeLast()).get
      spareBytes -= bytes
      pointer
    }

  protected def freeOldestSpare(): Unit = {
    val (address, (pointer, bytes)) = spare.head
    spare.remove(address)
    spareBySize(bytes).removeHead()
    spareBytes -= bytes
    hip.free(pointer)
  }

  protected def keepSpare(pointer: MemorySegment, bytes: Long): Unit =
    if (bytes > spareBudget) hip.free(pointer)
    else {
      while (spareBytes + bytes > spareBudget) freeOldestSpare()
      spare(pointer.address()) = (pointer, bytes)
      spareBySize.getOrElseUpdate(bytes, mutable.ArrayDeque.empty) +=
        pointer.address()
      spareBytes += bytes
    }
  protected val elementwiseKernels = new KernelModule(hip, "elementwise")
  protected val normKernels = new KernelModule(hip, "norm")
  protected val ropeKernels = new KernelModule(hip, "rope")
  protected val matvecKernels = new KernelModule(hip, "matvec")
  protected val convRotKernels = new KernelModule(hip, "convrot")

  /** hipBLAS, opened by the first GEMM. */
  private var blasHandle = Option.empty[HipBlas]
  protected def blas: HipBlas = blasHandle.getOrElse {
    val opened = new HipBlas(hip)
    blasHandle = Some(opened)
    opened
  }

  private val kernels = mutable.Map.empty[String, KernelFunction]
  protected def kernel(module: KernelModule, name: String): KernelFunction =
    kernels.getOrElseUpdate(name, module.function(name))

  /** Weight types with matrix-vector kernels, and whether they take x quantized
    * per 32 values (`quantize_x`) rather than as float.
    */
  protected val integerX: Map[DType, Boolean] = Map(
    GgmlQuants.Q8_0 -> true,
    GgmlQuants.Q4_0 -> true,
    KQuants.Q4_K -> true,
    KQuants.Q6_K -> true,
    GgmlQuants.Q5_1 -> true,
    KQuants.Q5_K -> true,
    IQuants.IQ4_NL -> true,
    IQuants.IQ4_XS -> true,
    RocmFp4.Dual -> true,
    RocmFp4.Fast -> true,
    DType.F32 -> false,
    DType.F16 -> false,
    DType.BF16 -> false,
    ConvRot.Codes -> false,
    ConvRot.Nibbles -> false
  )

  /** Up to this many rows of x, one matrix-vector pass serves them all. */
  val MaxMatVecRows = 8

  def name: String = "hip"

  /** Staging for `writeInts`, kept: an arena per write would cost more. A
    * longer write goes through it in several copies.
    */
  protected val intStaging =
    Arena.ofShared().allocate(4L * HipBase.StagedInts, 64)

  protected lazy val argmaxIndex = allocate(DType.I32, Shape.of(1))

  /** Launches and copies so far: what a kernel wrote stays as it is until the
    * next.
    */
  protected var deviceWrites = 0L

  /** Whether int8 products ran (the first `quantize` sets it): the kernels that
    * write a product's input (the norms) then quantize it as they write.
    */
  protected var quantizesX = false

  /** `values` at `address` that the kernel writing them also wrote quantized at
    * `codes` (codes, then scales, then sums, as `quantize` lays them out), as
    * of `written`. Blocks of 32 do not depend on rows: any rows within serve,
    * whatever their width.
    */
  final protected case class QuantizedValues(
      address: Long,
      values: Long,
      codes: MemorySegment,
      written: Long
  ) {
    def scales(from: Long): MemorySegment = offset(codes, values + 4 * from)
    def sums(from: Long): MemorySegment =
      offset(codes, values + 4 * (values / 32) + 4 * from)
  }
  private var quantizedValues = Option.empty[QuantizedValues]
  private var quantizedBuffer = Option.empty[Tensor]

  /** At most this many values are quantized as written: a decode step's or a
    * verification's inputs, not a prompt's (its products are GEMMs).
    */
  private val QuantizedMaximumValues = 8L * 16384

  /** Where a kernel writing `out` (rows of `cols`) writes it quantized too,
    * when int8 products may read it; and the kernel's `xq`, `xd` and `xs`
    * arguments (nulls when not).
    */
  protected def quantizedTarget(
      out: Tensor,
      cols: Long
  ): (Option[QuantizedValues], Seq[KernelArgument]) = {
    val values = out.shape.elementCount
    val eligible = quantizesX && out.dtype == DType.F32 &&
      values <= QuantizedMaximumValues && cols % 32 == 0
    if (!eligible) (None, Seq.fill(3)(Pointer(MemorySegment.NULL)))
    else {
      val bytes = values + 8 * (values / 32)
      val buffer = quantizedBuffer
        .filter(_.byteSize >= bytes)
        .getOrElse {
          quantizedBuffer.foreach(release)
          val fresh = allocate(DType.I8, Shape.of(math.max(bytes, 65536L)))
          quantizedBuffer = Some(fresh)
          fresh
        }
      val target =
        QuantizedValues(pointer(out).address(), values, pointer(buffer), 0)
      (
        Some(target),
        Seq(
          Pointer(target.codes),
          Pointer(target.scales(0)),
          Pointer(target.sums(0))
        )
      )
    }
  }

  /** After the launch that wrote `target`. */
  protected def wroteQuantized(target: Option[QuantizedValues]): Unit =
    target.foreach(t => quantizedValues = Some(t.copy(written = deviceWrites)))

  /** `m` rows of `x` (`k` wide) as the last kernel wrote them quantized, if it
    * did: codes, scales and sums.
    */
  protected def quantizedInput(
      x: Tensor,
      m: Long,
      k: Long
  ): Option[(MemorySegment, MemorySegment, MemorySegment)] =
    quantizedValues
      .filter(q => q.written == deviceWrites && x.dtype == DType.F32)
      .flatMap { q =>
        val bytes = pointer(x).address() - q.address
        val first = bytes / 4
        Option.when(
          bytes >= 0 && bytes % (4 * 32) == 0 && k % 32 == 0 &&
            first + m * k <= q.values
        )(
          (offset(q.codes, first), q.scales(first / 32), q.sums(first / 32))
        )
      }

  protected def launch(
      function: KernelFunction,
      grid: Long,
      block: Int,
      arguments: KernelArgument*
  ): Unit = {
    require(grid <= Int.MaxValue, s"${function.name} over $grid workgroups")
    deviceWrites += 1
    onWork()
    if (grid > 0)
      hip.launch(
        function,
        Dim3(grid.toInt),
        Dim3(block),
        0,
        MemorySegment.NULL,
        arguments*
      )
  }

  protected def launch(
      function: KernelFunction,
      grid: Dim3,
      block: Int,
      arguments: KernelArgument*
  ): Unit = {
    deviceWrites += 1
    onWork()
    hip.launch(
      function,
      grid,
      Dim3(block),
      0,
      MemorySegment.NULL,
      arguments*
    )
  }

  /** A grid-stride kernel over `count` values. */
  protected def elementwise(
      name: String,
      count: Long,
      arguments: KernelArgument*
  ): Unit =
    launch(
      kernel(elementwiseKernels, name),
      math.min((count + 255) / 256, 1L << 20),
      256,
      arguments*
    )

  /** `convert`'s kernel codes, by (from, to). */
  protected val conversionCodes: Map[(DType, DType), Int] = Map(
    (DType.F32, DType.F16) -> 0,
    (DType.F32, DType.BF16) -> 1,
    (DType.F16, DType.F32) -> 2,
    (DType.BF16, DType.F32) -> 3,
    (DType.F8E4M3, DType.F32) -> 4
  )

  /** A rotated linear (`rotated`), read where the file holds it, to BF16 at
    * `to`.
    */
  protected def dequantizeRotated(weight: Tensor, to: MemorySegment): Unit = {
    val parts = partsOf(weight)
    val blocks = weight.shape.elementCount / ConvRot.Group
    val grid = math.min((blocks + 15) / 16, 1L << 16)
    val shape = Seq(I64(blocks), I32(weight.shape.last.toInt))
    parts.levels match {
      case None =>
        launch(
          kernel(convRotKernels, "convrot_dequantize_bf16_i8"),
          grid,
          ConvRot.Group,
          (Seq(
            Pointer(pointer(weight)),
            Pointer(pointer(parts.scales)),
            I32(parts.scales.shape.elementCount.toInt),
            Pointer(to)
          ) ++ shape)*
        )
      case Some(levels) =>
        launch(
          kernel(convRotKernels, "convrot_dequantize_bf16_w4a8"),
          grid,
          ConvRot.Group,
          (Seq(
            Pointer(pointer(weight)),
            Pointer(pointer(levels.codebook)),
            Pointer(pointer(levels.relative)),
            Pointer(pointer(parts.scales)),
            Pointer(to)
          ) ++ shape :+ I32(levels.groupSize))*
        )
    }
  }

  protected def convertAt(
      from: MemorySegment,
      to: MemorySegment,
      count: Long,
      code: Int
  ): Unit =
    elementwise(
      "convert",
      count,
      Pointer(from),
      Pointer(to),
      I64(count),
      I32(code)
    )

  protected val imageKernels = new KernelModule(hip, "image")

  protected val audioKernels = new KernelModule(hip, "audio")

  protected val attentionKernels = new KernelModule(hip, "attention")
  protected val linearAttentionKernels =
    new KernelModule(hip, "linear_attention")

  protected val streamsKernels = new KernelModule(hip, "streams")

  /** Device memory for intermediate results, grown on demand and kept; one
    * region, so a caller takes all it needs at once.
    */
  private var scratchPointer = MemorySegment.NULL
  private var scratchBytes = 0L

  protected def scratch(at: Long, bytes: Long): MemorySegment = {
    if (at + bytes > scratchBytes) {
      if (scratchBytes > 0) hip.free(scratchPointer)
      scratchPointer = hip.allocate(at + bytes)
      scratchBytes = at + bytes
    }
    offset(scratchPointer, at)
  }

  protected def offset(address: MemorySegment, bytes: Long): MemorySegment =
    MemorySegment.ofAddress(address.address() + bytes)

  /** The device address of a tensor's first byte. */
  protected def pointer(tensor: Tensor): MemorySegment = {
    val base = tensor.storage match {
      case Storage.Device(pointer, _)    => pointer
      case Storage.Registered(_, device) => device
      case Storage.Host(_)               =>
        throw new IllegalArgumentException(
          "the hip backend cannot read unregistered host memory; upload the tensor first"
        )
    }
    offset(base, tensor.byteOffset)
  }

  def close(): Unit = {
    allocations.values.foreach(hip.free)
    allocations.clear()
    while (spare.nonEmpty) freeOldestSpare()
    if (scratchBytes > 0) hip.free(scratchPointer)
    scratchBytes = 0
    blasHandle.foreach(_.close())
    blasHandle = None
    Seq(
      elementwiseKernels,
      normKernels,
      ropeKernels,
      matvecKernels,
      convRotKernels,
      imageKernels,
      audioKernels,
      attentionKernels,
      linearAttentionKernels,
      streamsKernels
    )
      .foreach(_.close())
  }
}

private[ops] object HipBase {

  /** How many ints one staged copy carries. */
  val StagedInts = 65536
}
