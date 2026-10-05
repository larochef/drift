package drift.runner.ops

import drift.runner.native.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*

/** The GPU backend: each operation launches kernels of `runner/kernels` on the
  * default stream. Every allocation is freed when the backend closes.
  */
/** What quantized weights meet in a matrix-vector product: `Int8` quantizes x
  * per 32 values and multiplies four codes per `v_dot4`, as llama.cpp does;
  * `Float` keeps x as is and decodes the weights to floats in registers.
  */
enum MatVecInputs {
  case Int8
  case Float
}

final class HipOps(hip: HipRuntime, inputs: MatVecInputs) extends Ops {

  import KernelArgument.{F32, I32, I64, Pointer}

  private val allocations = mutable.ArrayBuffer.empty[MemorySegment]
  private val elementwiseKernels = new KernelModule(hip, "elementwise")
  private val normKernels = new KernelModule(hip, "norm")
  private val ropeKernels = new KernelModule(hip, "rope")
  private val matvecKernels = new KernelModule(hip, "matvec")

  /** hipBLAS, opened by the first GEMM. */
  private var blasHandle = Option.empty[HipBlas]
  private def blas: HipBlas = blasHandle.getOrElse {
    val opened = new HipBlas(hip)
    blasHandle = Some(opened)
    opened
  }

  private val kernels = mutable.Map.empty[String, KernelFunction]
  private def kernel(module: KernelModule, name: String): KernelFunction =
    kernels.getOrElseUpdate(name, module.function(name))

  /** Weight types with matrix-vector kernels, and whether they take x quantized
    * per 32 values (`quantize_x`) rather than as float.
    */
  private val integerX: Map[DType, Boolean] = Map(
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
    DType.BF16 -> false
  )

  /** Up to this many rows of x, one matrix-vector pass serves them all. */
  val MaxMatVecRows = 8

  /** From this many rows of x, `linear` is a GEMM: the weight dequantized to
    * F16 (a fixed cost, about 40 ms over a 4B model's weights), then hipBLAS.
    * Below, passes of `MaxMatVecRows` cost less.
    */
  val GemmMinimumRows = 32

  /** From this many slots, the experts' products group the slots by expert, so
    * that an expert's rows are read once per batch of up to 8 of its slots, not
    * once per slot: a prompt's tokens share experts. A verification's few slots
    * gain nothing from it (`specs/42`, step 10).
    */
  val GroupedExpertsMinimumSlots = 64

  def name: String = "hip"

  def allocate(dtype: DType, shape: Shape): Tensor = {
    val bytes = math.max(dtype.byteSize(shape.elementCount), 1)
    val pointer = hip.allocate(bytes)
    allocations += pointer
    Tensor(dtype, shape, Storage.Device(pointer, bytes), 0)
  }

  def release(tensor: Tensor): Unit = tensor.storage match {
    case Storage.Device(pointer, _) if tensor.byteOffset == 0 =>
      val at = allocations.indexWhere(_.address() == pointer.address())
      if (at >= 0) {
        allocations.remove(at)
        hip.free(pointer)
      }
    case _ => () // views, registered mappings: not ours to free
  }

  def mapFile(path: java.nio.file.Path): MappedWeights = {
    val registered = new drift.runner.native.RegisteredFile(hip, path)
    new MappedWeights(
      registered.host,
      Storage.Registered(registered.host, registered.device),
      () => registered.close()
    )
  }

  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    Ops.checkEmbedding(table, ids, out)
    require(
      integerX.contains(table.dtype),
      s"no kernels for ${table.dtype} tables yet"
    )
    val (count, columns) = (ids.shape.elementCount, table.shape.last)
    if (table.dtype == DType.F32)
      // any width: the matvec gathers take whole 32-value blocks
      launch(
        kernel(elementwiseKernels, "gather_rows_f32"),
        (count * columns + 255) / 256,
        256,
        Pointer(pointer(table)),
        Pointer(pointer(ids)),
        Pointer(pointer(out)),
        I32(count.toInt),
        I32(columns.toInt)
      )
    else
      launch(
        kernel(matvecKernels, s"gather_${table.dtype.name.toLowerCase}"),
        (count + 7) / 8,
        256,
        Pointer(pointer(table)),
        Pointer(pointer(ids)),
        Pointer(pointer(out)),
        I32(count.toInt),
        I32(columns.toInt)
      )
  }

  /** A new tensor filled from host memory by `write`. */
  private def upload(dtype: DType, shape: Shape)(
      write: MemorySegment => Unit
  ): Tensor = {
    val tensor = allocate(dtype, shape)
    val arena = Arena.ofConfined()
    try {
      val staging = arena.allocate(math.max(tensor.byteSize, 1), 64)
      write(staging)
      hip.copy(pointer(tensor), staging, tensor.byteSize)
      deviceWrites += 1
    } finally arena.close()
    tensor
  }

  def fromFloats(shape: Shape, values: Array[Float]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    upload(DType.F32, shape)(
      MemorySegment.copy(values, 0, _, JAVA_FLOAT, 0, values.length)
    )
  }

  def fromInts(shape: Shape, values: Array[Int]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    upload(DType.I32, shape)(
      MemorySegment.copy(values, 0, _, JAVA_INT, 0, values.length)
    )
  }

  /** Staging for `writeInts`, kept: an arena per write would cost more. */
  private val intStaging = Arena.ofShared().allocate(4L * 65536, 64)

  def writeInts(tensor: Tensor, values: Array[Int]): Unit = {
    require(
      tensor.dtype == DType.I32 && values.length <= tensor.shape.elementCount,
      s"${values.length} ints into ${tensor.shape}"
    )
    require(values.length <= 65536, s"${values.length} ints at once")
    MemorySegment.copy(values, 0, intStaging, JAVA_INT, 0, values.length)
    hip.copy(pointer(tensor), intStaging, 4L * values.length)
    deviceWrites += 1
  }

  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor = {
    Ops.checkBytes(dtype, shape, bytes)
    upload(dtype, shape)(
      MemorySegment.copy(MemorySegment.ofArray(bytes), 0, _, 0, bytes.length)
    )
  }

  def toFloats(tensor: Tensor): Array[Float] = {
    Ops.requireF32("toFloats", tensor)
    download(tensor)(_.toArray(JAVA_FLOAT))
  }

  def toInts(tensor: Tensor): Array[Int] = {
    require(tensor.dtype == DType.I32, s"toInts of ${tensor.dtype}")
    download(tensor)(_.toArray(JAVA_INT))
  }

  /** `read` of a tensor's bytes, staged in host memory. */
  private def download[A](tensor: Tensor)(read: MemorySegment => A): A = {
    val arena = Arena.ofConfined()
    try {
      val staging = arena.allocate(math.max(tensor.byteSize, 1), 64)
      hip.copy(staging, pointer(tensor), tensor.byteSize)
      read(staging.asSlice(0, tensor.byteSize))
    } finally arena.close()
  }

  private lazy val argmaxIndex = allocate(DType.I32, Shape.of(1))

  def argmax(values: Tensor): Int = {
    Ops.requireF32("argmax", values)
    launch(
      kernel(normKernels, "argmax_f32"),
      1,
      1024,
      Pointer(pointer(values)),
      Pointer(pointer(argmaxIndex)),
      I32(values.shape.elementCount.toInt)
    )
    hip.copy(intStaging, pointer(argmaxIndex), 4)
    intStaging.get(JAVA_INT, 0)
  }

  /** `candidates`' ids and values, `[rows, count]` pairs of ints. */
  private var candidateBuffer = Option.empty[Tensor]

  /** What each chunk of `candidates`' rows keeps, and the chunks each row
    * finished (the last leaves it at 0): `norm.hip`'s `candidates_f32`.
    */
  private var candidateKept = Option.empty[Tensor]
  private lazy val finishedCandidates = {
    val counters = allocate(DType.I32, Shape.of(65536))
    zero(counters)
    counters
  }
  private val CandidateChunk = 4096

  def candidates(values: Tensor, count: Int): Seq[Candidates] = {
    Ops.requireCandidates(values, count)
    val width = values.shape.last.toInt
    val rows = (values.shape.elementCount / width).toInt
    val kept = math.min(count, width)
    val pairs = 2L * rows * count
    require(pairs <= 65536, s"$rows rows of $count candidates")
    val buffer = candidateBuffer
      .filter(_.shape.elementCount >= pairs)
      .getOrElse {
        candidateBuffer.foreach(release)
        val fresh = allocate(DType.I32, Shape.of(math.max(pairs, 2048L)))
        candidateBuffer = Some(fresh)
        fresh
      }
    // each row's chunks keep their own candidates, then the last one to finish
    // selects among them
    val chunks = (width + CandidateChunk - 1) / CandidateChunk
    val keptValues = 2L * rows * chunks * count
    val chunkCandidates = candidateKept
      .filter(_.shape.elementCount >= keptValues)
      .getOrElse {
        candidateKept.foreach(release)
        val fresh = allocate(DType.I32, Shape.of(keptValues))
        candidateKept = Some(fresh)
        fresh
      }
    launch(
      kernel(normKernels, "candidates_f32"),
      rows.toLong * chunks,
      1024,
      Pointer(pointer(values)),
      Pointer(pointer(buffer)),
      I32(width),
      I32(count),
      Pointer(pointer(chunkCandidates)),
      Pointer(pointer(finishedCandidates))
    )
    hip.copy(intStaging, pointer(buffer), 4 * pairs)
    val all = intStaging.asSlice(0, 4 * pairs).toArray(JAVA_INT)
    (0 until rows).map { row =>
      val at = 2 * row * count
      Candidates(
        Array.tabulate(kept)(c => all(at + 2 * c + 1)),
        Array.tabulate(kept)(c =>
          java.lang.Float.intBitsToFloat(all(at + 2 * c))
        )
      )
    }
  }

  def maxAbs(x: Tensor): Float = {
    Ops.requireF32("maxAbs", x)
    hip.zero(pointer(argmaxIndex), 4)
    val count = x.shape.elementCount
    launch(
      kernel(normKernels, "max_abs_f32"),
      math.max(1L, math.min(1024L, (count + 255) / 256)),
      256,
      Pointer(pointer(x)),
      Pointer(pointer(argmaxIndex)),
      I64(count)
    )
    hip.copy(intStaging, pointer(argmaxIndex), 4)
    java.lang.Float.intBitsToFloat(intStaging.get(JAVA_INT, 0))
  }

  // ---- x quantized by the kernel that writes it ----------------------------------------

  /** Launches and copies so far: what a kernel wrote stays as it is until the
    * next.
    */
  private var deviceWrites = 0L

  /** Whether int8 products ran (the first `quantize` sets it): the kernels that
    * write a product's input (the norms) then quantize it as they write.
    */
  private var quantizesX = false

  /** `values` at `address` that the kernel writing them also wrote quantized at
    * `codes` (codes, then scales, then sums, as `quantize` lays them out), as
    * of `written`. Blocks of 32 do not depend on rows: any rows within serve,
    * whatever their width.
    */
  final private case class QuantizedValues(
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
  private def quantizedTarget(
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
  private def wroteQuantized(target: Option[QuantizedValues]): Unit =
    target.foreach(t => quantizedValues = Some(t.copy(written = deviceWrites)))

  /** `m` rows of `x` (`k` wide) as the last kernel wrote them quantized, if it
    * did: codes, scales and sums.
    */
  private def quantizedInput(
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

  // ---- launches ----------------------------------------------------------------------

  private def launch(
      function: KernelFunction,
      grid: Long,
      block: Int,
      arguments: KernelArgument*
  ): Unit = {
    require(grid <= Int.MaxValue, s"${function.name} over $grid workgroups")
    deviceWrites += 1
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

  private def launch(
      function: KernelFunction,
      grid: Dim3,
      block: Int,
      arguments: KernelArgument*
  ): Unit = {
    deviceWrites += 1
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
  private def elementwise(
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

  def add(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("add", a, b, out)
    elementwise(
      "add_f32",
      out.shape.elementCount,
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def mul(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("mul", a, b, out)
    elementwise(
      "mul_f32",
      out.shape.elementCount,
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def scale(x: Tensor, factor: Float, out: Tensor): Unit = {
    Ops.checkElementwise("scale", x, out)
    elementwise(
      "scale_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      F32(factor),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def addRow(x: Tensor, row: Tensor, out: Tensor): Unit = {
    Ops.checkAddRow(x, row, out)
    elementwise(
      "add_row_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(row)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit = {
    Ops.checkElementwise("activation", x, out)
    elementwise(
      "activation_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount),
      I32(kind.code),
      F32(inputScale)
    )
  }

  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("gated", gate, up, out)
    val (target, quantized) = quantizedTarget(out, out.shape.last)
    elementwise(
      "gated_f32",
      out.shape.elementCount,
      (Seq(
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        Pointer(pointer(out)),
        I64(out.shape.elementCount),
        I32(kind.code)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }

  /** `convert`'s kernel codes, by (from, to). */
  private val conversionCodes: Map[(DType, DType), Int] = Map(
    (DType.F32, DType.F16) -> 0,
    (DType.F32, DType.BF16) -> 1,
    (DType.F16, DType.F32) -> 2,
    (DType.BF16, DType.F32) -> 3,
    (DType.F8E4M3, DType.F32) -> 4
  )

  def convert(x: Tensor, out: Tensor): Unit = {
    Ops.checkConvert(x, out)
    convertAt(
      pointer(x),
      pointer(out),
      x.shape.elementCount,
      conversionCodes(x.dtype -> out.dtype)
    )
  }

  private def convertAt(
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

  /** One workgroup of 256 per row. */
  private def rowWise(
      name: String,
      rows: Long,
      arguments: KernelArgument*
  ): Unit =
    launch(kernel(normKernels, name), rows, 256, arguments*)

  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("rmsNorm", x, out, weight)
    // narrow rows (a pixel's channels), millions of them: a thread each
    if (x.shape.last <= 32)
      return launch(
        kernel(normKernels, "rms_norm_narrow_f32"),
        (rows + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(pointer(weight)),
        Pointer(pointer(out)),
        I64(rows),
        I32(x.shape.last.toInt),
        F32(epsilon),
        F32(weightOffset)
      )
    normRows(x, None, weight, epsilon, weightOffset, out, rows)
  }

  /** rms_norm_f32 over `rows`, `y` added to x first when given, the result
    * quantized too when int8 products may read it. A long row (the residual
    * stream) is latency-bound: more threads, for every row count, so a row's
    * sum does not depend on how many rows come with it (a verification's tokens
    * norm as a decode step's token does).
    */
  private def normRows(
      x: Tensor,
      y: Option[Tensor],
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor,
      rows: Long
  ): Unit = {
    val cols = x.shape.last
    val (target, quantized) = quantizedTarget(out, cols)
    launch(
      kernel(normKernels, "rms_norm_f32"),
      rows,
      if (cols >= 1024) 1024 else 256,
      (Seq(
        Pointer(pointer(x)),
        Pointer(y.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(weight)),
        Pointer(pointer(out)),
        I32(cols.toInt),
        F32(epsilon),
        F32(weightOffset),
        I32(1)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }

  override def addRmsNorm(x: Tensor, y: Tensor, norm: RowNorm): Unit = {
    Ops.checkElementwise("addRmsNorm", x, y)
    val rows = Ops.checkRowWise("addRmsNorm", x, norm.out, norm.weight)
    if (x.shape.last <= 32) super.addRmsNorm(x, y, norm)
    else
      normRows(
        x,
        Some(y),
        norm.weight,
        norm.epsilon,
        norm.weightOffset,
        norm.out,
        rows
      )
  }

  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("groupRmsNorm", x, out, weight)
    require(x.shape.last % groups == 0, s"${x.shape} in $groups groups")
    rowWise(
      "rms_norm_f32",
      rows * groups,
      Pointer(pointer(x)),
      Pointer(MemorySegment.NULL),
      Pointer(pointer(weight)),
      Pointer(pointer(out)),
      I32((x.shape.last / groups).toInt),
      F32(epsilon),
      F32(weightOffset),
      I32(groups),
      Pointer(MemorySegment.NULL),
      Pointer(MemorySegment.NULL),
      Pointer(MemorySegment.NULL)
    )
  }

  def modulate(x: Tensor, scale: Tensor, shift: Tensor, out: Tensor): Unit = {
    Ops.checkRowWise("modulate", x, out, scale, shift)
    elementwise(
      "modulate_f32",
      x.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(scale)),
      Pointer(pointer(shift)),
      Pointer(pointer(out)),
      I64(x.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def gatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    Ops.checkRowWise("gatedAdd", x, y, gate)
    elementwise(
      "gated_add_f32",
      x.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(gate)),
      I64(x.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor]
  ): Unit = {
    val (s, b, heads, kvHeads, d) = Ops.checkShortAttention(q, k, v, out)
    bias.foreach(b => Ops.checkAttentionBias(b, heads, s))
    val waves = s.toLong * b * heads
    launch(
      kernel(attentionKernels, "short_attention_f32"),
      (waves + 7) / 8,
      256,
      Pointer(pointer(q)),
      Pointer(pointer(k)),
      Pointer(pointer(v)),
      Pointer(pointer(out)),
      I32(s),
      I32(b),
      I32(heads),
      I32(kvHeads),
      I32(d),
      F32(scale),
      Pointer(bias.fold(MemorySegment.NULL)(pointer))
    )
  }

  private val imageKernels = new KernelModule(hip, "image")

  /** im2col's patches of this many bytes at most per GEMM. */
  private val PatchBudget = 256L << 20

  def conv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int,
      replicate: Boolean,
      reflect: Boolean
  ): Unit = {
    val (height, width, in, outChannels) =
      Ops.checkConv3x3(x, weight, bias, out, stride)
    require(
      weight.dtype == DType.BF16 || weight.dtype == DType.F16,
      s"conv3x3 takes BF16 or F16 weights, not ${weight.dtype}"
    )
    val half = weight.dtype == DType.F16
    val pixels = (height / stride).toLong * (width / stride)
    val columns = in * 9L
    val chunk = math.max(1L, math.min(pixels, PatchBudget / (2 * columns)))
    val patches = scratch(0, 2 * chunk * columns + 2 * chunk * outChannels)
    val result = offset(patches, 2 * chunk * columns)
    (0L until pixels by chunk).foreach { first =>
      val count = math.min(chunk, pixels - first)
      launch(
        kernel(imageKernels, if (half) "im2col_3x3_f16" else "im2col_3x3_bf16"),
        (count * columns + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(patches),
        I32(height),
        I32(width),
        I32(in),
        I32(width / stride),
        I32(stride),
        I32(if (replicate) 1 else if (reflect) 2 else 0),
        I64(first),
        I64(count)
      )
      val rows = out.view(pixels, outChannels.toLong).rows(first, count)
      if (half)
        // F16 operands, F32 sums: no rounding of the output
        blas.gemm(
          patches,
          pointer(weight),
          pointer(rows),
          count.toInt,
          outChannels,
          columns.toInt,
          HipBlas.RealF16,
          HipBlas.RealF32
        )
      else {
        blas.gemm(
          patches,
          pointer(weight),
          result,
          count.toInt,
          outChannels,
          columns.toInt,
          HipBlas.RealBF16
        )
        convertAt(result, pointer(rows), count * outChannels, 3)
      }
      addRow(rows, bias, rows)
    }
  }

  private val audioKernels = new KernelModule(hip, "audio")

  /** BF16 weights take BF16 patches, as `conv3x3`'s; F32 ones F32 patches. The
    * sums are F32 either way: a waveform does not take BF16's rounding at every
    * layer.
    */
  def conv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): Unit = {
    val (length, in, outLength, outChannels) = Ops.checkConv1d(
      x,
      weight,
      bias,
      taps,
      dilation,
      stride,
      padLeft,
      padRight,
      out
    )
    val (patchBytes, im2col, dataType) = weight.dtype match {
      case DType.BF16 => (2, "im2col_1d_bf16", HipBlas.RealBF16)
      case DType.F32  => (4, "im2col_1d_f32", HipBlas.RealF32)
      case other      =>
        throw new UnsupportedOperationException(
          s"conv1d takes BF16 or F32 weights, not $other"
        )
    }
    val columns = in.toLong * taps
    val chunk = math.max(
      1L,
      math.min(outLength.toLong, PatchBudget / (patchBytes * columns))
    )
    val patches = scratch(0, patchBytes * chunk * columns)
    (0L until outLength by chunk).foreach { first =>
      val count = math.min(chunk, outLength - first)
      launch(
        kernel(audioKernels, im2col),
        (count * columns + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(patches),
        I32(length),
        I32(in),
        I32(taps),
        I32(dilation),
        I32(stride),
        I32(padLeft),
        I64(first),
        I64(count)
      )
      val rows = out.rows(first, count)
      blas.gemm(
        patches,
        pointer(weight),
        pointer(rows),
        count.toInt,
        outChannels,
        columns.toInt,
        dataType,
        HipBlas.RealF32
      )
      bias.foreach(b => addRow(rows, b, rows))
    }
  }

  def overlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): Unit = {
    val (length, outLength, outChannels) =
      Ops.checkOverlapAdd(columns, taps, stride, pad, bias, out)
    launch(
      kernel(audioKernels, "overlap_add_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(columns)),
      Pointer(pointer(bias)),
      Pointer(pointer(out)),
      I32(length),
      I32(outLength),
      I32(outChannels),
      I32(taps),
      I32(stride),
      I32(pad)
    )
  }

  def snake(x: Tensor, alpha: Tensor, out: Tensor): Unit = {
    val (length, channels) = Ops.checkSnake(x, alpha, out)
    launch(
      kernel(audioKernels, "snake_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(alpha)),
      Pointer(pointer(out)),
      I64(length.toLong * channels),
      I32(channels)
    )
  }

  def antiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): Unit = {
    val (length, channels) = Ops.checkAntiAliasedSnake(
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    launch(
      kernel(audioKernels, "anti_aliased_snake_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(frequency)),
      Pointer(pointer(inverseMagnitude)),
      Pointer(pointer(upFilter)),
      Pointer(pointer(downFilter)),
      Pointer(pointer(out)),
      I32(length),
      I32(channels)
    )
  }

  /** Rows each block of a group norm's partial sums covers. */
  private val GroupRowsPerBlock = 1024

  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val (rows, channels) = Ops.checkGroupNorm(x, groups, weight, bias, out)
    val blocks = ((rows + GroupRowsPerBlock - 1) / GroupRowsPerBlock).toInt
    val partials = scratch(0, 4L * groups * blocks)
    val stats = scratch(4L * groups * blocks, 8L * groups)
    Seq(MemorySegment.NULL, stats).zipWithIndex.foreach { (center, mode) =>
      launch(
        kernel(normKernels, "group_partials_f32"),
        groups.toLong * blocks,
        256,
        Pointer(pointer(x)),
        Pointer(center),
        Pointer(partials),
        I32(rows.toInt),
        I32(channels),
        I32(groups),
        I32(blocks),
        I32(GroupRowsPerBlock)
      )
      launch(
        kernel(normKernels, "group_finish_f32"),
        groups,
        256,
        Pointer(partials),
        Pointer(stats),
        I32(groups),
        I32(blocks),
        I64(rows * (channels / groups)),
        F32(epsilon),
        I32(mode)
      )
    }
    launch(
      kernel(normKernels, "group_apply_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(stats),
      Pointer(pointer(weight)),
      Pointer(pointer(bias)),
      Pointer(pointer(out)),
      I64(x.shape.elementCount),
      I32(channels),
      I32(groups)
    )
  }

  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean
  ): Unit = {
    val (tokens, heads, d, pairs, perHead) =
      Ops.checkRopeTable(x, cosines, sines, out)
    launch(
      kernel(
        ropeKernels,
        if (halves) "rope_table_halves_f32" else "rope_table_f32"
      ),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(cosines)),
      Pointer(pointer(sines)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(heads),
      I32(d),
      I32(pairs),
      I32(if (perHead) 1 else 0)
    )
  }

  private def patchPixels(
      from: Tensor,
      to: Tensor,
      image: Tensor,
      patch: Int,
      back: Boolean
  ): Unit = {
    val (height, width, channels) =
      Ops.checkPatchPixels(image, if (back) from else to, patch)
    launch(
      kernel(imageKernels, "patch_pixels_f32"),
      (image.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(from)),
      Pointer(pointer(to)),
      I32(height),
      I32(width),
      I32(channels),
      I32(patch),
      I32(if (back) 1 else 0)
    )
  }

  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit =
    patchPixels(image, out, image, patch, back = false)

  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit =
    patchPixels(rows, out, out, patch, back = true)

  def modulateChunks(
      x: Tensor,
      table: Tensor,
      shift: Int,
      scale: Int,
      out: Tensor
  ): Unit = {
    val (rows, channels, chunks) =
      Ops.checkChunks("modulateChunks", x, table, out, shift, scale)
    launch(
      kernel(elementwiseKernels, "modulate_chunks_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(table)),
      Pointer(pointer(out)),
      I64(rows),
      I32(channels),
      I32(chunks),
      I32(shift),
      I32(scale)
    )
  }

  def gatedAddChunk(x: Tensor, y: Tensor, table: Tensor, gate: Int): Unit = {
    val (rows, channels, chunks) =
      Ops.checkChunks("gatedAddChunk", x, table, y, gate)
    launch(
      kernel(elementwiseKernels, "gated_add_chunk_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(table)),
      I64(rows),
      I32(channels),
      I32(chunks),
      I32(gate)
    )
  }

  def rowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    val (rows, cols) = Ops.checkRowGatedAdd(x, y, gate)
    launch(
      kernel(elementwiseKernels, "row_gated_add_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(gate)),
      I64(rows),
      I32(cols)
    )
  }

  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (gridHeight, gridWidth, channels) =
      Ops.checkPackPatches(image, patch, out)
    launch(
      kernel(imageKernels, "pack_patches_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(image)),
      Pointer(pointer(out)),
      I32(gridHeight),
      I32(gridWidth),
      I32(channels),
      I32(patch)
    )
  }

  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit = {
    val (rows, columns) = Ops.checkConcatColumns(parts, out)
    parts.zip(columns).zip(columns.scanLeft(0)(_ + _)).foreach {
      case ((part, width), start) =>
        launch(
          kernel(imageKernels, "copy_columns_f32"),
          (rows * width + 255) / 256,
          256,
          Pointer(pointer(part)),
          Pointer(pointer(out)),
          I64(rows),
          I32(width),
          I32(columns.sum),
          I32(start)
        )
    }
  }

  def upsample2x(x: Tensor, out: Tensor): Unit = {
    val (height, width, channels) = Ops.checkUpsample2x(x, out)
    launch(
      kernel(imageKernels, "upsample_2x_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels)
    )
  }

  def duplicateUp(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (height, width, channels, outChannels) =
      Ops.checkDuplicateUp(x, frames, out)
    launch(
      kernel(imageKernels, "duplicate_up_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels),
      I32(outChannels),
      I32(frames)
    )
  }

  def averageDown(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (height, width, channels, outChannels) =
      Ops.checkAverageDown(x, frames, out)
    launch(
      kernel(imageKernels, "average_down_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels),
      I32(outChannels),
      I32(frames)
    )
  }

  def transpose(x: Tensor, out: Tensor): Unit = {
    val Seq(rows, cols) = Ops.checkTranspose(x, out)
    launch(
      kernel(imageKernels, "transpose_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(rows),
      I32(cols)
    )
  }

  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit = {
    val (gridWidth, channels) =
      Ops.checkUnpackPatches(packed, gridHeight, patch, out)
    launch(
      kernel(imageKernels, "unpack_patches_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(packed)),
      Pointer(pointer(out)),
      I32(gridHeight),
      I32(gridWidth),
      I32(channels),
      I32(patch)
    )
  }

  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("layerNorm", x, out, (weight ++ bias).toSeq*)
    def optional(tensor: Option[Tensor]) = Pointer(
      tensor.fold(MemorySegment.NULL)(pointer)
    )
    rowWise(
      "layer_norm_f32",
      rows,
      Pointer(pointer(x)),
      optional(weight),
      optional(bias),
      Pointer(pointer(out)),
      I32(x.shape.last.toInt),
      F32(epsilon)
    )
  }

  def softmax(x: Tensor, scale: Float, out: Tensor): Unit = {
    val rows = Ops.checkRowWise("softmax", x, out)
    rowWise(
      "softmax_f32",
      rows,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(x.shape.last.toInt),
      F32(scale)
    )
  }

  /** `rope_f32`'s four section sizes. */
  private def ropeSections(rope: Rope): Seq[Int] = rope.sections match {
    case RopeSections.Single               => Seq(0, 0, 0, 0)
    case RopeSections.Contiguous(t, h, w)  => Seq(t, h, w, 0)
    case RopeSections.Interleaved(t, h, w) => Seq(t, h, w, 0)
    case RopeSections.Axes(pairs)          => pairs.padTo(4, 0)
  }

  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit = {
    val (tokens, heads, dimension) = Ops.checkRope(x, positions, rope, out)
    val Seq(s0, s1, s2, s3) = ropeSections(rope)
    val count = x.shape.elementCount
    launch(
      kernel(ropeKernels, "rope_f32"),
      (count + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(positions)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(heads),
      I32(dimension),
      I32(rope.rotaryDimensions),
      F32(rope.theta),
      I32(rope.layout.code),
      I32(rope.sections.code),
      I32(s0),
      I32(s1),
      I32(s2),
      I32(s3)
    )
  }

  // ---- linear ------------------------------------------------------------------------

  /** Up to `MaxMatVecRows` rows of x: one matrix-vector pass reads the weight
    * once for all of them (`kernels/matvec.hip`), eight weight rows per
    * workgroup. More: dequantize to F16 and a hipBLAS GEMM.
    */
  def linear(x: Tensor, weight: Tensor, out: Tensor): Unit =
    linears(x, Seq(weight), Seq(out))

  /** One preparation of `x` serves all the matrix-vector products: with `Int8`
    * inputs, one quantization.
    */
  override def linears(
      x: Tensor,
      weights: Seq[Tensor],
      outs: Seq[Tensor]
  ): Unit = {
    val shapes =
      weights.zip(outs).map((weight, out) => Ops.checkLinear(x, weight, out))
    val (m, _, k) = shapes.head
    weights.foreach { weight =>
      if (!integerX.contains(weight.dtype))
        throw new UnsupportedOperationException(
          s"no kernels for ${weight.dtype} weights yet"
        )
    }
    require(k <= Int.MaxValue && m * k <= Int.MaxValue, s"linear over [$m, $k]")
    // BF16 rows not in whole blocks of 32 (the matrix-vector kernels' unit)
    // go to hipBLAS at any row count
    val gemmOnly = m >= GemmMinimumRows || weights.forall(
      _.dtype == DType.BF16
    ) && k % 32 != 0
    if (!gemmOnly)
      weights.foreach(weight =>
        require(
          weight.dtype.byteSize(k) % 4 == 0,
          s"${weight.dtype} rows of $k are not whole words"
        )
      )
    if (gemmOnly)
      weights.zip(shapes).zip(outs).foreach { case ((weight, (_, n, _)), out) =>
        gemm(x, weight, out, m.toInt, n.toInt, k.toInt)
      }
    else if (m > MaxMatVecRows)
      (0L until m by MaxMatVecRows.toLong).foreach { first =>
        val count = math.min(MaxMatVecRows.toLong, m - first)
        linears(x.rows(first, count), weights, outs.map(_.rows(first, count)))
      }
    else {
      lazy val quantized =
        quantizedInput(x, m, k).getOrElse(quantize(x, m.toInt, k.toInt))
      val products =
        weights.lazyZip(outs).lazyZip(shapes).map((w, o, s) => (w, o, s._2))
      // float x: the products of each type share launches, four at a time
      val merged =
        if (inputs == MatVecInputs.Float)
          products
            .filter((weight, _, _) => integerX(weight.dtype))
            .groupBy(_._1.dtype)
            .values
            .filter(_.size > 1)
            .toSeq
        else Seq.empty
      merged.foreach(_.grouped(4).foreach(matVecs(x, quantized, _, m, k)))
      val alone = products.filterNot(p => merged.exists(_.exists(_ eq p)))
      alone.foreach { (weight, out, n) =>
        val name = weight.dtype.name.toLowerCase
        require(
          integerX(weight.dtype) || k % 32 == 0 ||
            (n <= SplitMaximumRows && weight.dtype == DType.F32 && k % 4 == 0),
          s"${weight.dtype} rows of $k: the dense kernels take whole blocks of 32"
        )
        val common = Seq(I32(n.toInt), I32(k.toInt), I32(m.toInt))
        if (rocmFp4Int8(weight.dtype, k) && n > SplitMaximumRows) {
          val (xq, xd, _) = quantized
          // rows of fewer than eight blocks of 256 are walked several at once
          val blocks = k / DirectBlockElements
          val group =
            if (8 % blocks == 0 && n % (8 / blocks) == 0) 8 / blocks else 1L
          launch(
            kernel(
              matvecKernels,
              if (group == 1) s"matvec_int8_$name"
              else s"matvec_int8_group${group}_$name"
            ),
            (n + 8 * Int8Rows * group - 1) / (8 * Int8Rows * group),
            256,
            (Seq(
              Pointer(pointer(weight)),
              Pointer(xq),
              Pointer(xd),
              Pointer(pointer(out))
            ) ++ common)*
          )
        } else if (integerX(weight.dtype) && inputs == MatVecInputs.Int8) {
          val (xq, xd, xs) = quantized
          launch(
            kernel(matvecKernels, s"matvec_$name"),
            (n + 7) / 8,
            256,
            (Seq(
              Pointer(pointer(weight)),
              Pointer(xq),
              Pointer(xd),
              Pointer(xs),
              Pointer(pointer(out))
            ) ++ common)*
          )
        }
        // Each row of the float-x kernels sums in an order that depends on the
        // matrix, never on m: a verification's tokens then get exactly a decode
        // step's logits, and drafting leaves greedy output as it was.
        else if (m > 1 && direct(weight.dtype, k) && n > SplitMaximumRows)
          // several tokens (a verification): the weights read once for all
          launch(
            kernel(matvecKernels, s"matvec_direct_vectors_$name"),
            (n + 15) / 16,
            256,
            Pointer(pointer(weight)),
            Pointer(pointer(x)),
            Pointer(pointer(out)),
            I32(n.toInt),
            I32(k.toInt),
            I32(m.toInt)
          )
        else if (m == 1 && direct(weight.dtype, k) && n > SplitMaximumRows)
          // read straight into registers, four rows per wave
          launch(
            kernel(matvecKernels, s"matvec_direct_$name"),
            (n + 31) / 32,
            256,
            Pointer(pointer(weight)),
            Pointer(pointer(x)),
            Pointer(pointer(out)),
            I32(n.toInt),
            I32(k.toInt)
          )
        else if (
          n <= SplitMaximumRows && weight.dtype == DType.F32 && k % 4 == 0
        )
          // F32 routers: float4 loads straight from memory, a workgroup per
          // four rows
          launch(
            kernel(matvecKernels, "matvec_rows_f32"),
            (n + 3) / 4, // F32_ROWS rows a workgroup
            256,
            Pointer(pointer(weight)),
            Pointer(pointer(x)),
            Pointer(pointer(out)),
            I32(n.toInt),
            I32(k.toInt),
            I32(m.toInt)
          )
        else if (n <= SplitMaximumRows)
          // few rows: the workgroup's eight waves share a (row, vector)
          launch(
            kernel(matvecKernels, s"matvec_split_$name"),
            Dim3(n.toInt, m.toInt, 1),
            256,
            Pointer(pointer(weight)),
            Pointer(pointer(x)),
            Pointer(pointer(out)),
            I32(n.toInt),
            I32(k.toInt)
          )
        else {
          val size = narrow(weight.dtype, k)
          val function =
            if (integerX(weight.dtype)) s"matvec_float_x${size}_$name"
            else s"matvec${size}_$name"
          launch(
            kernel(matvecKernels, function),
            (n + 7) / 8,
            256,
            (Seq(
              Pointer(pointer(weight)),
              Pointer(pointer(x)),
              Pointer(pointer(out))
            ) ++ common)*
          )
        }
      }
    }
  }

  /** Products of weights of one type (at most four) over the same float `x`, in
    * one launch; ROCmFP4's over `x` quantized once for all.
    */
  private def matVecs(
      x: Tensor,
      quantized: => (MemorySegment, MemorySegment, MemorySegment),
      products: Seq[(Tensor, Tensor, Long)],
      m: Long,
      k: Long
  ): Unit = {
    val dtype = products.head._1.dtype
    val directly = direct(dtype, k)
    val matrices = products
      .map((weight, out, n) =>
        Seq(Pointer(pointer(weight)), Pointer(pointer(out)), I32(n.toInt))
      )
      .padTo(
        4,
        Seq(Pointer(MemorySegment.NULL), Pointer(MemorySegment.NULL), I32(0))
      )
    if (rocmFp4Int8(dtype, k)) {
      val (xq, xd, _) = quantized
      launch(
        kernel(matvecKernels, s"matvec_int8_multi_${dtype.name.toLowerCase}"),
        (products.map((_, _, n) => (n + Int8Rows - 1) / Int8Rows).sum + 7) / 8,
        256,
        (matrices.flatten ++ Seq(
          Pointer(xq),
          Pointer(xd),
          I32(k.toInt),
          I32(m.toInt)
        ))*
      )
    } else if (directly && m == 1)
      launch(
        kernel(matvecKernels, s"matvec_direct_multi_${dtype.name.toLowerCase}"),
        (products.map((_, _, n) => (n + 3) / 4).sum + 7) / 8,
        256,
        (matrices.flatten ++ Seq(Pointer(pointer(x)), I32(k.toInt)))*
      )
    else if (directly)
      // several tokens (a verification): the weights read once for all
      launch(
        kernel(
          matvecKernels,
          s"matvec_direct_multi_vectors_${dtype.name.toLowerCase}"
        ),
        (products.map((_, _, n) => (n + 1) / 2).sum + 7) / 8,
        256,
        (matrices.flatten ++ Seq(
          Pointer(pointer(x)),
          I32(k.toInt),
          I32(m.toInt)
        ))*
      )
    else
      launch(
        kernel(
          matvecKernels,
          s"matvec_float_x_multi${narrow(dtype, k)}_${dtype.name.toLowerCase}"
        ),
        (products.map(_._3).sum + 7) / 8,
        256,
        (matrices.flatten ++ Seq(
          Pointer(pointer(x)),
          I32(k.toInt),
          I32(m.toInt)
        ))*
      )
  }

  /** Rows a wave of `matvec.hip`'s int8 products takes (INT8_ROWS). */
  private val Int8Rows = 2

  /** Single-vector products of at most this many rows split each row over a
    * workgroup's eight waves: one wave per row would leave most of the GPU idle
    * (Qwen 3.6's router, 256 rows, ran at 78 GB/s).
    */
  val SplitMaximumRows = 1024

  /** "_narrow" for rows short enough that the 1 KB-trip kernels serve them
    * better: the 4 KB ones reserve LDS that caps how many workgroups a compute
    * unit holds, and short rows are latency-bound.
    */
  private def narrow(dtype: DType, columns: Long): String =
    if (dtype.byteSize(columns) <= 1536) "_narrow" else ""

  /** `x` quantized per 32 values into scratch memory: codes, scales, sums. */
  private def quantize(
      x: Tensor,
      m: Int,
      k: Int
  ): (MemorySegment, MemorySegment, MemorySegment) = {
    quantizesX = true
    val blocks = m.toLong * k / 32
    val xq = scratch(0, m.toLong * k + 8 * blocks)
    val (xd, xs) =
      (offset(xq, m.toLong * k), offset(xq, m.toLong * k + 4 * blocks))
    launch(
      kernel(matvecKernels, "quantize_x"),
      (blocks + 7) / 8,
      256,
      Pointer(pointer(x)),
      Pointer(xq),
      Pointer(xd),
      Pointer(xs),
      I32(blocks.toInt)
    )
    (xq, xd, xs)
  }

  private def gemm(
      x: Tensor,
      weight: Tensor,
      out: Tensor,
      m: Int,
      n: Int,
      k: Int
  ): Unit = {
    if (weight.dtype == DType.BF16) {
      // BF16 weights as stored: x and the result in BF16 around them
      val xBf16 = scratch(0, 2L * m * k + 2L * m * n)
      val outBf16 = offset(xBf16, 2L * m * k)
      convertAt(pointer(x), xBf16, m.toLong * k, 1)
      blas.gemm(xBf16, pointer(weight), outBf16, m, n, k, HipBlas.RealBF16)
      convertAt(outBf16, pointer(out), m.toLong * n, 3)
    } else halfGemm(x, weight, out, m, n, k)
  }

  /** Weights dequantized to F16 (unless stored so), x and the result in F16;
    * with `wideProducts`, quantized weights to BF16 and all three in BF16.
    */
  private def halfGemm(
      x: Tensor,
      weight: Tensor,
      out: Tensor,
      m: Int,
      n: Int,
      k: Int
  ): Unit = {
    val wide = wideProducts && weight.dtype != DType.F16
    val weightBytes = 2L * n * k
    val xHalf = scratch(0, weightBytes + 2L * m * k + 2L * m * n)
    val outHalf = offset(xHalf, 2L * m * k + weightBytes)
    val weightHalf =
      if (weight.dtype == DType.F16) pointer(weight)
      else if (weight.dtype == DType.F32) {
        // dense F32 element by element: the dequantizers walk rows in blocks
        // of 32, which rows such as the H3 ControlNet's 196 are not
        val target = offset(xHalf, 2L * m * k)
        convertAt(
          pointer(weight),
          target,
          n.toLong * k,
          conversionCodes(DType.F32 -> (if (wide) DType.BF16 else DType.F16))
        )
        target
      } else {
        val target = offset(xHalf, 2L * m * k)
        val precision = if (wide) "bf16_" else ""
        launch(
          kernel(
            matvecKernels,
            s"dequantize_$precision${weight.dtype.name.toLowerCase}"
          ),
          (n + 7L) / 8,
          256,
          Pointer(pointer(weight)),
          Pointer(target),
          I32(n),
          I32(k)
        )
        target
      }
    if (wide) {
      convertAt(pointer(x), xHalf, m.toLong * k, 1)
      blas.gemm(xHalf, weightHalf, outHalf, m, n, k, HipBlas.RealBF16)
      convertAt(outHalf, pointer(out), m.toLong * n, 3)
    } else {
      convertAt(pointer(x), xHalf, m.toLong * k, 0)
      blas.gemm(xHalf, weightHalf, outHalf, m, n, k)
      convertAt(outHalf, pointer(out), m.toLong * n, 2)
    }
  }

  // ---- attention ---------------------------------------------------------------------

  private val attentionKernels = new KernelModule(hip, "attention")
  private val linearAttentionKernels = new KernelModule(hip, "linear_attention")

  def zero(tensor: Tensor): Unit = {
    hip.zero(pointer(tensor), tensor.byteSize)
    deviceWrites += 1
  }

  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val count = first.shape.elementCount
    val d = first.shape.last
    launch(
      kernel(linearAttentionKernels, "split_halves"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(x)),
      Pointer(pointer(first)),
      Pointer(pointer(second)),
      I64(count / d),
      I32(d.toInt)
    )
  }

  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val count = first.shape.elementCount
    val d = first.shape.last
    launch(
      kernel(linearAttentionKernels, "join_halves"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(first)),
      Pointer(pointer(second)),
      Pointer(pointer(x)),
      I64(count / d),
      I32(d.toInt)
    )
  }

  def copy(from: Tensor, to: Tensor): Unit = {
    Ops.checkCopy(from, to)
    hip.copyAsync(pointer(to), pointer(from), from.byteSize)
    deviceWrites += 1
  }

  def causalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val (tokens, channels, taps) =
      Ops.checkCausalConv(x, weight, dilation, state, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    launch(
      kernel(linearAttentionKernels, "causal_conv_silu"),
      (channels + 255L) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(weight)),
      Pointer(pointer(state)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(channels),
      I32(taps),
      I32(dilation),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
  }

  def gatedDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val tokens = Ops.checkDeltaRule(qkv, a, b, decay, dtBias, state, rule, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    require(
      Set(32, 64, 128).contains(rule.dimension),
      s"no delta-rule kernel for heads of ${rule.dimension}"
    )
    launch(
      kernel(linearAttentionKernels, s"delta_rule_d${rule.dimension}"),
      rule.valueHeads.toLong * rule.dimension / 32, // COLUMNS per workgroup
      128,
      Pointer(pointer(qkv)),
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(decay)),
      Pointer(pointer(dtBias)),
      Pointer(pointer(state)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(rule.keyHeads),
      I32(rule.valueHeads),
      I32(if (rule.tiledHeads) 1 else 0),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
  }

  override def convolvedDeltaRule(
      qkv: Tensor,
      convWeight: Tensor,
      convState: Tensor,
      convHistory: Option[Tensor],
      convolved: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val (tokens, _, taps) =
      Ops.checkCausalConv(qkv, convWeight, 1, convState, convolved)
    if (taps != 4 || rule.dimension != 128 || rule.keyHeads > 128)
      super.convolvedDeltaRule(
        qkv,
        convWeight,
        convState,
        convHistory,
        convolved,
        a,
        b,
        decay,
        dtBias,
        state,
        rule,
        history,
        out
      )
    else {
      Ops.checkDeltaRule(qkv, a, b, decay, dtBias, state, rule, out)
      convHistory.foreach(Ops.checkHistory(_, tokens, convState))
      history.foreach(Ops.checkHistory(_, tokens, state))
      launch(
        kernel(linearAttentionKernels, s"conv_delta_rule_d${rule.dimension}"),
        rule.valueHeads.toLong * rule.dimension / 32, // COLUMNS per workgroup
        128,
        Pointer(pointer(qkv)),
        Pointer(pointer(convWeight)),
        Pointer(pointer(convState)),
        Pointer(convHistory.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(a)),
        Pointer(pointer(b)),
        Pointer(pointer(decay)),
        Pointer(pointer(dtBias)),
        Pointer(pointer(state)),
        Pointer(pointer(out)),
        I32(tokens),
        I32(rule.keyHeads),
        I32(rule.valueHeads),
        I32(if (rule.tiledHeads) 1 else 0),
        Pointer(history.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(deltaCounters))
      )
    }
  }

  /** `conv_delta_rule`'s counters: zero between launches. */
  private lazy val deltaCounters = {
    val counters = allocate(DType.I32, Shape.of(256))
    zero(counters)
    counters
  }

  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit = {
    val (tokens, experts, k) = Ops.checkRoute(logits, ids, weights)
    launch(
      kernel(linearAttentionKernels, "route_topk"),
      (tokens + 7L) / 8,
      256,
      Pointer(pointer(logits)),
      Pointer(pointer(ids)),
      Pointer(pointer(weights)),
      I32(tokens),
      I32(experts),
      I32(k)
    )
  }

  /** Counts the workgroups of matvec_rows_f32_route that finished; the last
    * leaves it at 0.
    */
  private lazy val finishedRouters = {
    val counter = allocate(DType.I32, Shape.of(1))
    zero(counter)
    counter
  }

  override def routeLinear(
      x: Tensor,
      router: Tensor,
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): Unit = {
    val (m, n, k) = Ops.checkLinear(x, router, logits)
    val (_, _, used) = Ops.checkRoute(logits, ids, weights)
    // an F32 router of few rows, a verification's vectors at most: the last
    // workgroup routes, one wave per vector
    if (
      router.dtype == DType.F32 && n <= SplitMaximumRows && k % 4 == 0 &&
      m <= MaxMatVecRows
    )
      launch(
        kernel(matvecKernels, "matvec_rows_f32_route"),
        (n + 3) / 4, // F32_ROWS rows a workgroup
        256,
        Pointer(pointer(router)),
        Pointer(pointer(x)),
        Pointer(pointer(logits)),
        I32(n.toInt),
        I32(k.toInt),
        I32(m.toInt),
        Pointer(pointer(ids)),
        Pointer(pointer(weights)),
        I32(used),
        Pointer(pointer(finishedRouters))
      )
    else super.routeLinear(x, router, logits, ids, weights)
  }

  def expertsLinear(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, weight, ids, out)
    require(
      integerX
        .contains(weight.dtype) && weight.dtype.byteSize(k.toLong) % 4 == 0,
      s"no expert kernel for ${weight.dtype} rows of $k"
    )
    lazy val directGroup = directRows(k)
    if (
      slots >= GroupedExpertsMinimumSlots && matrixExperts(weight.dtype, n, k)
    )
      expertsMatrix(x, weight, None, ids, out, slots, n, k, divisor)
    else if (slots >= GroupedExpertsMinimumSlots) {
      val group = rowsPerWave(weight.dtype, n, k)
      val (order, batches, maxBatches) =
        groupSlots(ids, slots, weight.shape.dimensions.head.toInt, 8)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_grouped_${weight.dtype.name.toLowerCase}"
        ),
        (maxBatches.toLong * (n / group) + 7) / 8,
        256,
        Pointer(pointer(weight)),
        I64(weight.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(order),
        Pointer(batches),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(maxBatches),
        I32(group)
      )
    } else if (direct(weight.dtype, k) && n % (directGroup * 2) == 0)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_direct_${weight.dtype.name.toLowerCase}"
        ),
        (slots.toLong * n / (directGroup * 2) + 7) / 8,
        256,
        Pointer(pointer(weight)),
        I64(weight.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(slots),
        I32(directGroup)
      )
    else expertsLinearStaged(x, weight, ids, out, slots, n, k, divisor)
  }

  private def expertsLinearStaged(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int
  ): Unit = {
    val group = rowsPerWave(weight.dtype, n, k)
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts${narrow(weight.dtype, k.toLong * group)}_${weight.dtype.name.toLowerCase}"
      ),
      (slots.toLong * n / group + 7) / 8,
      256,
      Pointer(pointer(weight)),
      I64(weight.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(slots),
      I32(group)
    )
  }

  /** The slots of `ids` grouped by expert (`group_slots`) in scratch memory:
    * their order, and batches of up to 8 slots of one expert, room for as many
    * batches as there can be.
    */
  /** A prompt's ROCmFP4 experts on the matrix units (`matvec.hip`'s
    * `experts_matrix`): slots grouped by expert in batches of 16, tiles of 16
    * rows.
    */
  private def matrixExperts(dtype: DType, n: Int, k: Int): Boolean =
    (dtype == RocmFp4.Fast || dtype == RocmFp4.Dual) &&
      k % DirectBlockElements == 0 && n % 16 == 0

  private def expertsMatrix(
      x: Tensor,
      gate: Tensor,
      up: Option[Tensor],
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int
  ): Unit = {
    val (order, batches, maxBatches) =
      groupSlots(ids, slots, gate.shape.dimensions.head.toInt, 16)
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts_matrix_${gate.dtype.name.toLowerCase}"
      ),
      (maxBatches.toLong * (n / 16) + 7) / 8,
      256,
      Pointer(pointer(gate)),
      Pointer(up.fold(MemorySegment.NULL)(pointer)),
      I64(gate.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(order),
      Pointer(batches),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(maxBatches)
    )
  }

  private def groupSlots(
      ids: Tensor,
      slots: Int,
      experts: Int,
      batchSize: Int
  ): (MemorySegment, MemorySegment, Int) = {
    require(experts <= 1024, s"$experts experts; grouping takes 1024 at most")
    val maxBatches = slots / batchSize + math.min(experts, slots)
    val order = scratch(0, 4L * slots + 8L * maxBatches)
    val batches = offset(order, 4L * slots)
    launch(
      kernel(matvecKernels, "group_slots"),
      1,
      1024,
      Pointer(pointer(ids)),
      I32(slots),
      I32(experts),
      Pointer(order),
      Pointer(batches),
      I32(maxBatches),
      I32(batchSize)
    )
    (order, batches, maxBatches)
  }

  /** Types whose rows of `k` the `direct` kernels read straight into registers,
    * no LDS staging: blocks of 256 that keep lanes' bytes aligned (Q4_K and
    * Q5_K to 16 bytes, IQ4_XS to 8), and ROCmFP4's blocks of 32 eight at a time
    * (to 2 bytes; a lane loads from the word its bytes start in), in rows of
    * whole such eights.
    */
  private def direct(dtype: DType, k: Long): Boolean =
    Set(
      KQuants.Q4_K,
      KQuants.Q5_K,
      IQuants.IQ4_XS,
      RocmFp4.Dual,
      RocmFp4.Fast
    ).contains(dtype) && k % DirectBlockElements == 0

  /** ROCmFP4 products that meet x quantized to int8 whatever the inputs
    * (`matvec.hip`'s `rocmfp4_int8_rows`): rows of whole 256s, as the fork's
    * MMVQ multiplies them. A verification's vectors then cost little more than
    * one, where float x makes every vector pay the conversions and products.
    */
  private def rocmFp4Int8(dtype: DType, k: Long): Boolean =
    (dtype == RocmFp4.Dual || dtype == RocmFp4.Fast) &&
      k % DirectBlockElements == 0

  /** What a `direct` kernel takes as one block. */
  private val DirectBlockElements = 256

  /** How many consecutive rows of `k` columns a `direct` kernel walks as one: a
    * pass takes eight blocks, so rows of fewer that divide eight go several at
    * once.
    */
  private def directRows(k: Int): Int = {
    val blocks = k / DirectBlockElements
    if (blocks < 8 && 8 % blocks == 0) 8 / blocks else 1
  }

  /** How many consecutive rows of `n` share a wave in `matvec_experts`: a wave
    * decodes a chunk of 32 lanes' parts at once (32 values each), so rows of
    * fewer blocks than a chunk, which divide it, are walked several at a time.
    */
  private def rowsPerWave(dtype: DType, n: Int, k: Int): Int = {
    val chunkBlocks = 1024 / dtype.blockElements
    val rowBlocks = k / dtype.blockElements
    val group = chunkBlocks / rowBlocks
    if (
      rowBlocks < chunkBlocks && chunkBlocks % rowBlocks == 0 && n % group == 0
    )
      group
    else 1
  }

  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit = {
    Ops.requireF32(
      "moeCombine",
      (Seq(expertOut, weights, out) ++ residual ++ shared.toSeq.flatMap(e =>
        Seq(e.output, e.input, e.router)
      ))*
    )
    val Seq(tokens, hidden) = out.shape.dimensions
    val k = weights.shape.elementCount / tokens
    def optional(tensor: SharedExpert => Tensor) = Pointer(
      shared.fold(MemorySegment.NULL)(e => pointer(tensor(e)))
    )
    launch(
      kernel(linearAttentionKernels, "moe_combine"),
      tokens * 8, // SLICES in the kernel
      256,
      Pointer(pointer(expertOut)),
      Pointer(pointer(weights)),
      optional(_.output),
      optional(_.input),
      optional(_.router),
      Pointer(residual.fold(MemorySegment.NULL)(pointer)),
      Pointer(pointer(out)),
      I32(k.toInt),
      I32(hidden.toInt)
    )
  }

  /** matvec.hip's `moe` counters: zero between launches. */
  private lazy val moeCounters = {
    val counters = allocate(DType.I32, Shape.of(512))
    zero(counters)
    counters
  }

  /** ROCmFP4 experts of a decode step: the routing, then the rest in one launch
    * (`matvec.hip`'s `moe_rocmfp4`), which computes every part as the separate
    * launches do, where these would take the same kernels. A verification's
    * tokens take the separate launches, which measured faster for them.
    */
  override def expertOutputs(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Unit =
    if (!mixtureInOneLaunch(x, weights, into))
      super.expertOutputs(x, weights, into)
    else mixture(x, weights, into, None)

  override def mixtureNorm(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      residual: Tensor,
      norm: RowNorm
  ): Unit =
    if (!mixtureInOneLaunch(x, weights, into) || residual.shape.last < 1024)
      super.mixtureNorm(x, weights, into, residual, norm)
    else {
      Ops.requireF32("mixtureNorm", residual, norm.out, weights.sharedRouter)
      Ops.checkRowWise("mixtureNorm", residual, norm.out, norm.weight)
      mixture(x, weights, into, Some((residual, norm)))
    }

  /** Whether `moe_rocmfp4` computes these experts as the separate launches
    * would.
    */
  private def mixtureInOneLaunch(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Boolean = {
    val dtype = weights.gate.dtype
    val hidden = weights.router.shape.last
    val intermediate = weights.gate.shape.dimensions(1)
    (dtype == RocmFp4.Fast || dtype == RocmFp4.Dual) &&
    Seq(
      weights.up,
      weights.down,
      weights.sharedGate,
      weights.sharedUp,
      weights.sharedDown
    )
      .forall(_.dtype == dtype) &&
    x.shape.dimensions.head == 1 &&
    hidden % DirectBlockElements == 0 && intermediate % DirectBlockElements == 0 &&
    intermediate <= MoeSharedValues &&
    hidden % (Int8Rows * directRows(intermediate.toInt)) == 0 &&
    rocmFp4Int8(dtype, intermediate) && hidden > SplitMaximumRows &&
    weights.sharedGate.shape == Shape.of(1, intermediate, hidden) &&
    weights.sharedDown.shape == Shape.of(hidden, intermediate)
  }

  /** The routing, then `moe_rocmfp4`: the experts' outputs, and with a residual
    * and a norm their sum and its norm.
    */
  private def mixture(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      sum: Option[(Tensor, RowNorm)]
  ): Unit = {
    val ExpertProjections(
      router,
      gate,
      up,
      down,
      sharedGate,
      sharedUp,
      sharedDown,
      sharedRouter
    ) =
      weights
    val hidden = router.shape.last
    val intermediate = gate.shape.dimensions(1)
    val tokens = x.shape.dimensions.head
    val k = into.routeWeights.shape.last
    val slots = tokens * k
    val dtype = gate.dtype
    // rows of the down projections walked at once, as the separate launches do
    val group = directRows(intermediate.toInt)
    val ids = into.ids.view(slots)
    Ops.checkExperts(x, gate, ids, into.gated)
    Ops.checkExperts(into.gated, down, ids, into.expertOut)
    Ops.checkExperts(x, sharedGate, into.sharedIds, into.sharedGated)
    Ops.checkLinear(into.sharedGated, sharedDown, into.sharedOut)
    routeLinear(x, router, into.logits, into.ids, into.routeWeights)
    val pairs = intermediate / 2
    def workgroups(waves: Long) = (waves + 7) / 8
    // with a sum: a workgroup per token for its shared expert's gate, and
    // four (SUM_WAVES) for the sum itself
    val grid = sum.fold(0L)(_ => 5 * tokens) + workgroups(tokens * pairs) +
      workgroups(hidden / (Int8Rows * group)) + workgroups(slots * pairs) +
      workgroups(slots * hidden / (2 * group))
    val (target, quantized) = sum.fold(
      (Option.empty[QuantizedValues], Seq.fill(3)(Pointer(MemorySegment.NULL)))
    )((_, norm) => quantizedTarget(norm.out, hidden))
    def summing(part: ((Tensor, RowNorm)) => Tensor) =
      Pointer(sum.fold(MemorySegment.NULL)(s => pointer(part(s))))
    launch(
      kernel(matvecKernels, s"moe_group${group}_${dtype.name.toLowerCase}"),
      grid,
      256,
      (Seq(
        Pointer(pointer(x)),
        Pointer(pointer(into.ids)),
        I32(hidden.toInt),
        I32(tokens.toInt),
        I32(k.toInt),
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(dtype.byteSize(intermediate * hidden)),
        Pointer(pointer(into.gated)),
        I32(intermediate.toInt),
        Pointer(pointer(sharedGate)),
        Pointer(pointer(sharedUp)),
        Pointer(pointer(into.sharedGated)),
        Pointer(pointer(down)),
        I64(dtype.byteSize(hidden * intermediate)),
        Pointer(pointer(into.expertOut)),
        Pointer(pointer(sharedDown)),
        Pointer(pointer(into.sharedOut)),
        Pointer(pointer(into.routeWeights)),
        Pointer(pointer(sharedRouter)),
        Pointer(offset(pointer(moeCounters), 4L * MoeCounters)),
        summing(_._1),
        summing(_._2.weight),
        summing(_._2.out),
        F32(sum.fold(0f)(_._2.epsilon)),
        F32(sum.fold(0f)(_._2.weightOffset))
      ) ++ quantized :+ Pointer(pointer(moeCounters)))*
    )
    wroteQuantized(target)
  }

  /** `moe_rocmfp4`'s counters, then its tokens' shared expert gates and sums'
    * totals.
    */
  private val MoeCounters = 96

  /** `moe_rocmfp4`'s LDS for the shared expert's quantized input. */
  private val MoeSharedValues = 8192

  override def moeCombineNorm(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    Ops.requireF32(
      "moeCombineNorm",
      (Seq(expertOut, weights, residual, norm.out) ++ shared.toSeq.flatMap(e =>
        Seq(e.output, e.input, e.router)
      ))*
    )
    val rows =
      Ops.checkRowWise("moeCombineNorm", residual, norm.out, norm.weight)
    val hidden = residual.shape.last
    val k = weights.shape.elementCount / rows
    if (hidden < 1024)
      super.moeCombineNorm(expertOut, weights, shared, residual, norm)
    else {
      def optional(tensor: SharedExpert => Tensor) = Pointer(
        shared.fold(MemorySegment.NULL)(e => pointer(tensor(e)))
      )
      val (target, quantized) = quantizedTarget(norm.out, hidden)
      launch(
        kernel(normKernels, "moe_combine_norm"),
        rows,
        1024,
        (Seq(
          Pointer(pointer(expertOut)),
          Pointer(pointer(weights)),
          optional(_.output),
          optional(_.input),
          optional(_.router),
          Pointer(pointer(residual)),
          I32(k.toInt),
          I32(hidden.toInt),
          Pointer(pointer(norm.weight)),
          Pointer(pointer(norm.out)),
          F32(norm.epsilon),
          F32(norm.weightOffset)
        ) ++ quantized)*
      )
      wroteQuantized(target)
    }
  }

  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("expertsGatedLinear", gate, up)
    require(
      gate.dtype == up.dtype,
      s"expertsGatedLinear: gate ${gate.dtype}, up ${up.dtype}"
    )
    val (slots, n, k, divisor) = Ops.checkExperts(x, gate, ids, out)
    require(
      integerX.contains(gate.dtype) && gate.dtype.byteSize(k.toLong) % 4 == 0,
      s"no expert kernel for ${gate.dtype} rows of $k"
    )
    val directly = direct(gate.dtype, k) && n % 2 == 0
    if (slots >= GroupedExpertsMinimumSlots && matrixExperts(gate.dtype, n, k))
      expertsMatrix(x, gate, Some(up), ids, out, slots, n, k, divisor)
    else if (slots >= GroupedExpertsMinimumSlots) {
      val (order, batches, maxBatches) =
        groupSlots(ids, slots, gate.shape.dimensions.head.toInt, 8)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_gated_grouped_${gate.dtype.name.toLowerCase}"
        ),
        (maxBatches.toLong * n + 7) / 8,
        256,
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(gate.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(order),
        Pointer(batches),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(maxBatches)
      )
    } else if (directly)
      gatedDirect(x, gate, up, ids, out, slots, n, k, divisor, None)
    else
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_gated_${gate.dtype.name.toLowerCase}"
        ),
        (slots.toLong * n + 7) / 8,
        256,
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(gate.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(slots)
      )
  }

  /** matvec_experts_gated_direct: the routed slots, then the shared expert's
    * gate, up and output (`[1, N, K]`, `[tokens, N]`) when given.
    */
  private def gatedDirect(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int,
      shared: Option[(Tensor, Tensor, Tensor)]
  ): Unit = {
    val sharedSlots = shared.fold(0L)(_._3.shape.dimensions.head)
    def sharedPointer(part: ((Tensor, Tensor, Tensor)) => Tensor) =
      Pointer(shared.fold(MemorySegment.NULL)(s => pointer(part(s))))
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts_gated_direct_${gate.dtype.name.toLowerCase}"
      ),
      ((slots + sharedSlots) * n / 2 + 7) / 8,
      256,
      Pointer(pointer(gate)),
      Pointer(pointer(up)),
      I64(gate.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(slots),
      sharedPointer(_._1),
      sharedPointer(_._2),
      sharedPointer(_._3),
      I32(sharedSlots.toInt)
    )
  }

  override def expertsGatedLinearWithShared(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      sharedGate: Tensor,
      sharedUp: Tensor,
      sharedIds: Tensor,
      sharedOut: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, gate, ids, out)
    val (sharedSlots, sharedN, _, sharedDivisor) =
      Ops.checkExperts(x, sharedGate, sharedIds, sharedOut)
    val fused = slots < GroupedExpertsMinimumSlots && direct(gate.dtype, k) &&
      n % 2 == 0 && gate.shape == up.shape && gate.dtype == up.dtype &&
      sharedGate.dtype == gate.dtype && sharedUp.dtype == gate.dtype &&
      sharedGate.shape == sharedUp.shape && sharedN == n && sharedDivisor == 1
    if (!fused)
      super.expertsGatedLinearWithShared(
        x,
        gate,
        up,
        ids,
        out,
        sharedGate,
        sharedUp,
        sharedIds,
        sharedOut
      )
    else
      gatedDirect(
        x,
        gate,
        up,
        ids,
        out,
        slots,
        n,
        k,
        divisor,
        Some((sharedGate, sharedUp, sharedOut))
      )
  }

  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("gatedRmsNorm", x, out, weight)
    Ops.requireSameShape("gatedRmsNorm", x, gate)
    require(
      Set(Activation.Silu, Activation.Sigmoid).contains(gateActivation),
      s"gatedRmsNorm gates by SiLU or sigmoid, not $gateActivation"
    )
    // the heads' rows side by side are a product's input rows
    val (target, quantized) = quantizedTarget(out, out.shape.last)
    rowWise(
      "gated_rms_norm_f32",
      rows,
      (Seq(
        Pointer(pointer(x)),
        Pointer(pointer(weight)),
        Pointer(pointer(gate)),
        Pointer(pointer(out)),
        I32(x.shape.last.toInt),
        F32(epsilon),
        I32(if (gateActivation == Activation.Sigmoid) 1 else 0)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }

  // ---- residual streams and the n-gram embedding ------------------------------------

  private val streamsKernels = new KernelModule(hip, "streams")

  def streamsMix(
      gates: Tensor,
      normed: Tensor,
      streams: Int,
      out: Tensor
  ): Unit = {
    val (tokens, found, width) =
      Ops.checkStreams("streamsMix", Seq(gates, normed), out)
    require(found == streams, s"streamsMix: $found streams, expected $streams")
    launch(
      kernel(streamsKernels, "streams_mix"),
      (tokens.toLong * width + 255) / 256,
      256,
      Pointer(pointer(gates)),
      Pointer(pointer(normed)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(streams),
      I32(width)
    )
  }

  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit = {
    val (tokens, streams, width) = Ops.checkStreams("streamsCombine", Seq(x), y)
    Ops.requireF32("streamsCombine", logits)
    require(
      logits.shape == Shape.of(tokens, streams),
      s"streamsCombine: logits ${logits.shape}"
    )
    launch(
      kernel(streamsKernels, "streams_combine"),
      (tokens.toLong * streams * width + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(logits)),
      I32(tokens),
      I32(streams),
      I32(width)
    )
  }

  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit = {
    val (tokens, streams, width) =
      Ops.checkStreams("pleGate", Seq(key, query, out), value)
    launch(
      kernel(streamsKernels, "ple_gate"),
      tokens.toLong * streams,
      256,
      Pointer(pointer(key)),
      Pointer(pointer(query)),
      Pointer(pointer(value)),
      Pointer(pointer(out)),
      I32(streams),
      I32(width)
    )
  }

  /** Each hash's multipliers, sizes and offsets on the device, as 64-bit words
    * (two ints each), uploaded once.
    */
  private val hashTables = mutable.Map.empty[NgramHash, Seq[Tensor]]

  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit = {
    val count = Ops.checkNgramRows(tokens, state, hash, rows)
    history.foreach(Ops.checkHistory(_, count, state))
    val Seq(multipliers, sizes, offsets) = hashTables.getOrElseUpdate(
      hash,
      Seq(hash.multipliers, hash.sizes, hash.offsets).map(longs =>
        fromInts(
          Shape.of(2L * longs.size),
          longs.flatMap(l => Seq(l.toInt, (l >>> 32).toInt)).toArray
        )
      )
    )
    launch(
      kernel(streamsKernels, "ngram_rows"),
      (count.toLong * hash.heads + 255) / 256,
      256,
      Pointer(pointer(tokens)),
      Pointer(pointer(state)),
      Pointer(pointer(multipliers)),
      Pointer(pointer(sizes)),
      Pointer(pointer(offsets)),
      Pointer(pointer(rows)),
      I32(count),
      I32(hash.ngramSize),
      I32(hash.headsPerNgram),
      I32(hash.eos),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
    launch(
      kernel(streamsKernels, "ngram_state"),
      1,
      32,
      Pointer(pointer(tokens)),
      Pointer(pointer(state)),
      I32(count),
      I32(hash.kept)
    )
  }

  def allocateCache(
      pages: Int,
      pageSize: Int,
      kvHeads: Int,
      headDimension: Int
  ): KvCache =
    KvCache(
      allocate(DType.F16, Shape.of(pages, pageSize, kvHeads, headDimension)),
      allocate(DType.F16, Shape.of(pages, kvHeads, headDimension, pageSize)),
      pageSize
    )

  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit = {
    val tokens = Ops.checkCacheWrite(k, v, cache, pageTable, start)
    val count = k.shape.elementCount
    launch(
      kernel(attentionKernels, "cache_write"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(k)),
      Pointer(pointer(v)),
      Pointer(pointer(cache.keys)),
      Pointer(pointer(cache.values)),
      Pointer(pointer(pageTable)),
      I32(tokens),
      I32(cache.kvHeads),
      I32(cache.headDimension),
      I32(cache.pageSize),
      I32(start)
    )
  }

  override def attentionInputs(
      queryAndGate: Tensor,
      keys: Tensor,
      values: Tensor,
      norms: QueryKeyNorms,
      positions: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int,
      queries: Tensor,
      gate: Tensor,
      rotatedKeys: Tensor,
      rotatedQueries: Tensor
  ): Unit = {
    val Seq(t, heads, d) = queries.shape.dimensions
    val kvHeads = keys.shape.dimensions(1)
    if (d > AttentionInputDimension)
      super.attentionInputs(
        queryAndGate,
        keys,
        values,
        norms,
        positions,
        cache,
        pageTable,
        start,
        queries,
        gate,
        rotatedKeys,
        rotatedQueries
      )
    else {
      Ops.checkSplitHalves(queryAndGate, queries, gate)
      Ops.checkRope(queries, positions, norms.rope, rotatedQueries)
      Ops.checkRope(keys, positions, norms.rope, rotatedKeys)
      Ops.checkCacheWrite(rotatedKeys, values, cache, pageTable, start)
      val rope = norms.rope
      val Seq(s0, s1, s2, s3) = ropeSections(rope)
      launch(
        kernel(attentionKernels, "attention_inputs"),
        t * (heads + kvHeads),
        256,
        Pointer(pointer(queryAndGate)),
        Pointer(pointer(keys)),
        Pointer(pointer(values)),
        Pointer(pointer(norms.queryNorm)),
        Pointer(pointer(norms.keyNorm)),
        F32(norms.epsilon),
        F32(norms.weightOffset),
        Pointer(pointer(positions)),
        I32(t.toInt),
        I32(heads.toInt),
        I32(kvHeads.toInt),
        I32(d.toInt),
        I32(rope.rotaryDimensions),
        F32(rope.theta),
        I32(rope.layout.code),
        I32(rope.sections.code),
        I32(s0),
        I32(s1),
        I32(s2),
        I32(s3),
        Pointer(pointer(rotatedQueries)),
        Pointer(pointer(gate)),
        Pointer(pointer(cache.keys)),
        Pointer(pointer(cache.values)),
        Pointer(pointer(pageTable)),
        I32(cache.pageSize),
        I32(start)
      )
    }
  }

  /** `attention.hip`'s ATTENTION_INPUT_DIMENSION. */
  private val AttentionInputDimension = 512

  /** The kernel's `AttentionArguments`, field for field. */
  private val attentionLayout = MemoryLayout.structLayout(
    (Seq(
      "q",
      "keys",
      "values",
      "page_table",
      "sinks",
      "out",
      "partial_o",
      "partial_ml"
    )
      .map(ADDRESS.withName(_)) ++
      Seq(
        "tokens",
        "q_heads",
        "kv_heads",
        "page_size",
        "query_start",
        "key_count"
      )
        .map(JAVA_INT.withName(_)) ++
      Seq(
        JAVA_FLOAT.withName("scale"),
        JAVA_INT.withName("causal"),
        JAVA_INT.withName("window"),
        JAVA_FLOAT.withName("softcap"),
        JAVA_INT.withName("splits"),
        JAVA_INT.withName("token_tiles"),
        JAVA_INT.withName("first_group")
      ))*
  )

  /** Each field's byte offset, resolved once: resolving a layout path per call
    * costs more than the launch.
    */
  private val attentionOffsets: Map[String, Long] =
    attentionLayout
      .memberLayouts()
      .asScala
      .flatMap(member =>
        member
          .name()
          .map[(String, Long)](name =>
            name -> attentionLayout
              .byteOffset(MemoryLayout.PathElement.groupElement(name))
          )
          .toScala
      )
      .toMap

  /** With `token_tiles`, how many splits a token's keys go into at most, and
    * the fewest blocks of 16 keys a split takes (`TOKEN_SPLITS`,
    * `MIN_SPLIT_BLOCKS`).
    */
  private val TokenSplits = 32
  private val MinimumSplitBlocks = 2

  /** Enough workgroups to fill the GPU: a decode step has few query rows, so
    * its keys are split (flash decoding), each split at least 16 blocks of 16
    * (fewer made the per-workgroup setup dominate at 4096 keys).
    */
  private def splitsFor(workgroups: Long, keyCount: Int): Int = {
    val target = 160L
    if (workgroups >= target) 1
    else
      math.max(
        1,
        math
          .min((target + workgroups - 1) / workgroups, (keyCount / 256).toLong)
          .toInt
      )
  }

  /** Query rows (tokens × the group's heads) from which the tiled kernel runs,
    * and its waves per workgroup (`attention.hip`'s `TILED_WAVES`).
    */
  private val TiledMinimumRows = 256
  private val TiledWaves = 8

  /** The most one launch of the tiled kernel is given, in query rows × keys ×
    * key heads: about 50 ms on the Radeon 8060S. PiD's attention over a 4096²
    * tile was one launch of 2.1 s, 56 times a tile, and the desktop's frames
    * waited behind each (bug 38); cut in launches this size the GPU is free for
    * them in between, and the tile takes the same time.
    */
  private val TiledLaunchWork = 1.5e9

  def attention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): Unit = {
    val (tokens, qHeads) = Ops.checkAttention(
      q,
      cache,
      pageTable,
      queryStart,
      keyCount,
      attention,
      out
    )
    val dimension = cache.headDimension
    require(
      Attention.KernelHeadWidths.contains(dimension),
      s"no attention kernel for heads of $dimension yet"
    )
    val group = qHeads / cache.kvHeads
    val rows = tokens.toLong * group
    // a decode step or a verification: a tile per token and splits of fixed
    // runs of keys (`token_tiles`), so each token attends as it would alone
    val tokenTiles = tokens <= MaxMatVecRows
    val tiles =
      if (tokenTiles) tokens.toLong * ((group + 15) / 16) else (rows + 15) / 16
    require(
      tiles <= Int.MaxValue && rows <= Int.MaxValue,
      s"attention over $rows rows"
    )
    // many rows: the tiled kernel, its waves sharing the keys and values
    val tiled = Option.when(
      !tokenTiles && rows >= TiledMinimumRows && dimension <= 128
    )(
      (s"attention_tiled_d$dimension", TiledWaves)
    )
    val splits =
      if (tiled.isDefined) 1
      else if (tokenTiles) {
        // as many as the tile with the most can use: each cuts its blocks
        // into up to TokenSplits runs of at least MinimumSplitBlocks
        val keys =
          attention.window.fold(keyCount)(window =>
            math.min(keyCount, window + 16)
          )
        val blocks = (keys + 15) / 16
        math.min(
          TokenSplits,
          (blocks + MinimumSplitBlocks - 1) / MinimumSplitBlocks
        )
      } else splitsFor(tiles * cache.kvHeads, keyCount)
    val partials = 4L * cache.kvHeads * rows * splits
    val partialO =
      if (splits > 1) scratch(0, partials * (dimension + 2))
      else MemorySegment.NULL
    val partialMl =
      if (splits > 1) offset(partialO, partials * dimension)
      else MemorySegment.NULL
    val arena = Arena.ofConfined()
    try {
      val arguments = arena.allocate(attentionLayout)
      def at(name: String) = attentionOffsets(name)
      def address(name: String, value: MemorySegment) =
        arguments.set(ADDRESS, at(name), value)
      def int(name: String, value: Int) =
        arguments.set(JAVA_INT, at(name), value)
      def float(name: String, value: Float) =
        arguments.set(JAVA_FLOAT, at(name), value)
      address("q", pointer(q))
      address("keys", pointer(cache.keys))
      address("values", pointer(cache.values))
      address("page_table", pointer(pageTable))
      address("sinks", attention.sinks.fold(MemorySegment.NULL)(pointer))
      address("out", pointer(out))
      address("partial_o", partialO)
      address("partial_ml", partialMl)
      int("tokens", tokens)
      int("q_heads", qHeads)
      int("kv_heads", cache.kvHeads)
      int("page_size", cache.pageSize)
      int("query_start", queryStart)
      int("key_count", keyCount)
      float("scale", attention.scale)
      int("causal", if (attention.causal) 1 else 0)
      int("window", attention.window.getOrElse(0))
      float("softcap", attention.softcap.getOrElse(0f))
      int("splits", splits)
      int("token_tiles", if (tokenTiles) 1 else 0)
      val (name, waves) = tiled.getOrElse((s"attention_d$dimension", 1))
      val groups = ((tiles + waves - 1) / waves).toInt
      // the tiled kernel's rows in launches short enough for the desktop
      val launches =
        if (tiled.isEmpty) 1
        else
          math
            .ceil(rows.toDouble * keyCount * cache.kvHeads / TiledLaunchWork)
            .toInt
            .max(1)
            .min(groups)
      val each = (groups + launches - 1) / launches
      (0 until groups by each).foreach { first =>
        int("first_group", first)
        hip.launch(
          kernel(attentionKernels, name),
          Dim3(math.min(each, groups - first), cache.kvHeads, splits),
          Dim3(32 * waves),
          0,
          MemorySegment.NULL,
          KernelArgument.Struct(arguments)
        )
      }
      if (splits > 1)
        hip.launch(
          kernel(attentionKernels, "attention_combine"),
          Dim3(rows.toInt, cache.kvHeads, 1),
          Dim3(dimension),
          0,
          MemorySegment.NULL,
          KernelArgument.Struct(arguments),
          I32(dimension)
        )
    } finally arena.close()
  }

  // ---- memory ------------------------------------------------------------------------

  /** Device memory for intermediate results, grown on demand and kept; one
    * region, so a caller takes all it needs at once.
    */
  private var scratchPointer = MemorySegment.NULL
  private var scratchBytes = 0L

  private def scratch(at: Long, bytes: Long): MemorySegment = {
    if (at + bytes > scratchBytes) {
      if (scratchBytes > 0) hip.free(scratchPointer)
      scratchPointer = hip.allocate(at + bytes)
      scratchBytes = at + bytes
    }
    offset(scratchPointer, at)
  }

  private def offset(address: MemorySegment, bytes: Long): MemorySegment =
    MemorySegment.ofAddress(address.address() + bytes)

  /** The device address of a tensor's first byte. */
  private def pointer(tensor: Tensor): MemorySegment = {
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
    allocations.foreach(hip.free)
    allocations.clear()
    if (scratchBytes > 0) hip.free(scratchPointer)
    scratchBytes = 0
    blasHandle.foreach(_.close())
    blasHandle = None
    Seq(
      elementwiseKernels,
      normKernels,
      ropeKernels,
      matvecKernels,
      attentionKernels,
      linearAttentionKernels
    )
      .foreach(_.close())
  }
}
