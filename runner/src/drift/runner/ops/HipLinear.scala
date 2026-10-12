package drift.runner.ops

import drift.runner.native.*
import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.*

/** The GPU backend's linear algebra: matrix-vector passes over quantized
  * weights for a few rows, hipBLAS GEMMs for many, and the gathers of embedding
  * tables.
  */
private[ops] trait HipLinear extends HipBase {

  /** From this many rows of x, `linear` is a GEMM: the weight dequantized to
    * F16 (a fixed cost, about 40 ms over a 4B model's weights), then hipBLAS.
    * Below, passes of `MaxMatVecRows` cost less.
    */
  val GemmMinimumRows = 32

  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    Ops.checkEmbedding(table, ids, out)
    require(
      integerX.contains(table.dtype),
      s"no kernels for ${table.dtype} tables yet"
    )
    val (count, columns) = (ids.shape.elementCount, table.shape.last)
    if (ConvRot.stored(table.dtype)) gatherRotated(table, ids, out)
    else if (table.dtype == DType.F32)
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

  /** Rows `ids` of a rotated table (`rotated`), read where the file holds it,
    * as the BF16 it holds.
    */
  private def gatherRotated(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    val parts = partsOf(table)
    val blocks = ids.shape.elementCount * (table.shape.last / ConvRot.Group)
    val grid = math.min((blocks + 15) / 16, 1L << 16)
    val shape = Seq(
      Pointer(pointer(ids)),
      Pointer(pointer(out)),
      I64(blocks),
      I32(table.shape.last.toInt)
    )
    parts.levels match {
      case None =>
        launch(
          kernel(convRotKernels, "convrot_gather_f32_i8"),
          grid,
          ConvRot.Group,
          (Seq(
            Pointer(pointer(table)),
            Pointer(pointer(parts.scales)),
            I32(parts.scales.shape.elementCount.toInt)
          ) ++ shape)*
        )
      case Some(levels) =>
        launch(
          kernel(convRotKernels, "convrot_gather_f32_w4a8"),
          grid,
          ConvRot.Group,
          (Seq(
            Pointer(pointer(table)),
            Pointer(pointer(levels.codebook)),
            Pointer(pointer(levels.relative)),
            Pointer(pointer(parts.scales))
          ) ++ shape :+ I32(levels.groupSize))*
        )
    }
  }

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
    // go to hipBLAS at any row count, and so do rotated ones, which have no
    // matrix-vector kernels
    val gemmOnly = m >= GemmMinimumRows || weights.forall(
      _.dtype == DType.BF16
    ) && k % 32 != 0 || weights.exists(w => ConvRot.stored(w.dtype))
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
  protected val Int8Rows = 2

  /** Single-vector products of at most this many rows split each row over a
    * workgroup's eight waves: one wave per row would leave most of the GPU idle
    * (Qwen 3.6's router, 256 rows, ran at 78 GB/s).
    */
  val SplitMaximumRows = 1024

  /** "_narrow" for rows short enough that the 1 KB-trip kernels serve them
    * better: the 4 KB ones reserve LDS that caps how many workgroups a compute
    * unit holds, and short rows are latency-bound.
    */
  protected def narrow(dtype: DType, columns: Long): String =
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
    // rotated weights were BF16 before they were quantized
    val wide =
      (wideProducts || ConvRot.stored(
        weight.dtype
      )) && weight.dtype != DType.F16
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
      } else if (ConvRot.stored(weight.dtype)) {
        val target = offset(xHalf, 2L * m * k)
        dequantizeRotated(weight, target)
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

  /** Types whose rows of `k` the `direct` kernels read straight into registers,
    * no LDS staging: blocks of 256 that keep lanes' bytes aligned (Q4_K and
    * Q5_K to 16 bytes, IQ4_XS to 8), and ROCmFP4's blocks of 32 eight at a time
    * (to 2 bytes; a lane loads from the word its bytes start in), in rows of
    * whole such eights.
    */
  protected def direct(dtype: DType, k: Long): Boolean =
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
  protected def rocmFp4Int8(dtype: DType, k: Long): Boolean =
    (dtype == RocmFp4.Dual || dtype == RocmFp4.Fast) &&
      k % DirectBlockElements == 0

  /** What a `direct` kernel takes as one block. */
  protected val DirectBlockElements = 256

  /** How many consecutive rows of `k` columns a `direct` kernel walks as one: a
    * pass takes eight blocks, so rows of fewer that divide eight go several at
    * once.
    */
  protected def directRows(k: Int): Int = {
    val blocks = k / DirectBlockElements
    if (blocks < 8 && 8 % blocks == 0) 8 / blocks else 1
  }

  /** How many consecutive rows of `n` share a wave in `matvec_experts`: a wave
    * decodes a chunk of 32 lanes' parts at once (32 values each), so rows of
    * fewer blocks than a chunk, which divide it, are walked several at a time.
    */
  protected def rowsPerWave(dtype: DType, n: Int, k: Int): Int = {
    val chunkBlocks = 1024 / dtype.blockElements
    val rowBlocks = k / dtype.blockElements
    val group = chunkBlocks / rowBlocks
    if (
      rowBlocks < chunkBlocks && chunkBlocks % rowBlocks == 0 && n % group == 0
    )
      group
    else 1
  }
}
