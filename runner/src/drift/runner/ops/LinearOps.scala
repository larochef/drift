package drift.runner.ops

import drift.runner.tensor.*

import java.util.concurrent.ConcurrentSkipListMap

/** Linear algebra over stored weights: linear layers, embedding tables, and the
  * rotated linears read where their file holds them.
  */
trait LinearOps {

  /** `out[t] = table[ids[t]]`: rows of an embedding table in any storage type,
    * as F32 (a rotated table's, `rotated`, as the BF16 it holds). `ids` is I32
    * `[tokens]`, `out` `[tokens, D]`.
    */
  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit

  // the rotated weights by where their codes start
  private val rotatedWeights =
    new ConcurrentSkipListMap[Long, (Tensor, ConvRotParts)]

  /** ComfyUI's rotated linear `[N, K]` left where the file holds it: `codes`,
    * I8 `[N, K]`, or with `levels` the 4-bit codes `[N, K / 2]`, seen as the
    * weight they make with `parts` (`ConvRot.Codes`, `ConvRot.Nibbles`).
    * `linear`, `embedding` and `convert` decode it, and rows of it
    * (`Tensor.rows`) as they do the whole; `parts` are the caller's to release,
    * after `forgetRotated`.
    */
  def rotated(codes: Tensor, parts: ConvRotParts): Tensor = {
    val weight = Ops.checkRotated(codes, parts)
    rotatedWeights.put(Ops.address(weight), (weight, parts))
    weight
  }

  def forgetRotated(weight: Tensor): Unit = {
    rotatedWeights.remove(Ops.address(weight))
    ()
  }

  /** What decodes a rotated linear, or rows of one: those rows' scales. */
  protected def partsOf(weight: Tensor): ConvRotParts = {
    val address = Ops.address(weight)
    Option(rotatedWeights.floorEntry(address))
      .map(_.getValue)
      .flatMap { (whole, parts) =>
        val rowBytes = whole.dtype.byteSize(whole.shape.last)
        val offset = address - Ops.address(whole)
        val (from, count) = (offset / rowBytes, weight.shape.dimensions.head)
        Option.when(
          weight.dtype == whole.dtype && weight.shape.rank == 2 &&
            weight.shape.last == whole.shape.last && offset % rowBytes == 0 &&
            from + count <= whole.shape.dimensions.head
        )(
          ConvRotParts(
            if (parts.scales.shape.elementCount == 1) parts.scales
            else parts.scales.rows(from, count),
            parts.levels
              .map(l => l.copy(relative = l.relative.rows(from, count)))
          )
        )
      }
      .getOrElse(
        throw new IllegalArgumentException(
          s"${weight.dtype} ${weight.shape}: codes this backend holds no scales for"
        )
      )
  }

  /** Whether the products of quantized weights over many rows run in BF16
    * rather than F16 (a backend that converts them): F32's range, for
    * activations past F16's (MiniMax H3's MLP). One setting for the process,
    * set by the pipeline that needs it.
    */
  @volatile var wideProducts: Boolean = false

  /** `out = x · weightᵀ`, a linear layer without bias: `x` is `[M, K]` F32,
    * `weight` `[N, K]` in any storage type, `out` `[M, N]` F32.
    */
  def linear(x: Tensor, weight: Tensor, out: Tensor): Unit

  /** Several linear layers of the same `x` (Q, K and V; gate and up): a backend
    * may prepare `x` once for all of them.
    */
  def linears(x: Tensor, weights: Seq[Tensor], outs: Seq[Tensor]): Unit =
    weights.zip(outs).foreach((weight, out) => linear(x, weight, out))
}

/** The rules of `LinearOps`.
  */
private[ops] trait LinearChecks extends CommonChecks {

  def checkEmbedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    require(
      table.shape.rank == 2,
      s"an embedding table is [vocabulary, D], not ${table.shape}"
    )
    require(
      ids.dtype == DType.I32 && ids.shape.rank == 1,
      s"embedding ids are I32 [tokens], not ${ids.dtype} ${ids.shape}"
    )
    requireF32("embedding", out)
    require(
      out.shape == Shape.of(ids.shape.dimensions.head, table.shape.last),
      s"embedding: out ${out.shape} for ${ids.shape.dimensions.head} rows of ${table.shape.last}"
    )
  }

  /** Where a tensor's bytes start on the host: what tells two tensors over
    * mapped files apart.
    */
  def address(tensor: Tensor): Long =
    tensor.byteOffset + (tensor.storage match {
      case Storage.Host(segment)       => segment.address()
      case Storage.Registered(host, _) => host.address()
      case Storage.Device(pointer, _)  => pointer.address()
    })

  /** Returns the weight `codes` are with `parts`. */
  def checkRotated(codes: Tensor, parts: ConvRotParts): Tensor = {
    require(
      codes.dtype == DType.I8 && codes.shape.rank == 2,
      s"rotated: ${codes.dtype} codes ${codes.shape}"
    )
    val n = codes.shape.dimensions(0)
    val k = codes.shape.last * parts.levels.fold(1)(_ => 2)
    require(k % ConvRot.Group == 0, s"rotated: rows of $k columns")
    val scaleCount = parts.scales.shape.elementCount
    require(
      parts.scales.dtype == DType.F32 &&
        (scaleCount == n || scaleCount == 1 && parts.levels.isEmpty),
      s"rotated: $scaleCount ${parts.scales.dtype} scales for $n rows"
    )
    parts.levels.foreach { l =>
      require(
        l.groupSize > 0 && l.groupSize % 16 == 0 && k % l.groupSize == 0,
        s"rotated: $k columns in groups of ${l.groupSize}"
      )
      require(
        l.codebook.dtype == DType.F32 && l.codebook.shape.elementCount == 16,
        s"rotated: a codebook ${l.codebook.dtype} ${l.codebook.shape}"
      )
      require(
        l.relative.dtype == DType.F8E4M3 &&
          l.relative.shape == Shape.of(n, k / l.groupSize),
        s"rotated: relative scales ${l.relative.dtype} ${l.relative.shape}"
      )
    }
    Tensor(
      if (parts.levels.isEmpty) ConvRot.Codes else ConvRot.Nibbles,
      Shape.of(n, k),
      codes.storage,
      codes.byteOffset
    )
  }

  /** Returns `(M, N, K)`. */
  def checkLinear(
      x: Tensor,
      weight: Tensor,
      out: Tensor
  ): (Long, Long, Long) = {
    requireF32("linear", x, out)
    require(
      x.shape.rank == 2 && weight.shape.rank == 2 && out.shape.rank == 2,
      s"linear takes [M, K] · [N, K]ᵀ → [M, N], got ${x.shape}, ${weight.shape}, ${out.shape}"
    )
    val Seq(m, k) = x.shape.dimensions
    val Seq(n, weightK) = weight.shape.dimensions
    require(
      weightK == k,
      s"linear: x rows of $k against weight rows of $weightK"
    )
    require(
      out.shape == Shape.of(m, n),
      s"linear: out ${out.shape} for [$m, $n]"
    )
    (m, n, k)
  }
}
