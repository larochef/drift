package drift.runner.models

import drift.runner.ops.*
import drift.runner.tensor.*

import scala.collection.mutable

/** What LLaDA-Image's small models share: their weights under a prefix of a
  * file (a connectors file names them `queryformer.`, `text_projection.` and
  * `sigvq.`; the released folders hold each alone), linear layers with a bias,
  * and RMS norms without weights.
  */
abstract private[models] class LladaPart(
    protected val ops: Ops,
    source: WeightSource,
    prefix: String
) extends AutoCloseable {
  protected val weights = new HybridWeights(ops, source, gguf = false)
  protected val Epsilon = 1e-6f

  protected def has(name: String): Boolean = source.has(prefix + name)
  protected def matrix(name: String): Tensor = source(prefix + name)
  protected def dimensions(name: String): Seq[Int] =
    source(prefix + name).shape.dimensions.map(_.toInt)
  protected def floats(name: String): Tensor = {
    val full = prefix + name
    weights.floats(full, full, Shape.of(source(full).shape.elementCount))
  }
  protected def hostFloats(name: String): Array[Float] =
    weights.hostFloats(prefix + name)

  final protected case class Affine(weight: Tensor, bias: Tensor)
  protected def affineLayer(name: String): Affine =
    Affine(matrix(s"$name.weight"), floats(s"$name.bias"))
  protected def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  /** Floats as BF16 (round to nearest even), uploaded. */
  protected def bf16(shape: Shape, values: Array[Float]): Tensor = {
    val bytes = new Array[Byte](2 * values.length)
    values.indices.foreach { i =>
      val bits = java.lang.Float.floatToRawIntBits(values(i))
      val rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16
      bytes(2 * i) = rounded.toByte
      bytes(2 * i + 1) = (rounded >>> 8).toByte
    }
    weights.upload(DType.BF16, shape, bytes)
  }

  protected def constant(count: Long, value: Float): Tensor =
    weights.keep(
      ops.fromFloats(Shape.of(count), Array.fill(count.toInt)(value))
    )

  /** Tensors of one call, released together. */
  final protected class Scratch {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    def rows(count: Long, columns: Long): Tensor = {
      val tensor = ops.allocate(DType.F32, Shape.of(count, columns))
      held += tensor
      tensor
    }
    def keep(tensor: Tensor): Tensor = {
      held += tensor
      tensor
    }
    def release(): Unit = held.foreach(ops.release)
  }

  def close(): Unit = weights.release()
}

/** LLaDA-Image's QueryFormer (`LLaDAImageQueryFormerModel`): learned queries
  * (256) that read the prompt's token embeddings by cross attention (heads of
  * 128, one fused projection with a bias), then an MLP (GELU, tanh); each
  * residual starts from the layer-normed queries.
  */
final class LladaQueryFormer(ops: Ops, source: WeightSource, prefix: String)
    extends LladaPart(ops, source, prefix) {

  private val Seq(queries, width) = dimensions("meta_queries")
  private val heads = width / 128
  private val meta = weights.keep(
    ops.fromFloats(Shape.of(queries, width), hostFloats("meta_queries"))
  )

  final private class Block(at: String) {
    private val fused = matrix(s"$at.cross_attn.in_proj_weight")
    private val bias = floats(s"$at.cross_attn.in_proj_bias")
    private def part(i: Int) = Affine(
      fused.rows(i.toLong * width, width),
      bias.rows(i.toLong * width, width)
    )
    val (query, key, value) = (part(0), part(1), part(2))
    val output: Affine = affineLayer(s"$at.cross_attn.out_proj")
    val up: Affine = affineLayer(s"$at.mlp.fc1")
    val down: Affine = affineLayer(s"$at.mlp.fc2")
  }
  private val blocks = Iterator
    .from(0)
    .map(i => s"query_blocks.$i")
    .takeWhile(at => has(s"$at.cross_attn.in_proj_weight"))
    .map(new Block(_))
    .toSeq
  private val attention = new PlainAttention(ops, heads, heads, width / heads)

  /** The queries for a prompt's token embeddings (`[tokens, hidden]`):
    * `[queries, hidden]`.
    */
  def apply(embeds: Tensor): Tensor = {
    val tokens = embeds.shape.dimensions.head
    val scratch = new Scratch
    val x = ops.allocate(DType.F32, Shape.of(queries, width))
    try {
      ops.copy(meta, x)
      val (q, attended, projected) = (
        scratch.rows(queries, width),
        scratch.rows(queries, width),
        scratch.rows(queries, width)
      )
      val (read, k, v) = (
        scratch.rows(tokens, width),
        scratch.rows(tokens, width),
        scratch.rows(tokens, width)
      )
      blocks.foreach { block =>
        ops.layerNorm(x, None, None, Epsilon, x)
        ops.layerNorm(embeds, None, None, Epsilon, read)
        affine(x, block.query, q)
        affine(read, block.key, k)
        affine(read, block.value, v)
        attention(q, k, v, attended)
        affine(attended, block.output, projected)
        ops.add(x, projected, x)
        ops.layerNorm(x, None, None, Epsilon, x)
        val inner = scratch.rows(queries, block.up.bias.shape.elementCount)
        affine(x, block.up, inner)
        ops.activation(Activation.GeluTanh, inner, inner)
        affine(inner, block.down, projected)
        ops.add(x, projected, x)
      }
      x
    } catch {
      case error: Throwable =>
        ops.release(x)
        throw error
    } finally scratch.release()
  }

  override def close(): Unit = {
    attention.close()
    super.close()
  }
}

/** LLaDA-Image's text projection (`LLaDAImageTextProjectionModel`): blocks of
  * self attention (heads of 64, RMS norms on q and k, every linear with a bias)
  * and an MLP (GELU, tanh), each after an RMS norm without weights, then a
  * linear to the transformer's caption features.
  */
final class LladaTextProjection(ops: Ops, source: WeightSource, prefix: String)
    extends LladaPart(ops, source, prefix) {

  private val width = dimensions("projector.weight").last
  private val heads = width / 64
  private val head = width / heads

  /** The caption features' width. */
  val outputs: Int = dimensions("projector.weight").head

  final private class Block(at: String) {
    val query: Affine = affineLayer(s"$at.self_attn.q_proj")
    val key: Affine = affineLayer(s"$at.self_attn.k_proj")
    val value: Affine = affineLayer(s"$at.self_attn.v_proj")
    val output: Affine = affineLayer(s"$at.self_attn.out_proj")
    val up: Affine = affineLayer(s"$at.mlp.fc1")
    val down: Affine = affineLayer(s"$at.mlp.fc2")
  }
  private val blocks = Iterator
    .from(0)
    .map(i => s"layers.$i")
    .takeWhile(at => has(s"$at.self_attn.q_proj.weight"))
    .map(new Block(_))
    .toSeq
  private val projector = affineLayer("projector")
  private val ones = constant(width, 1f)
  private val headOnes = constant(head, 1f)
  private val attention = new PlainAttention(ops, heads, heads, head)

  /** The caption features of the text model's hidden states (`[tokens,
    * hidden]`): `[tokens, outputs]`.
    */
  def apply(hidden: Tensor): Tensor = {
    val tokens = hidden.shape.dimensions.head
    val scratch = new Scratch
    try {
      val x = scratch.rows(tokens, width)
      ops.copy(hidden, x)
      val Seq(normed, q, k, v, attended, projected) =
        Seq.fill(6)(scratch.rows(tokens, width))
      blocks.foreach { block =>
        ops.rmsNorm(x, ones, Epsilon, 0f, normed)
        affine(normed, block.query, q)
        affine(normed, block.key, k)
        affine(normed, block.value, v)
        Seq(q, k).foreach(t =>
          ops.rmsNorm(
            t.view(tokens * heads, head),
            headOnes,
            Epsilon,
            0f,
            t.view(tokens * heads, head)
          )
        )
        attention(q, k, v, attended)
        affine(attended, block.output, projected)
        ops.add(x, projected, x)
        ops.rmsNorm(x, ones, Epsilon, 0f, normed)
        val inner = scratch.rows(tokens, block.up.bias.shape.elementCount)
        affine(normed, block.up, inner)
        ops.activation(Activation.GeluTanh, inner, inner)
        affine(inner, block.down, projected)
        ops.add(x, projected, x)
      }
      val out = ops.allocate(DType.F32, Shape.of(tokens, outputs))
      affine(x, projector, out)
      out
    } finally scratch.release()
  }

  override def close(): Unit = {
    attention.close()
    super.close()
  }
}

/** LLaDA-Image's SigVQ encoder (`LLaDAImageSigVQModel`, GLM's vision
  * tokenizer), which reads the source image of an edit: patches of 16 through a
  * linear, a learned position table sampled bilinearly at the patches' centres,
  * pre-norm transformer blocks (heads of 96, GELU), a 1×1 to the codebook's
  * width and the nearest code by cosine (the choice made on the host); then
  * each code's semantic embedding through a small MLP. Heads of 96 are padded
  * to the attention kernels' 128 in the weights themselves: zero rows in the
  * projections of q, k and v, zero columns in the output's.
  */
final class LladaSigVq(ops: Ops, source: WeightSource, prefix: String)
    extends LladaPart(ops, source, prefix) {

  private val HeadWidth = 96
  private val width = dimensions("visual.patch_embed.proj.weight").head

  /** Image pixels per token along each side. */
  val patch: Int = dimensions("visual.patch_embed.proj.weight").last
  private val heads = width / HeadWidth
  private val padded = Attention.paddedHeadWidth(HeadWidth)
  private val wide = heads.toLong * padded

  /** The semantic features' width. */
  val outputs: Int = dimensions("prior_token_embedding.weight").last

  private val patchEmbed = Affine(
    bf16(
      Shape.of(width, 3L * patch * patch),
      hostFloats("visual.patch_embed.proj.weight")
    ),
    floats("visual.patch_embed.proj.bias")
  )
  private val positionTable =
    hostFloats("visual.embeddings.position_embedding.weight")
  private val side =
    math.round(math.sqrt(positionTable.length / width.toDouble)).toInt

  final private class Block(at: String) {
    val norm1: (Tensor, Tensor) =
      (floats(s"$at.norm1.weight"), floats(s"$at.norm1.bias"))
    val norm2: (Tensor, Tensor) =
      (floats(s"$at.norm2.weight"), floats(s"$at.norm2.bias"))
    private val fused = hostFloats(s"$at.attn.qkv.weight")
    private val fusedBias = hostFloats(s"$at.attn.qkv.bias")
    // row `h × padded + j` of a part is its stored row `h × 96 + j`
    private def part(index: Int) = Affine(
      bf16(
        Shape.of(wide, width),
        Array.tabulate(wide.toInt * width) { i =>
          val (row, column) = (i / width, i % width)
          val (h, j) = (row / padded, row % padded)
          if (j >= HeadWidth) 0f
          else fused((index * width + h * HeadWidth + j) * width + column)
        }
      ),
      weights.keep(
        ops.fromFloats(
          Shape.of(wide),
          Array.tabulate(wide.toInt) { row =>
            val (h, j) = (row / padded, row % padded)
            if (j >= HeadWidth) 0f
            else fusedBias(index * width + h * HeadWidth + j)
          }
        )
      )
    )
    val (query, key, value) = (part(0), part(1), part(2))
    val output: Affine = {
      val stored = hostFloats(s"$at.attn.proj.weight")
      Affine(
        bf16(
          Shape.of(width, wide),
          Array.tabulate(width * wide.toInt) { i =>
            val (row, column) = (i / wide.toInt, i % wide.toInt)
            val (h, j) = (column / padded, column % padded)
            if (j >= HeadWidth) 0f
            else stored(row * width + h * HeadWidth + j)
          }
        ),
        floats(s"$at.attn.proj.bias")
      )
    }
    val up: Affine = affineLayer(s"$at.mlp.fc1")
    val down: Affine = affineLayer(s"$at.mlp.fc2")
  }
  private val blocks = Iterator
    .from(0)
    .map(i => s"visual.blocks.$i")
    .takeWhile(at => has(s"$at.attn.qkv.weight"))
    .map(new Block(_))
    .toSeq

  private val codeWidth = dimensions("vqmodel.quant_conv.weight").head
  private val toCode = Affine(
    bf16(Shape.of(codeWidth, width), hostFloats("vqmodel.quant_conv.weight")),
    floats("vqmodel.quant_conv.bias")
  )
  // the codebook, each code at unit length: the nearest by cosine is the
  // largest product
  private val codebook = {
    val stored = hostFloats("vqmodel.quantize.embedding.weight")
    val codes = stored.length / codeWidth
    val unit = new Array[Float](stored.length)
    (0 until codes).foreach { code =>
      val at = code * codeWidth
      var sum = 0.0
      (at until at + codeWidth).foreach(i =>
        sum += stored(i).toDouble * stored(i)
      )
      val length = math.max(math.sqrt(sum), 1e-12)
      (at until at + codeWidth).foreach(i =>
        unit(i) = (stored(i) / length).toFloat
      )
    }
    bf16(Shape.of(codes, codeWidth), unit)
  }
  private val codes = codebook.shape.dimensions.head.toInt
  // F32: the gather of other types takes rows of whole blocks of 32
  private val semanticTable = weights.floats(
    prefix + "prior_token_embedding.weight",
    prefix + "prior_token_embedding.weight",
    source.shape(prefix + "prior_token_embedding.weight")
  )
  private val semanticIn = affineLayer("prior_projector.net.0.proj")
  private val semanticOut = affineLayer("prior_projector.net.2")
  private val attention =
    new PlainAttention(ops, heads, heads, padded, Some(HeadWidth))

  /** The position table sampled at the centres of a `gridHeight × gridWidth`
    * grid (bilinear, the border held), `[tokens, width]`.
    */
  private def positions(gridHeight: Int, gridWidth: Int): Array[Float] = {
    val out = new Array[Float](gridHeight * gridWidth * width)
    def taps(index: Int, count: Int) = {
      val at =
        math.max(0.0, math.min(side - 1.0, (index + 0.5) / count * side - 0.5))
      val low = math.floor(at).toInt
      (low, math.min(low + 1, side - 1), (at - low).toFloat)
    }
    for {
      y <- 0 until gridHeight
      x <- 0 until gridWidth
    } {
      val (y0, y1, fy) = taps(y, gridHeight)
      val (x0, x1, fx) = taps(x, gridWidth)
      val at = (y * gridWidth + x) * width
      var c = 0
      while (c < width) {
        def value(row: Int, column: Int) =
          positionTable((row * side + column) * width + c)
        out(at + c) =
          (1 - fy) * ((1 - fx) * value(y0, x0) + fx * value(y0, x1)) +
            fy * ((1 - fx) * value(y1, x0) + fx * value(y1, x1))
        c += 1
      }
    }
    out
  }

  /** The codes of `image` (`[H, W, 3]`, values in [−1, 1], sides multiples of
    * `patch`), row-major.
    */
  def tokens(image: Tensor): Array[Int] = {
    val Seq(height, imageWidth, _) = image.shape.dimensions.map(_.toInt)
    require(
      height % patch == 0 && imageWidth % patch == 0,
      s"SigVQ: image ${image.shape}"
    )
    val (gridHeight, gridWidth) = (height / patch, imageWidth / patch)
    val count = gridHeight.toLong * gridWidth
    val scratch = new Scratch
    try {
      val packed = scratch.rows(count, 3L * patch * patch)
      ops.packPatches(image, patch, packed)
      val x = scratch.rows(count, width)
      affine(packed, patchEmbed, x)
      val placed = scratch.keep(
        ops.fromFloats(Shape.of(count, width), positions(gridHeight, gridWidth))
      )
      ops.add(x, placed, x)
      val (normed, projected) =
        (scratch.rows(count, width), scratch.rows(count, width))
      val Seq(q, k, v, attended) = Seq.fill(4)(scratch.rows(count, wide))
      blocks.foreach { block =>
        ops.layerNorm(
          x,
          Some(block.norm1._1),
          Some(block.norm1._2),
          Epsilon,
          normed
        )
        affine(normed, block.query, q)
        affine(normed, block.key, k)
        affine(normed, block.value, v)
        attention(q, k, v, attended)
        affine(attended, block.output, projected)
        ops.add(x, projected, x)
        ops.layerNorm(
          x,
          Some(block.norm2._1),
          Some(block.norm2._2),
          Epsilon,
          normed
        )
        val inner = scratch.rows(count, block.up.bias.shape.elementCount)
        affine(normed, block.up, inner)
        ops.activation(Activation.GeluErf, inner, inner)
        affine(inner, block.down, projected)
        ops.add(x, projected, x)
      }
      val code = scratch.rows(count, codeWidth)
      affine(x, toCode, code)
      val products = scratch.rows(count, codes)
      ops.linear(code, codebook, products)
      val values = ops.toFloats(products)
      Array.tabulate(count.toInt) { t =>
        var best = 0
        var c = 1
        while (c < codes) {
          if (values(t * codes + c) > values(t * codes + best)) best = c
          c += 1
        }
        best
      }
    } finally scratch.release()
  }

  /** The semantic features of `codes`, `[codes, outputs]`. */
  def semantic(codes: Array[Int]): Tensor = {
    val scratch = new Scratch
    try {
      val ids = scratch.keep(ops.fromInts(Shape.of(codes.length), codes))
      val embedded = scratch.rows(codes.length, outputs)
      ops.embedding(semanticTable, ids, embedded)
      val inner =
        scratch.rows(codes.length, semanticIn.bias.shape.elementCount)
      affine(embedded, semanticIn, inner)
      ops.activation(Activation.Silu, inner, inner)
      val out = ops.allocate(DType.F32, Shape.of(codes.length, outputs))
      affine(inner, semanticOut, out)
      out
    } finally scratch.release()
  }

  override def close(): Unit = {
    attention.close()
    super.close()
  }
}
