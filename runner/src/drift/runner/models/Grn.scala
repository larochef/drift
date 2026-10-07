package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** GRN's transformer, read off its weights (the released 2B: 2304 wide, 28
  * blocks, 18 heads of 128, MLPs of 8192, text features of 4096, 256 bits a
  * token).
  */
final case class GrnConfig(
    hidden: Int,
    blocks: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    intermediate: Int,
    textWidth: Int,
    /** Bits a token holds: the tokenizer's latent channels times its rounds. */
    bits: Int,
    timeWidth: Int
)

object GrnConfig {
  val Theta = 10000f
  val Epsilon = 1e-6f

  /** Whether the weights are a GRN transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("word_embed.weight") && source.has("pt_embedder.mlp.0.weight") &&
      source.has("block_chunks.0.module.0.attn.q_proj.weight")
}

/** GRN's transformer (bytedance's Generative Refinement Network,
  * `grn/models/grn.py`): given a picture's bits as they stand and how far the
  * refinement has come, the logits of every bit.
  *   - One sequence: a token per latent (its bits one-hot through
  *     `word_embed`), then the prompt (umT5's features through `text_proj`),
  *     then one token for the progress (a sinusoid of it through an MLP). Every
  *     token sees every other.
  *   - Blocks as Qwen 3's: RMS norm, attention (RMS norms on q and k per head,
  *     no biases), a residual; RMS norm, a SwiGLU MLP, a residual. No
  *     modulation.
  *   - RoPE on three axes (theta 10000; 21, 21 and 22 pairs of a head of 128,
  *     pairs interleaved): text token `i` at `(i, 0, 0)`, the progress token at
  *     `(512, 0, 0)`, the latents at `(600, y', x')` with rows and columns
  *     stretched onto the grid the model was trained to cover for the image's
  *     shape (129 × 129 for a square).
  *   - The head: an RMS norm and a linear to two logits a bit.
  */
final class Grn private (ops: Ops, source: WeightSource) extends AutoCloseable {

  private def dimensions(name: String) =
    source(name).shape.dimensions.map(_.toInt)
  if (!GrnConfig.holds(source))
    throw new FormatException(
      "no GRN transformer (word_embed.weight, pt_embedder.mlp.0.weight)"
    )
  private val blockNames = Iterator
    .from(0)
    .takeWhile(c => source.has(s"block_chunks.$c.module.0.attn.q_proj.weight"))
    .flatMap(c =>
      Iterator
        .from(0)
        .map(m => s"block_chunks.$c.module.$m")
        .takeWhile(at => source.has(s"$at.attn.q_proj.weight"))
    )
    .toSeq

  val config: GrnConfig = {
    val headDimension = dimensions(
      s"${blockNames.head}.attn.q_norm.weight"
    ).head
    val hidden = dimensions("word_embed.weight").head
    GrnConfig(
      hidden = hidden,
      blocks = blockNames.size,
      heads = hidden / headDimension,
      kvHeads = dimensions(
        s"${blockNames.head}.attn.k_proj.weight"
      ).head / headDimension,
      headDimension = headDimension,
      intermediate =
        dimensions(s"${blockNames.head}.mlp.gate_proj.weight").head,
      textWidth = dimensions("text_proj.weight").last,
      bits = dimensions("word_embed.weight").last / 2,
      timeWidth = dimensions("pt_embedder.mlp.0.weight").last
    )
  }
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = false)
  private val width = c.hidden.toLong
  private val (queryWidth, kvWidth) =
    (c.heads.toLong * c.headDimension, c.kvHeads.toLong * c.headDimension)

  private def floats(name: String): Tensor =
    weights.floats(name, name, Shape.of(source(name).shape.elementCount))
  final private case class Affine(weight: Tensor, bias: Tensor)
  private def affineLayer(name: String) =
    Affine(source(s"$name.weight"), floats(s"$name.bias"))
  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  final private class Block(at: String) {
    val query: Tensor = source(s"$at.attn.q_proj.weight")
    val key: Tensor = source(s"$at.attn.k_proj.weight")
    val value: Tensor = source(s"$at.attn.v_proj.weight")
    val output: Tensor = source(s"$at.attn.o_proj.weight")
    val queryNorm: Tensor = floats(s"$at.attn.q_norm.weight")
    val keyNorm: Tensor = floats(s"$at.attn.k_norm.weight")
    val gate: Tensor = source(s"$at.mlp.gate_proj.weight")
    val up: Tensor = source(s"$at.mlp.up_proj.weight")
    val down: Tensor = source(s"$at.mlp.down_proj.weight")
    val inputNorm: Tensor = floats(s"$at.input_layernorm.weight")
    val mlpNorm: Tensor = floats(s"$at.post_attention_layernorm.weight")
  }
  private val blocks = blockNames.map(new Block(_))
  private val bitsIn = affineLayer("word_embed")
  private val textIn = affineLayer("text_proj")
  private val timeIn = affineLayer("pt_embedder.mlp.0")
  private val timeOut = affineLayer("pt_embedder.mlp.2")
  private val headNorm = floats("head.norm.weight")
  private val head = affineLayer("head.proj")

  private val rope = {
    val pairs = c.headDimension / 2
    val former = pairs / 3
    Rope(
      GrnConfig.Theta,
      c.headDimension,
      RopeLayout.Interleaved,
      RopeSections.Axes(Seq(former, former, pairs - 2 * former))
    )
  }
  private val attention =
    new PlainAttention(ops, c.heads, c.kvHeads, c.headDimension)

  /** The prompt for every pass: umT5's features (`[L, textWidth]`) as tokens
    * (`[L, hidden]`).
    */
  def text(features: Tensor): Tensor = {
    val out =
      ops.allocate(DType.F32, Shape.of(features.shape.dimensions.head, width))
    affine(features, textIn, out)
    out
  }

  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def rows(dtype: DType, count: Long, columns: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(count, columns))
      held += tensor
      tensor
    }
    val x: Tensor = rows(DType.F32, tokens, width)
    val normed: Tensor = rows(DType.F32, tokens, width)
    val q: Tensor = rows(DType.F32, tokens, queryWidth)
    val rotatedQueries: Tensor = rows(DType.F32, tokens, queryWidth)
    val attended: Tensor = rows(DType.F32, tokens, queryWidth)
    val k: Tensor = rows(DType.F32, tokens, kvWidth)
    val rotatedKeys: Tensor = rows(DType.F32, tokens, kvWidth)
    val v: Tensor = rows(DType.F32, tokens, kvWidth)
    val projected: Tensor = rows(DType.F32, tokens, width)
    val mlpGate: Tensor = rows(DType.F32, tokens, c.intermediate)
    val mlpUp: Tensor = rows(DType.F32, tokens, c.intermediate)
    val positions: Tensor = rows(DType.I32, 3, tokens)
    val time: Tensor = rows(DType.F32, 1, timeIn.bias.shape.elementCount)
    def release(): Unit = held.foreach(ops.release)
  }
  // one set per sequence length: the prompt's and the negative prompt's
  // sequences alternate at every step
  private val buffers = mutable.Map.empty[Int, Buffers]
  private def buffersFor(tokens: Int): Buffers =
    buffers.getOrElseUpdate(
      tokens, {
        if (buffers.size >= 2) {
          buffers.values.foreach(_.release())
          buffers.clear()
        }
        new Buffers(tokens)
      }
    )

  /** The logits of every bit (`out`, `[latents, 2 × bits]`: bit `d`'s two at
    * `2 d` and `2 d + 1`) of a `gridHeight × gridWidth` picture whose bits
    * stand as `oneHot` (`[latents, 2 × bits]`, a one at `2 d + bit d`), under
    * `text` (from `text`), at `progress` (the share of bits already kept, 0 to
    * 1).
    */
  def logits(
      oneHot: Tensor,
      text: Tensor,
      progress: Float,
      gridHeight: Int,
      gridWidth: Int,
      out: Tensor
  ): Unit = {
    val latents = gridHeight * gridWidth
    val length = text.shape.dimensions.head.toInt
    val tokens = latents + length + 1
    require(
      oneHot.shape == Shape
        .of(latents, 2L * c.bits) && out.shape == oneHot.shape,
      s"logits: bits ${oneHot.shape}, out ${out.shape} for a $gridHeight × $gridWidth grid"
    )
    val w = buffersFor(tokens)
    val (heads, kvHeads, d) =
      (c.heads.toLong, c.kvHeads.toLong, c.headDimension.toLong)

    // [latents ; text ; progress]
    affine(oneHot, bitsIn, w.x.rows(0, latents))
    ops.copy(text, w.x.rows(latents, length))
    val half = c.timeWidth / 2
    val sinusoid = ops.fromFloats(
      Shape.of(1, c.timeWidth),
      Array.tabulate(2 * half) { i =>
        val angle = progress * math.exp(-math.log(10000) * (i % half) / half)
        (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
      }
    )
    try affine(sinusoid, timeIn, w.time)
    finally ops.release(sinusoid)
    ops.activation(Activation.Silu, w.time, w.time)
    affine(w.time, timeOut, w.x.rows(tokens - 1, 1))
    val (rows, columns) = Grn.stretched(gridHeight, gridWidth)
    ops.writeInts(
      w.positions.view(3L * tokens),
      Array.tabulate(3 * tokens) { i =>
        val (axis, token) = (i / tokens, i % tokens)
        if (token < latents)
          axis match {
            case 0 => Grn.LatentFrame
            case 1 => rows(token / gridWidth)
            case _ => columns(token % gridWidth)
          }
        else if (axis > 0) 0
        else if (token < tokens - 1) token - latents
        else Grn.ProgressPosition
      }
    )

    blocks.foreach { block =>
      ops.rmsNorm(w.x, block.inputNorm, GrnConfig.Epsilon, 0f, w.normed)
      ops.linears(
        w.normed,
        Seq(block.query, block.key, block.value),
        Seq(w.q, w.k, w.v)
      )
      ops.rmsNorm(
        w.q.view(tokens * heads, d),
        block.queryNorm,
        GrnConfig.Epsilon,
        0f,
        w.q.view(tokens * heads, d)
      )
      ops.rmsNorm(
        w.k.view(tokens * kvHeads, d),
        block.keyNorm,
        GrnConfig.Epsilon,
        0f,
        w.k.view(tokens * kvHeads, d)
      )
      ops.rope(
        w.q.view(tokens, heads, d),
        w.positions,
        rope,
        w.rotatedQueries.view(tokens, heads, d)
      )
      ops.rope(
        w.k.view(tokens, kvHeads, d),
        w.positions,
        rope,
        w.rotatedKeys.view(tokens, kvHeads, d)
      )
      attention(w.rotatedQueries, w.rotatedKeys, w.v, w.attended)
      ops.linear(w.attended, block.output, w.projected)
      ops.add(w.x, w.projected, w.x)
      ops.rmsNorm(w.x, block.mlpNorm, GrnConfig.Epsilon, 0f, w.normed)
      ops.linears(w.normed, Seq(block.gate, block.up), Seq(w.mlpGate, w.mlpUp))
      ops.gated(Activation.Silu, w.mlpGate, w.mlpUp, w.mlpGate)
      ops.linear(w.mlpGate, block.down, w.projected)
      ops.add(w.x, w.projected, w.x)
    }
    val normed = w.normed.rows(0, latents)
    ops.rmsNorm(w.x.rows(0, latents), headNorm, GrnConfig.Epsilon, 0f, normed)
    affine(normed, head, out)
  }

  def close(): Unit = {
    attention.close()
    buffers.values.foreach(_.release())
    weights.release()
    source.close()
  }
}

object Grn {

  /** Where the latents and the progress token sit on the first axis. */
  val LatentFrame = 600
  val ProgressPosition = 512

  /** The shapes (height over width) the model was trained on. */
  private val Templates: Seq[Double] =
    Seq(
      3.0,
      2.5,
      2.0,
      16.0 / 9,
      1.5,
      4.0 / 3,
      1.16,
      1.0,
      100.0 / 116,
      0.75,
      2.0 / 3,
      9.0 / 16,
      0.5,
      0.4,
      1.0 / 3
    ).map(r => (r * 1000).toInt / 1000.0)

  /** The row and column positions of a `gridHeight × gridWidth` grid: stretched
    * onto the largest grid of the nearest trained shape (rows of 225 for the
    * tallest, a third as many columns; 129 × 129 for a square), each rounded
    * half to even as the released code rounds.
    */
  def stretched(gridHeight: Int, gridWidth: Int): (Array[Int], Array[Int]) = {
    val shape = gridHeight.toDouble / gridWidth
    val template = Templates.minBy(t => math.abs(t - shape))
    val (extremeHeight, extremeRatio) = (225.0, 3.0)
    val across =
      math.sqrt(extremeHeight * (extremeHeight / extremeRatio) / template)
    val (high, wide) = ((template * across).toInt, across.toInt)
    (
      Array.tabulate(gridHeight)(y =>
        math.rint(y * (high.toDouble / gridHeight)).toInt
      ),
      Array.tabulate(gridWidth)(x =>
        math.rint(x * (wide.toDouble / gridWidth)).toInt
      )
    )
  }

  def open(ops: Ops, path: Path): Grn = {
    val source = WeightSource.open(ops, path)
    try new Grn(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
