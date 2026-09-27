package drift.runner.models

import drift.runner.diffusion.{Lora, LoraUpdates}
import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** Qwen Image 2.1's transformer, read off its weights: `hidden` wide, `blocks`
  * deep, heads of `headDimension`, a SwiGLU of `intermediate`, text features of
  * `textWidth`, latents of `latentChannels` (one token per latent pixel).
  */
final case class QwenImage21Config(
    hidden: Int,
    blocks: Int,
    heads: Int,
    headDimension: Int,
    intermediate: Int,
    textWidth: Int,
    latentChannels: Int
) {

  /** RoPE's three axes (frame, row, column) in pairs: an eighth of the head for
    * the frame, the rest halved (16/56/56 for heads of 128).
    */
  def ropeAxes: RopeSections.Axes = {
    val frame = headDimension / 16
    val side = (headDimension / 2 - frame) / 2
    RopeSections.Axes(Seq(frame, side, side))
  }
}

object QwenImage21Config {
  val Theta = 10000f
  val Epsilon = 1e-6f

  /** Whether the weights are a Qwen Image 2.1 transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("txt_in.text_norm.weight")

  def of(source: WeightSource): QwenImage21Config = {
    if (!holds(source))
      throw new FormatException(
        "no Qwen Image 2.1 transformer (txt_in.text_norm.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    val Seq(hidden, latentChannels) = dimensions("img_in.weight")
    val headDimension =
      dimensions("transformer_blocks.0.attn.norm_q.weight").head
    QwenImage21Config(
      hidden = hidden,
      blocks = Iterator
        .from(0)
        .takeWhile(i => source.has(s"transformer_blocks.$i.attn.to_q.weight"))
        .size,
      heads = hidden / headDimension,
      headDimension = headDimension,
      intermediate =
        if (source.has("transformer_blocks.0.img_mlp.gate_up.weight"))
          dimensions("transformer_blocks.0.img_mlp.gate_up.weight").head / 2
        else dimensions("transformer_blocks.0.img_mlp.proj.weight").head,
      textWidth = dimensions("txt_in.in_layer.weight").last,
      latentChannels = latentChannels
    )
  }
}

/** A prompt's keys and values at every block (the text and the reference images
  * are modulated from `t = 0` and attend to what comes before them, so they do
  * not change from step to step), and room after them for an image of
  * `imageTokens`. Pages of 16 tokens: each block has its own for the prompt;
  * the image's pages are shared by all blocks, each rewriting them before it
  * attends (the prompt's last page, when partial, takes the image's first keys
  * at every block alike). The image's frame position is `frame`.
  */
final class QwenImage21Prefix private[models] (
    ops: Ops,
    val tokens: Int,
    val frame: Int,
    val imageTokens: Int,
    val cache: KvCache,
    /** I32 `[blocks, pages]`: each block's page table. */
    val pageTables: Tensor
) extends AutoCloseable {

  def pageTable(block: Int): Tensor = {
    val pages = pageTables.shape.last
    pageTables.rows(block, 1).view(pages)
  }

  def close(): Unit = {
    ops.release(cache.keys)
    ops.release(cache.values)
    ops.release(pageTables)
  }
}

/** A reference image in a prompt's prefix (editing): it takes the text's slots
  * `at until at + slots` (the text encoder's image tokens, each standing for 2
  * × 2 latents), replaced by its `latents` (`[gridHeight × gridWidth,
  * latentChannels]`, row-major, as the VAE encodes it).
  */
final case class PrefixReference(
    at: Int,
    slots: Int,
    latents: Tensor,
    gridHeight: Int,
    gridWidth: Int
)

/** Queries `from until until` of a run, attending causally or to every slot up
  * to `until`.
  */
final private[models] case class Segment(from: Int, until: Int, causal: Boolean)

/** Qwen Image 2.1's diffusion transformer (diffusers'
  * `QwenImage21Transformer2DModel`, sd-cpp's `qwen_image_2_1.hpp`) from
  * ComfyUI's single file (diffusers' names, the MLP's gate and proj fused as
  * `gate_up`) or diffusers' own.
  *   - One sequence [text ; image], block-causal: the text causal, each image
  *     block (a reference among the text, then the image made) seeing itself
  *     whole and everything before it. Every block is single-stream: layer
  *     norm, a scale (no shift), attention with q/k RMS norm and a three-axis
  *     RoPE, a `tanh`-gated residual; the same for a SwiGLU MLP.
  *   - One modulation for all the blocks, from the timestep: the image takes
  *     the step's, the text `t = 0`'s. So the text never changes: `prefix` runs
  *     it once per prompt and keeps its keys and values, `velocity` runs the
  *     image's tokens alone at each step.
  *   - Positions: the text's `(p, p, p)`, counting on by one; an image block's
  *     frame is the position reached, its rows and columns centred on zero, and
  *     the text after it resumes past its longer side.
  */
final class QwenImage21 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: QwenImage21Config = QwenImage21Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = QwenImage21Config.Epsilon
  private val PageSize = 16

  private def weight(name: String): Tensor = source(name)
  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  /** A block's weights: the MLP's gate and proj as row views of `gate_up` when
    * it is fused.
    */
  final private class Block(i: Int) {
    val prefix = s"transformer_blocks.$i"
    val q: Tensor = weight(s"$prefix.attn.to_q.weight")
    val k: Tensor = weight(s"$prefix.attn.to_k.weight")
    val v: Tensor = weight(s"$prefix.attn.to_v.weight")
    val o: Tensor = weight(s"$prefix.attn.to_out.0.weight")
    val qNorm: Tensor = floats(s"$prefix.attn.norm_q.weight", c.headDimension)
    val kNorm: Tensor = floats(s"$prefix.attn.norm_k.weight", c.headDimension)
    val (gate, proj) =
      if (source.has(s"$prefix.img_mlp.gate_up.weight")) {
        val fused = weight(s"$prefix.img_mlp.gate_up.weight")
        (
          fused.rows(0, c.intermediate),
          fused.rows(c.intermediate, c.intermediate)
        )
      } else
        (
          weight(s"$prefix.img_mlp.gate_layer.weight"),
          weight(s"$prefix.img_mlp.proj.weight")
        )
    val out: Tensor = weight(s"$prefix.img_mlp.out.weight")

    def sites: Seq[(String, Tensor)] = Seq(
      "attn.to_q" -> q,
      "attn.to_k" -> k,
      "attn.to_v" -> v,
      "attn.to_out.0" -> o,
      "img_mlp.gate_layer" -> gate,
      "img_mlp.proj" -> proj,
      "img_mlp.out" -> out
    ).map((part, tensor) => s"$prefix.$part" -> tensor)
  }

  private val blocks = (0 until c.blocks).map(new Block(_))
  private val textNorm = floats("txt_in.text_norm.weight", c.textWidth)

  /** Every linear by its diffusers name (a LoRA's site). */
  private val siteWeights: Map[String, Tensor] =
    (blocks.flatMap(_.sites) ++ Seq(
      "txt_in.in_layer",
      "txt_in.out_layer",
      "img_in",
      "time_text_embed.timestep_embedder.linear_1",
      "time_text_embed.timestep_embedder.linear_2",
      "modulation.1",
      "norm_out.linear",
      "proj_out"
    ).map(name => name -> weight(s"$name.weight"))).toMap

  private val rope = Rope(
    QwenImage21Config.Theta,
    c.headDimension,
    RopeLayout.Interleaved,
    c.ropeAxes
  )
  private def attention(causal: Boolean) = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal,
    None,
    None,
    None
  )

  // ---- LoRAs ---------------------------------------------------------------------

  private val updates = new LoraUpdates(ops)

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read diffusers' names and ComfyUI's (the MLP's `gate_up` whole, its
    * update split by the rows of its `up`). Returns the targets that matched
    * nothing or a weight of another shape, left unapplied. A prompt's prefix
    * made before depends on the LoRAs of its time.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        val sites =
          if (target.endsWith(".img_mlp.gate_up")) {
            val block = target.stripSuffix(".gate_up")
            Seq(s"$block.gate_layer", s"$block.proj")
          } else Seq(target)
        val known = sites.filter(siteWeights.contains)
        val fits = known.size == sites.size &&
          known.map(siteWeights(_).shape.dimensions.head).sum ==
          pair.up.shape.dimensions.head &&
          known.forall(siteWeights(_).shape.last == pair.down.shape.last)
        (
          target,
          pair.copy(scale = pair.scale * multiplier),
          known.filter(_ => fits)
        )
      }
    )
    val bySite = placed.flatMap { (_, pair, sites) =>
      val rows = sites.map(siteWeights(_).shape.dimensions.head)
      sites.zip(rows.scanLeft(0L)(_ + _)).zip(rows).map {
        case ((site, first), count) =>
          site -> pair.copy(up = pair.up.rows(first, count))
      }
    }
    updates.use(bySite.groupMap(_._1)(_._2))
    textModulationMade.foreach(_.release())
    textModulationMade = None
    placed.collect { case (target, _, Nil) => target }.distinct
  }

  /** `out = x · weightᵀ` for the linear at `site`, and its LoRA updates. */
  private def linear(x: Tensor, site: String, out: Tensor): Unit = {
    ops.linear(x, siteWeights(site), out)
    updates(x, site, out)
  }

  // ---- buffers ---------------------------------------------------------------------

  /** Flat buffers for `tokens` rows, grown when too small; each use takes the
    * `prefix` it needs.
    */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(dtype: DType, elements: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(elements))
      held += tensor
      tensor
    }
    private val wide = tokens.toLong * math.max(c.hidden, c.textWidth)
    private val mlp = tokens.toLong * c.intermediate
    val x: Tensor = flat(DType.F32, wide)
    val normed: Tensor = flat(DType.F32, wide)
    val q: Tensor = flat(DType.F32, wide)
    val k: Tensor = flat(DType.F32, wide)
    val v: Tensor = flat(DType.F32, wide)
    val rotatedKeys: Tensor = flat(DType.F32, wide)
    val attended: Tensor = flat(DType.F32, wide)
    val projected: Tensor = flat(DType.F32, wide)
    val mlpGate: Tensor = flat(DType.F32, mlp)
    val mlpUp: Tensor = flat(DType.F32, mlp)
    val positions: Tensor = flat(DType.I32, 3L * tokens)
    def release(): Unit = held.foreach(ops.release)
  }

  private var buffers = Option.empty[Buffers]
  private def buffersFor(tokens: Int): Buffers =
    buffers.filter(_.tokens >= tokens).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens)
      buffers = Some(created)
      created
    }

  // ---- the modulation --------------------------------------------------------------

  /** The blocks' scales and gates (the gates through `tanh`), and the final
    * norm's scale, each F32 `[hidden]`; `zero` the shift `modulate` takes.
    */
  final private case class Modulation(
      attentionScale: Tensor,
      attentionGate: Tensor,
      mlpScale: Tensor,
      mlpGate: Tensor,
      finalScale: Tensor,
      zero: Tensor
  ) {
    def release(): Unit =
      Seq(attentionScale, attentionGate, mlpScale, mlpGate, finalScale, zero)
        .foreach(ops.release)
  }

  /** The modulation at `timestep` (σ in [0, 1]): a sinusoid of `1000 σ`
    * (cosines first) through the timestep MLP, then SiLU and the shared
    * `modulation` linear (scale, gate, scale, gate) and `norm_out.linear`.
    */
  private def modulation(timestep: Float): Modulation = {
    val f = c.hidden.toLong
    val half = siteWeights(
      "time_text_embed.timestep_embedder.linear_1"
    ).shape.last.toInt / 2
    val sinusoid = Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(10000) * (i % half) / half).toFloat
      val angle = timestep * 1000f * frequency
      if (i < half) math.cos(angle).toFloat else math.sin(angle).toFloat
    }
    val small = mutable.ArrayBuffer.empty[Tensor]
    def vector(elements: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(1, elements))
      small += tensor
      tensor
    }
    try {
      val embedded = ops.fromFloats(Shape.of(1, 2L * half), sinusoid)
      small += embedded
      val (inner, time) = (vector(f), vector(f))
      linear(embedded, "time_text_embed.timestep_embedder.linear_1", inner)
      ops.activation(Activation.Silu, inner, inner)
      linear(inner, "time_text_embed.timestep_embedder.linear_2", time)
      ops.activation(Activation.Silu, time, time)
      val (table, scale) = (vector(4 * f), vector(f))
      linear(time, "modulation.1", table)
      linear(time, "norm_out.linear", scale)
      val values = ops.toFloats(table)
      def chunk(i: Int, gate: Boolean) = {
        val part = values.slice(i * c.hidden, (i + 1) * c.hidden)
        ops.fromFloats(
          Shape.of(f),
          if (gate) part.map(v => math.tanh(v.toDouble).toFloat) else part
        )
      }
      Modulation(
        chunk(0, gate = false),
        chunk(1, gate = true),
        chunk(2, gate = false),
        chunk(3, gate = true),
        ops.fromFloats(Shape.of(f), ops.toFloats(scale)),
        ops.fromFloats(Shape.of(f), new Array[Float](c.hidden))
      )
    } finally small.foreach(ops.release)
  }

  /** The modulation of the text (`t = 0`), which every prompt shares; remade
    * after the LoRAs change (they may target its linears).
    */
  private var textModulationMade = Option.empty[Modulation]
  private def textModulation: Modulation =
    textModulationMade.getOrElse {
      val made = modulation(0f)
      textModulationMade = Some(made)
      made
    }

  // ---- the blocks ------------------------------------------------------------------

  /** Block `i` on `x` (`[rows, hidden]`, in place, in slots `start` on): its
    * keys and values written through `prefix`'s page table, each of the
    * `segments`' queries attending to the slots up to its end (causally for
    * text).
    */
  private def block(
      i: Int,
      x: Tensor,
      m: Modulation,
      positions: Tensor,
      prefix: QwenImage21Prefix,
      start: Int,
      segments: Seq[Segment],
      w: Buffers
  ): Unit = {
    val b = blocks(i)
    val Seq(rows, width) = x.shape.dimensions
    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    val normed = w.normed.prefix(rows, width)
    ops.layerNorm(x, None, None, epsilon, normed)
    ops.modulate(normed, m.attentionScale, m.zero, normed)
    val (q, k, v) =
      (
        w.q.prefix(rows, width),
        w.k.prefix(rows, width),
        w.v.prefix(rows, width)
      )
    ops.linears(normed, Seq(b.q, b.k, b.v), Seq(q, k, v))
    Seq("attn.to_q" -> q, "attn.to_k" -> k, "attn.to_v" -> v)
      .foreach((part, y) => updates(normed, s"${b.prefix}.$part", y))
    ops.rmsNorm(
      q.view(rows * heads, d),
      b.qNorm,
      epsilon,
      0f,
      q.view(rows * heads, d)
    )
    ops.rmsNorm(
      k.view(rows * heads, d),
      b.kNorm,
      epsilon,
      0f,
      k.view(rows * heads, d)
    )
    // the rotated queries in place of the normed input, which is spent
    val rotatedQueries = w.normed.prefix(rows, heads, d)
    val rotatedKeys = w.rotatedKeys.prefix(rows, heads, d)
    ops.rope(q.view(rows, heads, d), positions, rope, rotatedQueries)
    ops.rope(k.view(rows, heads, d), positions, rope, rotatedKeys)
    val pageTable = prefix.pageTable(i)
    ops.cacheWrite(
      rotatedKeys,
      v.view(rows, heads, d),
      prefix.cache,
      pageTable,
      start
    )
    val attended = w.attended.prefix(rows, width)
    segments.foreach { segment =>
      val count = (segment.until - segment.from).toLong
      ops.attention(
        rotatedQueries.rows(segment.from, count),
        prefix.cache,
        pageTable,
        start + segment.from,
        start + segment.until,
        attention(segment.causal),
        attended.view(rows, heads, d).rows(segment.from, count)
      )
    }
    val projected = w.projected.prefix(rows, width)
    linear(attended, s"${b.prefix}.attn.to_out.0", projected)
    ops.gatedAdd(x, projected, m.attentionGate)
    ops.layerNorm(x, None, None, epsilon, normed.view(rows, width))
    ops.modulate(
      normed.view(rows, width),
      m.mlpScale,
      m.zero,
      normed.view(rows, width)
    )
    val (gate, up) = (
      w.mlpGate.prefix(rows, c.intermediate),
      w.mlpUp.prefix(rows, c.intermediate)
    )
    ops.linears(normed.view(rows, width), Seq(b.gate, b.proj), Seq(gate, up))
    updates(normed.view(rows, width), s"${b.prefix}.img_mlp.gate_layer", gate)
    updates(normed.view(rows, width), s"${b.prefix}.img_mlp.proj", up)
    ops.gated(Activation.Silu, gate, up, gate)
    linear(gate, s"${b.prefix}.img_mlp.out", projected)
    ops.gatedAdd(x, projected, m.mlpGate)
  }

  /** The prompt's prefix for images of `imageTokens`: `text` (`[L, textWidth]`,
    * the encoder's last hidden states, the template's system turn dropped)
    * projected, each reference's slots in it (`references`, in order) replaced
    * by its latents, then run through every block, block-causally, its keys and
    * values kept.
    */
  def prefix(
      text: Tensor,
      imageTokens: Int,
      references: Seq[PrefixReference] = Nil
  ): QwenImage21Prefix = {
    val Seq(length, width) = text.shape.dimensions.map(_.toInt)
    require(width == c.textWidth, s"prefix: text ${text.shape}")
    references.foreach(r =>
      require(
        r.latents.shape == Shape
          .of(r.gridHeight * r.gridWidth, c.latentChannels) &&
          r.gridHeight * r.gridWidth == 4 * r.slots,
        s"prefix: a reference of ${r.gridHeight} × ${r.gridWidth} latents ${r.latents.shape} for ${r.slots} slots"
      )
    )
    // the joint sequence: text runs and reference blocks, in order
    val runs = {
      val parts = Vector.newBuilder[Either[(Int, Int), PrefixReference]]
      var from = 0
      references.sortBy(_.at).foreach { r =>
        if (r.at > from) parts += Left(from -> r.at)
        parts += Right(r)
        from = r.at + r.slots
      }
      if (length > from) parts += Left(from -> length)
      parts.result()
    }
    val tokens = runs.map {
      case Left((from, until)) => until - from
      case Right(r)            => r.gridHeight * r.gridWidth
    }.sum
    val prefixPages = (tokens + PageSize - 1) / PageSize
    val imagePages =
      (tokens + imageTokens + PageSize - 1) / PageSize - prefixPages
    val cache = ops.allocateCache(
      c.blocks * prefixPages + imagePages,
      PageSize,
      c.heads,
      c.headDimension
    )
    val tables = Array.tabulate(c.blocks * (prefixPages + imagePages)) { n =>
      val (i, page) =
        (n / (prefixPages + imagePages), n % (prefixPages + imagePages))
      if (page < prefixPages) i * prefixPages + page
      else c.blocks * prefixPages + page - prefixPages
    }
    // positions (frame, row, column) and the attention's segments
    val frames, rowsAt, columnsAt = Array.newBuilder[Int]
    val segments = Vector.newBuilder[Segment]
    var (position, slot) = (0, 0)
    runs.foreach {
      case Left((from, until)) =>
        val n = until - from
        val counted = Array.range(position, position + n)
        frames ++= counted; rowsAt ++= counted; columnsAt ++= counted
        segments += Segment(slot, slot + n, causal = true)
        position += n
        slot += n
      case Right(r) =>
        val n = r.gridHeight * r.gridWidth
        val (top, left) =
          (r.gridHeight - r.gridHeight / 2, r.gridWidth - r.gridWidth / 2)
        frames ++= Array.fill(n)(position)
        rowsAt ++= Array.tabulate(n)(_ / r.gridWidth - top)
        columnsAt ++= Array.tabulate(n)(_ % r.gridWidth - left)
        segments += Segment(slot, slot + n, causal = false)
        position += math.max(r.gridHeight, r.gridWidth)
        slot += n
    }
    val prefix = new QwenImage21Prefix(
      ops,
      tokens,
      position,
      imageTokens,
      cache,
      ops.fromInts(Shape.of(c.blocks, prefixPages + imagePages), tables)
    )
    try {
      val w = buffersFor(math.max(math.max(length, tokens), imageTokens))
      val x = w.x.prefix(tokens, c.hidden)
      val normed = w.normed.prefix(length, c.textWidth)
      ops.rmsNorm(text, textNorm, epsilon, 1f, normed)
      val inner = w.q.prefix(length, c.hidden)
      linear(normed, "txt_in.in_layer", inner)
      ops.activation(Activation.GeluTanh, inner, inner)
      // the text's rows go straight to their places; the references' through
      // img_in
      val projected = w.k.prefix(length, c.hidden)
      linear(inner, "txt_in.out_layer", projected)
      var at = 0
      runs.foreach {
        case Left((from, until)) =>
          ops.copy(projected.rows(from, until - from), x.rows(at, until - from))
          at += until - from
        case Right(r) =>
          val n = r.gridHeight * r.gridWidth
          linear(r.latents, "img_in", x.rows(at, n))
          at += n
      }
      val positions = w.positions.prefix(3, tokens)
      ops.writeInts(
        positions.view(3L * tokens),
        frames.result() ++ rowsAt.result() ++ columnsAt.result()
      )
      (0 until c.blocks).foreach(i =>
        block(i, x, textModulation, positions, prefix, 0, segments.result(), w)
      )
      prefix
    } catch {
      case error: Throwable =>
        prefix.close()
        throw error
    }
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (`[gridHeight ×
    * gridWidth, latentChannels]`, row-major), given the prompt's `prefix`, into
    * `out` (like `latents`).
    */
  def velocity(
      latents: Tensor,
      prefix: QwenImage21Prefix,
      timestep: Float,
      gridHeight: Int,
      gridWidth: Int,
      out: Tensor
  ): Unit = {
    val images = gridHeight * gridWidth
    require(
      latents.shape == Shape.of(images, c.latentChannels) &&
        out.shape == latents.shape && prefix.imageTokens == images,
      s"velocity: latents ${latents.shape}, out ${out.shape} for a $gridHeight × $gridWidth grid " +
        s"and a prefix for ${prefix.imageTokens} image tokens"
    )
    val w = buffersFor(math.max(prefix.tokens, images))
    val m = modulation(timestep)
    try {
      val x = w.x.prefix(images, c.hidden)
      linear(latents, "img_in", x)
      // (frame, row, column): the frame after the prompt, the grid centred
      val positions = w.positions.prefix(3, images)
      val (top, left) =
        (gridHeight - gridHeight / 2, gridWidth - gridWidth / 2)
      ops.writeInts(
        positions.view(3L * images),
        Array.fill(images)(prefix.frame) ++
          Array.tabulate(images)(_ / gridWidth - top) ++
          Array.tabulate(images)(_ % gridWidth - left)
      )
      (0 until c.blocks).foreach(i =>
        block(
          i,
          x,
          m,
          positions,
          prefix,
          prefix.tokens,
          Seq(Segment(0, images, causal = false)),
          w
        )
      )
      val normed = w.normed.prefix(images, c.hidden)
      ops.layerNorm(x, None, None, epsilon, normed)
      ops.modulate(normed, m.finalScale, m.zero, normed)
      linear(normed, "proj_out", out)
    } finally m.release()
  }

  def close(): Unit = {
    textModulationMade.foreach(_.release())
    updates.close()
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object QwenImage21 {

  def open(ops: Ops, path: Path): QwenImage21 = {
    val source = WeightSource.open(ops, path)
    try new QwenImage21(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
