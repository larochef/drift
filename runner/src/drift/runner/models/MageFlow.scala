package drift.runner.models

import drift.runner.diffusion.{Lora, LoraUpdates}
import drift.runner.formats.FormatException
import drift.runner.models.Flux2.Grid
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** Mage-Flow's transformer, read off its weights (released: 3072 wide, 12
  * blocks, 24 heads of 128, MLPs of 12288, text features of 2560, 128 latent
  * channels, one token per latent).
  */
final case class MageFlowConfig(
    hidden: Int,
    blocks: Int,
    heads: Int,
    headDimension: Int,
    intermediate: Int,
    textWidth: Int,
    latentChannels: Int,
    timeChannels: Int
) {

  /** RoPE's three axes (image, row, column) in pairs: an eighth of the head for
    * the image, the rest halved (16/56/56 for heads of 128).
    */
  def ropeAxes: RopeSections.Axes = {
    val image = headDimension / 16
    val side = (headDimension / 2 - image) / 2
    RopeSections.Axes(Seq(image, side, side))
  }
}

object MageFlowConfig {
  val Theta = 10000f
  val Epsilon = 1e-6f

  /** The released models' latent channels (the first Qwen Image, whose
    * transformer has the same names, takes 64).
    */
  val LatentChannels = 128

  /** Whether the weights are a Mage-Flow transformer's (or the first Qwen
    * Image's, the same blocks on packed latents).
    */
  def holds(source: WeightSource): Boolean =
    source.has("transformer_blocks.0.img_mod.1.weight") &&
      source.has("txt_norm.weight") &&
      !source.has("time_text_embed.addition_t_embedding.weight")

  def of(source: WeightSource): MageFlowConfig = {
    if (!holds(source))
      throw new FormatException(
        "no Mage-Flow transformer (transformer_blocks.0.img_mod.1.weight, txt_norm.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    val Seq(hidden, latentChannels) = dimensions("img_in.weight")
    val headDimension =
      dimensions("transformer_blocks.0.attn.norm_q.weight").head
    MageFlowConfig(
      hidden = hidden,
      blocks = Iterator
        .from(0)
        .takeWhile(i => source.has(s"transformer_blocks.$i.attn.to_q.weight"))
        .size,
      heads = hidden / headDimension,
      headDimension = headDimension,
      intermediate =
        dimensions("transformer_blocks.0.img_mlp.net.0.proj.weight").head,
      textWidth = dimensions("txt_in.weight").last,
      latentChannels = latentChannels,
      timeChannels =
        dimensions("time_text_embed.timestep_embedder.linear_1.weight").last
    )
  }
}

/** Mage-Flow's diffusion transformer (microsoft/Mage's `MageFlow`, the first
  * Qwen Image's blocks; sd-cpp's `mage_flow.hpp`), from the released names.
  *   - The text: the encoder's last hidden state through an RMS norm and
  *     `txt_in`, once per prompt (`embedText`).
  *   - The image: the target's latents then each reference's, one token per
  *     latent, through `img_in`. Positions on three RoPE axes (theta 10000,
  *     16/56/56 values, pairs interleaved): the image's index (0 the target,
  *     `k` reference `k`), its row and its column, both centred (`y − ⌈h /
  *     2⌉`). The text is not turned (position 0 on every axis).
  *   - The timestep: a sinusoid of `1000 σ` (cosines first) through an MLP; the
  *     released model runs in BF16 and was trained on that rounding, so σ and
  *     the frequencies are rounded to BF16 before the product, the cosines and
  *     sines after. Each block has its own modulation of each stream from it
  *     (shift, scale and gate, twice).
  *   - Double-stream blocks only: text and image each normed and modulated on
  *     their own, joint attention over [text ; image] (per-head RMS norms on q
  *     and k), each stream's own projection and GELU (tanh) MLP, gated
  *     residuals. Every linear has a bias.
  *   - The final layer on the target's tokens alone: a layer norm modulated by
  *     (scale ; shift) from the timestep, then `proj_out`.
  * The weights are used as stored (BF16 through BF16 GEMMs, quants through
  * their own kernels), the activations F32.
  */
final class MageFlow private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: MageFlowConfig = MageFlowConfig.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = MageFlowConfig.Epsilon
  private val width = c.hidden.toLong

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  /** A linear layer with a bias, by its name (a LoRA's site). */
  final private case class Affine(name: String, weight: Tensor, bias: Tensor)

  private val layers = mutable.Map.empty[String, Affine]
  private def affineLayer(name: String, outputs: Long): Affine = {
    val layer =
      Affine(name, source(s"$name.weight"), floats(s"$name.bias", outputs))
    layers(name) = layer
    layer
  }

  /** One stream's weights in a block: `q`, `k`, `v`, `out` and the head norms
    * under `attn.` by the stream's own names, the MLP and the modulation under
    * the stream's prefix.
    */
  final private class Stream(
      block: String,
      stream: String,
      q: String,
      k: String,
      v: String,
      out: String,
      norms: String
  ) {
    val query: Affine = affineLayer(s"$block.attn.$q", width)
    val key: Affine = affineLayer(s"$block.attn.$k", width)
    val value: Affine = affineLayer(s"$block.attn.$v", width)
    val output: Affine = affineLayer(s"$block.attn.$out", width)
    val queryNorm: Tensor =
      floats(s"$block.attn.norm_${norms}q.weight", c.headDimension)
    val keyNorm: Tensor =
      floats(s"$block.attn.norm_${norms}k.weight", c.headDimension)
    val mlpIn: Affine =
      affineLayer(s"$block.${stream}_mlp.net.0.proj", c.intermediate)
    val mlpOut: Affine = affineLayer(s"$block.${stream}_mlp.net.2", width)
    val modulation: Affine = affineLayer(s"$block.${stream}_mod.1", 6 * width)
  }

  final private class Block(index: Int) {
    private val prefix = s"transformer_blocks.$index"
    val image =
      new Stream(prefix, "img", "to_q", "to_k", "to_v", "to_out.0", "")
    val text = new Stream(
      prefix,
      "txt",
      "add_q_proj",
      "add_k_proj",
      "add_v_proj",
      "to_add_out",
      "added_"
    )
  }

  private val blocks = (0 until c.blocks).map(new Block(_))
  private val textNorm = floats("txt_norm.weight", c.textWidth)
  private val textIn = affineLayer("txt_in", width)
  private val imageIn = affineLayer("img_in", width)
  private val timeIn =
    affineLayer("time_text_embed.timestep_embedder.linear_1", width)
  private val timeOut =
    affineLayer("time_text_embed.timestep_embedder.linear_2", width)
  private val finalModulation = affineLayer("norm_out.linear", 2 * width)
  private val finalOut = affineLayer("proj_out", c.latentChannels)

  // ---- LoRAs ---------------------------------------------------------------------

  /** The active LoRAs' updates of each linear, by its name. */
  private val updates = new LoraUpdates(ops)

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they name the linears as the checkpoint does (diffusers' names). Returns
    * the targets that matched nothing or a weight of another shape, left
    * unapplied.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        val fits = layers.get(target).exists { layer =>
          layer.weight.shape.dimensions.head == pair.up.shape.dimensions.head &&
          layer.weight.shape.last == pair.down.shape.last
        }
        (target, pair.copy(scale = pair.scale * multiplier), fits)
      }
    )
    updates.use(
      placed
        .collect { case (target, pair, true) => target -> pair }
        .groupMap(_._1)(_._2)
    )
    placed.collect { case (target, _, false) => target }.distinct
  }

  /** `out = x · weightᵀ + bias`, and the active LoRAs' updates. */
  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    updates(x, layer.name, out)
    ops.addRow(out, layer.bias, out)
  }

  private val rope = Rope(
    MageFlowConfig.Theta,
    c.headDimension,
    RopeLayout.Interleaved,
    c.ropeAxes
  )
  private val attention = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  /** The attention's values go through an F16 cache (largest 65504), which
    * Mage-Flow's outgrow (the released model runs in BF16; sd-cpp scales them
    * by 1/128). Attention is linear in the values, so values beyond this are
    * scaled down by a power of two before the cache and the output back.
    */
  private val ValueLimit = 8192f

  /** The power of two `values` are divided by (1 when they fit), applied. */
  private def fitValues(values: Tensor): Float = {
    val largest = ops.maxAbs(values)
    if (largest <= ValueLimit) 1f
    else {
      val factor =
        math
          .pow(2, math.ceil(math.log(largest / ValueLimit) / math.log(2)))
          .toFloat
      ops.scale(values, 1 / factor, values)
      factor
    }
  }

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
    private val wide = tokens * width
    val x: Tensor = flat(DType.F32, wide)
    val normed: Tensor = flat(DType.F32, wide)
    val q: Tensor = flat(DType.F32, wide)
    val k: Tensor = flat(DType.F32, wide)
    val v: Tensor = flat(DType.F32, wide)
    val rotatedKeys: Tensor = flat(DType.F32, wide)
    val attended: Tensor = flat(DType.F32, wide)
    val projected: Tensor = flat(DType.F32, wide)
    val mlp: Tensor = flat(DType.F32, tokens.toLong * c.intermediate)
    val positions: Tensor = flat(DType.I32, 3L * tokens)
    // attention's keys and values, one page for the whole sequence (pages
    // hold a multiple of 16 tokens)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, c.heads, c.headDimension)
    val pageTable: Tensor = {
      val tensor = ops.fromInts(Shape.of(1), Array(0))
      held += tensor
      tensor
    }
    def release(): Unit = {
      held.foreach(ops.release)
      ops.release(cache.keys)
      ops.release(cache.values)
    }
  }

  private var buffers = Option.empty[Buffers]
  private def buffersFor(tokens: Int): Buffers =
    buffers.filter(_.tokens >= tokens).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens)
      buffers = Some(created)
      created
    }

  /** The timestep's vectors, kept between steps: its embedding and every
    * modulation made from it.
    */
  private val small = mutable.ArrayBuffer.empty[Tensor]
  private def vector(elements: Long): Tensor = {
    val tensor = ops.allocate(DType.F32, Shape.of(1, elements))
    small += tensor
    tensor
  }
  private val timeInner = vector(width)
  private val time = vector(width)
  private val modulations =
    blocks.map(_ => (vector(6 * width), vector(6 * width)))
  private val finalParts = vector(2 * width)

  /** The prompt's text for every step: the encoder's last hidden state (`[L,
    * textWidth]`) into `out` (`[L, hidden]`).
    */
  def embedText(features: Tensor, out: Tensor): Unit = {
    val tokens = features.shape.dimensions.head
    require(
      features.shape == Shape.of(tokens, c.textWidth) &&
        out.shape == Shape.of(tokens, width),
      s"embedText: features ${features.shape}, out ${out.shape}"
    )
    val normed = ops.allocate(DType.F32, features.shape)
    try {
      ops.rmsNorm(features, textNorm, epsilon, 0f, normed)
      affine(normed, textIn, out)
    } finally ops.release(normed)
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (the target's on
    * `grid`, row-major), given the prompt's `text` (`[L, hidden]` from
    * `embedText`) and the references' latents on their grids, into `out` (like
    * `latents`).
    */
  def velocity(
      latents: Tensor,
      grid: Grid,
      references: Seq[(Tensor, Grid)],
      text: Tensor,
      timestep: Float,
      out: Tensor
  ): Unit = {
    val length = text.shape.dimensions.head.toInt
    val images = grid.tokens + references.map(_._2.tokens).sum
    val tokens = length + images
    require(
      latents.shape == Shape.of(grid.tokens, c.latentChannels) &&
        out.shape == latents.shape &&
        references.forall((r, g) =>
          r.shape == Shape.of(g.tokens, c.latentChannels)
        ),
      s"velocity: latents ${latents.shape}, out ${out.shape} on $grid, references ${references
          .map((r, g) => s"${r.shape} on $g")}"
    )
    val w = buffersFor(tokens)

    // the timestep through its MLP, then SiLU once for every modulation
    val sinusoid = ops.fromFloats(
      Shape.of(1, c.timeChannels),
      MageFlow.sinusoid(timestep, c.timeChannels)
    )
    try affine(sinusoid, timeIn, timeInner)
    finally ops.release(sinusoid)
    ops.activation(Activation.Silu, timeInner, timeInner)
    affine(timeInner, timeOut, time)
    ops.activation(Activation.Silu, time, time)
    blocks.zip(modulations).foreach { (block, parts) =>
      affine(time, block.text.modulation, parts._1)
      affine(time, block.image.modulation, parts._2)
    }
    affine(time, finalModulation, finalParts)
    def part(modulation: Tensor, i: Int) =
      modulation
        .view(modulation.shape.last / width, width)
        .rows(i, 1)
        .view(width)

    // [text ; target ; references]
    val stream = w.x.prefix(tokens, width)
    ops.copy(text, stream.rows(0, length))
    affine(latents, imageIn, stream.rows(length, grid.tokens))
    references.foldLeft(length + grid.tokens) { case (at, (reference, g)) =>
      affine(reference, imageIn, stream.rows(at, g.tokens))
      at + g.tokens
    }
    val axes = Seq.tabulate(3)(_ => mutable.ArrayBuilder.make[Int])
    (0 until length).foreach(_ => axes.foreach(_ += 0))
    (grid +: references.map(_._2)).zipWithIndex.foreach { (g, index) =>
      val (top, left) =
        (g.height - g.height / 2, g.width - g.width / 2)
      (0 until g.tokens).foreach { i =>
        axes(0) += index
        axes(1) += i / g.width - top
        axes(2) += i % g.width - left
      }
    }
    val positions = w.positions.prefix(3, tokens)
    ops.writeInts(
      positions.view(3L * tokens),
      axes.flatMap(_.result()).toArray
    )

    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    def headNorm(x: Tensor, weight: Tensor) = {
      val rows = x.shape.dimensions.head * heads
      ops.rmsNorm(x.view(rows, d), weight, epsilon, 0f, x.view(rows, d))
    }
    val (q, k, v, attended, projected) = (
      w.q.prefix(tokens, width),
      w.k.prefix(tokens, width),
      w.v.prefix(tokens, width),
      w.attended.prefix(tokens, width),
      w.projected.prefix(tokens, width)
    )
    // the text's rows then the images'
    val spans = Seq((0, length), (length, images))

    blocks.zip(modulations).foreach { (block, parts) =>
      val streams = Seq((block.text, parts._1), (block.image, parts._2))
      streams.zip(spans).foreach { case ((s, mod), (first, n)) =>
        def rows(t: Tensor) = t.rows(first, n)
        val normed = rows(w.normed.prefix(tokens, width))
        ops.layerNorm(rows(stream), None, None, epsilon, normed)
        ops.modulate(normed, part(mod, 1), part(mod, 0), normed)
        ops.linears(
          normed,
          Seq(s.query.weight, s.key.weight, s.value.weight),
          Seq(rows(q), rows(k), rows(v))
        )
        Seq(s.query -> q, s.key -> k, s.value -> v).foreach { (layer, y) =>
          updates(normed, layer.name, rows(y))
          ops.addRow(rows(y), layer.bias, rows(y))
        }
        headNorm(rows(q), s.queryNorm)
        headNorm(rows(k), s.keyNorm)
      }
      val rotatedQueries = w.normed.prefix(tokens, heads, d)
      val rotatedKeys = w.rotatedKeys.prefix(tokens, heads, d)
      ops.rope(q.view(tokens, heads, d), positions, rope, rotatedQueries)
      ops.rope(k.view(tokens, heads, d), positions, rope, rotatedKeys)
      val factor = fitValues(v)
      ops.cacheWrite(
        rotatedKeys,
        v.view(tokens, heads, d),
        w.cache,
        w.pageTable,
        0
      )
      ops.attention(
        rotatedQueries,
        w.cache,
        w.pageTable,
        0,
        tokens,
        attention,
        attended.view(tokens, heads, d)
      )
      if (factor != 1f) ops.scale(attended, factor, attended)
      streams.zip(spans).foreach { case ((s, mod), (first, n)) =>
        def rows(t: Tensor) = t.rows(first, n)
        val x = rows(stream)
        affine(rows(attended), s.output, rows(projected))
        ops.gatedAdd(x, rows(projected), part(mod, 2))
        val normed = rows(w.normed.prefix(tokens, width))
        ops.layerNorm(x, None, None, epsilon, normed)
        ops.modulate(normed, part(mod, 4), part(mod, 3), normed)
        val up = w.mlp.prefix(n, c.intermediate)
        affine(normed, s.mlpIn, up)
        ops.activation(Activation.GeluTanh, up, up)
        affine(up, s.mlpOut, rows(projected))
        ops.gatedAdd(x, rows(projected), part(mod, 5))
      }
    }

    // the final layer on the target's tokens: (scale ; shift)
    val target = stream.rows(length, grid.tokens)
    val normed = w.normed.prefix(grid.tokens, width)
    ops.layerNorm(target, None, None, epsilon, normed)
    ops.modulate(normed, part(finalParts, 0), part(finalParts, 1), normed)
    affine(normed, finalOut, out)
  }

  def close(): Unit = {
    updates.close()
    small.foreach(ops.release)
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object MageFlow {

  /** `value` rounded to BF16 (to nearest, ties to even). */
  private def bf16(value: Float): Float = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    java.lang.Float.intBitsToFloat(
      ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16
    )
  }

  /** The timestep's sinusoid as the released model computes it in BF16: σ and
    * each frequency rounded, their product times 1000 in F32, its cosine (first
    * half) and sine (second) rounded.
    */
  def sinusoid(timestep: Float, channels: Int): Array[Float] = {
    val half = channels / 2
    val level = bf16(timestep)
    Array.tabulate(2 * half) { i =>
      val frequency =
        bf16(math.exp(-math.log(10000) * (i % half) / half).toFloat)
      val angle = (level * frequency * 1000f).toDouble
      bf16((if (i < half) math.cos(angle) else math.sin(angle)).toFloat)
    }
  }

  def open(ops: Ops, path: Path): MageFlow = {
    val source = WeightSource.open(ops, path)
    try new MageFlow(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
