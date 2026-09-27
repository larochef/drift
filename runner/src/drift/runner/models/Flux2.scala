package drift.runner.models

import drift.runner.diffusion.{Lora, LoraUpdates}
import drift.runner.formats.FormatException
import drift.runner.models.Flux2.Grid
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** FLUX.2's diffusion transformer, its sizes read from the weights (Klein 9B:
  * 4096 wide, 32 heads of 128, 8 double-stream and 24 single-stream blocks,
  * text features 12288 wide, 128 latent features; dev: 6144 wide, 48 heads, 8
  * and 48 blocks, text features 15360 wide). `guidance`: it embeds a distilled
  * guidance scale (dev), Klein does not.
  */
final case class Flux2Config(
    hidden: Int,
    heads: Int,
    headDimension: Int,
    doubleBlocks: Int,
    singleBlocks: Int,
    intermediate: Int,
    textWidth: Int,
    latentChannels: Int,
    timeChannels: Int,
    guidance: Boolean
)

object Flux2Config {
  val Theta = 2000f
  val Epsilon = 1e-6f

  /** Whether the weights are a FLUX.2 transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("double_stream_modulation_img.lin.weight")

  def of(source: WeightSource): Flux2Config = {
    if (!holds(source))
      throw new FormatException(
        "no FLUX.2 transformer (double_stream_modulation_img.lin.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    def count(blocks: String, weight: String) =
      Iterator.from(0).takeWhile(i => source.has(s"$blocks.$i.$weight")).size
    val Seq(hidden, latentChannels) = dimensions("img_in.weight")
    val headDimension = Seq("weight", "scale")
      .map(s => s"double_blocks.0.img_attn.norm.query_norm.$s")
      .find(source.has)
      .map(source(_).shape.dimensions.head.toInt)
      .getOrElse(
        throw new FormatException("no FLUX.2 query norm in double_blocks.0")
      )
    Flux2Config(
      hidden = hidden,
      heads = hidden / headDimension,
      headDimension = headDimension,
      doubleBlocks = count("double_blocks", "img_attn.qkv.weight"),
      singleBlocks = count("single_blocks", "linear1.weight"),
      intermediate = dimensions("double_blocks.0.img_mlp.0.weight").head / 2,
      textWidth = dimensions("txt_in.weight").last,
      latentChannels = latentChannels,
      timeChannels = dimensions("time_in.in_layer.weight").last,
      guidance = source.has("guidance_in.in_layer.weight")
    )
  }
}

/** FLUX.2's diffusion transformer (BFL's `flux2`, diffusers'
  * `Flux2Transformer2DModel`, sd-cpp's `flux.hpp`), from the original names.
  *   - The text: the encoder's taps side by side (`[L, 12288]`) through
  *     `txt_in`, once per prompt (`embedText`).
  *   - The image: the target's packed latents then each reference's, through
  *     `img_in`. Positions on four RoPE axes (theta 2000, 32 values each, pairs
  *     interleaved): the text `(0, 0, 0, l)`, the target `(0, y, x, 0)`,
  *     reference `k` (from 1) `(10 k, y, x, 0)`.
  *   - The timestep: a sinusoid (cosines first) of σ × 1000 through an MLP,
  *     plus (dev) the guidance scale × 1000 the same way through its own MLP;
  *     one modulation per block kind from it, shared by every block of the
  *     kind.
  *   - Double-stream blocks: text and image each normed and modulated on their
  *     own, joint attention over [text ; image] (per-head RMS norms on q and
  *     k), each stream's own projection and SwiGLU MLP, gated residuals.
  *   - Single-stream blocks on [text ; image]: one fused linear to q, k, v and
  *     the MLP's gate and up, attention, and one linear from the attention and
  *     the MLP side by side.
  *   - The final layer on the target's tokens alone (the references' outputs
  *     are dropped). No biases anywhere. The weights are used as stored (BF16
  *     through BF16 GEMMs, quants through their own kernels); the fused ones
  *     are split into row views.
  */
final class Flux2 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: Flux2Config = Flux2Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = Flux2Config.Epsilon

  private def matrix(name: String): Tensor = source(name)

  /** A norm's weight, as ComfyUI (`.weight`) or BFL (`.scale`) names it. */
  private def normWeight(name: String): Tensor = {
    val full = Seq("weight", "scale")
      .map(s => s"$name.$s")
      .find(source.has)
      .getOrElse(throw new FormatException(s"no $name.weight or $name.scale"))
    weights.floats(full, full, Shape.of(c.headDimension))
  }

  private val width = c.hidden.toLong

  /** One stream's attention and MLP weights in a double-stream block. */
  final private class Stream(val prefix: String) {
    private val qkv = matrix(s"${prefix}_attn.qkv.weight")
    val (q, k, v) =
      (qkv.rows(0, width), qkv.rows(width, width), qkv.rows(2 * width, width))
    val qNorm: Tensor = normWeight(s"${prefix}_attn.norm.query_norm")
    val kNorm: Tensor = normWeight(s"${prefix}_attn.norm.key_norm")
    val proj: Tensor = matrix(s"${prefix}_attn.proj.weight")
    private val mlpIn = matrix(s"${prefix}_mlp.0.weight")
    val (mlpGate, mlpUp) = (
      mlpIn.rows(0, c.intermediate),
      mlpIn.rows(c.intermediate, c.intermediate)
    )
    val mlpDown: Tensor = matrix(s"${prefix}_mlp.2.weight")
  }

  final private class DoubleBlock(index: Int) {
    val image = new Stream(s"double_blocks.$index.img")
    val text = new Stream(s"double_blocks.$index.txt")
  }

  final private class SingleBlock(index: Int) {
    val prefix = s"single_blocks.$index"
    private val linear1 = matrix(s"$prefix.linear1.weight")
    private def part(first: Long, count: Long) = linear1.rows(first, count)
    val (q, k, v) =
      (part(0, width), part(width, width), part(2 * width, width))
    val (mlpGate, mlpUp) = (
      part(3 * width, c.intermediate),
      part(3 * width + c.intermediate, c.intermediate)
    )
    val qNorm: Tensor = normWeight(s"$prefix.norm.query_norm")
    val kNorm: Tensor = normWeight(s"$prefix.norm.key_norm")
    val linear2: Tensor = matrix(s"$prefix.linear2.weight")
  }

  private val doubles = (0 until c.doubleBlocks).map(new DoubleBlock(_))
  private val singles = (0 until c.singleBlocks).map(new SingleBlock(_))
  private val textIn = matrix("txt_in.weight")
  private val imageIn = matrix("img_in.weight")
  private val timeIn = matrix("time_in.in_layer.weight")
  private val timeOut = matrix("time_in.out_layer.weight")
  private val guidanceIn =
    Option.when(c.guidance)(matrix("guidance_in.in_layer.weight"))
  private val guidanceOut =
    Option.when(c.guidance)(matrix("guidance_in.out_layer.weight"))
  private val imageModulation = matrix(
    "double_stream_modulation_img.lin.weight"
  )
  private val textModulation = matrix("double_stream_modulation_txt.lin.weight")
  private val singleModulation = matrix("single_stream_modulation.lin.weight")
  private val finalModulation = matrix("final_layer.adaLN_modulation.1.weight")
  private val finalOut = matrix("final_layer.linear.weight")

  // ---- LoRAs ---------------------------------------------------------------------

  /** The active LoRAs' updates, by site: a weight in BFL's naming, a fused
    * one's parts as `<block>.q`, `.k`, `.v`, `.proj`, `.gate`, `.up`, `.down`,
    * `.linear2`.
    */
  private val updates = new LoraUpdates(ops)

  /** `up` matrices made at load (a final modulation's halves swapped). */
  private var madeForLoras = Seq.empty[Tensor]

  private val DoubleOriginal =
    """double_blocks\.(\d+)\.(img|txt)_(attn\.qkv|attn\.proj|mlp\.0|mlp\.2)""".r
  private val SingleOriginal = """single_blocks\.(\d+)\.(linear1|linear2)""".r
  private val DoubleDiffusers =
    """transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.add_q_proj|attn\.add_k_proj|attn\.add_v_proj|attn\.to_out\.0|attn\.to_add_out|ff\.linear_in|ff\.linear_out|ff_context\.linear_in|ff_context\.linear_out)""".r
  private val SingleDiffusers =
    """single_transformer_blocks\.(\d+)\.attn\.(to_qkv_mlp_proj|to_out)""".r
  private val Kohya = """lora_unet_(.+)""".r

  /** Whole weights: each naming → the site. */
  private val Wholes: Map[String, String] = {
    val same = Seq(
      "img_in",
      "txt_in",
      "time_in.in_layer",
      "time_in.out_layer",
      "guidance_in.in_layer",
      "guidance_in.out_layer",
      "double_stream_modulation_img.lin",
      "double_stream_modulation_txt.lin",
      "single_stream_modulation.lin",
      "final_layer.adaLN_modulation.1",
      "final_layer.linear"
    ).map(name => name -> name)
    (same ++ Seq(
      "x_embedder" -> "img_in",
      "context_embedder" -> "txt_in",
      "time_guidance_embed.timestep_embedder.linear_1" -> "time_in.in_layer",
      "time_guidance_embed.timestep_embedder.linear_2" -> "time_in.out_layer",
      "time_guidance_embed.guidance_embedder.linear_1" -> "guidance_in.in_layer",
      "time_guidance_embed.guidance_embedder.linear_2" -> "guidance_in.out_layer",
      "double_stream_modulation_img.linear" -> "double_stream_modulation_img.lin",
      "double_stream_modulation_txt.linear" -> "double_stream_modulation_txt.lin",
      "single_stream_modulation.linear" -> "single_stream_modulation.lin",
      "proj_out" -> "final_layer.linear"
    )).toMap
  }

  /** Every site's weight (a fused one's row view). */
  private lazy val siteWeights: Map[String, Tensor] = {
    val streams = doubles.flatMap(b => Seq(b.image, b.text)).flatMap { s =>
      Seq(
        "q" -> s.q,
        "k" -> s.k,
        "v" -> s.v,
        "proj" -> s.proj,
        "gate" -> s.mlpGate,
        "up" -> s.mlpUp,
        "down" -> s.mlpDown
      ).map((part, weight) => s"${s.prefix}.$part" -> weight)
    }
    val blocks = singles.flatMap { b =>
      Seq(
        "q" -> b.q,
        "k" -> b.k,
        "v" -> b.v,
        "gate" -> b.mlpGate,
        "up" -> b.mlpUp,
        "linear2" -> b.linear2
      ).map((part, weight) => s"${b.prefix}.$part" -> weight)
    }
    (streams ++ blocks ++ Seq(
      "img_in" -> imageIn,
      "txt_in" -> textIn,
      "time_in.in_layer" -> timeIn,
      "time_in.out_layer" -> timeOut,
      "double_stream_modulation_img.lin" -> imageModulation,
      "double_stream_modulation_txt.lin" -> textModulation,
      "single_stream_modulation.lin" -> singleModulation,
      "final_layer.adaLN_modulation.1" -> finalModulation,
      "final_layer.linear" -> finalOut
    ) ++ guidanceIn.map("guidance_in.in_layer" -> _) ++
      guidanceOut.map("guidance_in.out_layer" -> _)).toMap
  }

  /** kohya's names (`lora_unet_double_blocks_0_img_attn_qkv`): BFL's with every
    * dot an underscore.
    */
  private lazy val kohyaNames: Map[String, String] = {
    val doubles = (0 until c.doubleBlocks).flatMap(n =>
      for {
        stream <- Seq("img", "txt")
        part <- Seq("attn.qkv", "attn.proj", "mlp.0", "mlp.2")
      } yield s"double_blocks.$n.${stream}_$part"
    )
    val singles = (0 until c.singleBlocks).flatMap(n =>
      Seq("linear1", "linear2").map(part => s"single_blocks.$n.$part")
    )
    (doubles ++ singles ++ Wholes.values)
      .map(name => name.replace('.', '_') -> name)
      .toMap
  }

  /** The sites a LoRA target's update covers, taking the rows of its `up` in
    * turn; empty when it names no weight of this model.
    */
  private def placement(target: String): Seq[String] = {
    def parts(block: String, names: String*) = names.map(n => s"$block.$n")
    // the fused input (q, k, v, the MLP's gate and up) or the output linear
    def single(n: String, input: Boolean) =
      if (input) parts(s"single_blocks.$n", "q", "k", "v", "gate", "up")
      else parts(s"single_blocks.$n", "linear2")
    target match {
      case DoubleOriginal(n, stream, part) =>
        val block = s"double_blocks.$n.$stream"
        part match {
          case "attn.qkv"  => parts(block, "q", "k", "v")
          case "attn.proj" => parts(block, "proj")
          case "mlp.0"     => parts(block, "gate", "up")
          case _           => parts(block, "down")
        }
      case DoubleDiffusers(n, part) =>
        val (stream, names) = part match {
          case "attn.to_q"            => ("img", Seq("q"))
          case "attn.to_k"            => ("img", Seq("k"))
          case "attn.to_v"            => ("img", Seq("v"))
          case "attn.add_q_proj"      => ("txt", Seq("q"))
          case "attn.add_k_proj"      => ("txt", Seq("k"))
          case "attn.add_v_proj"      => ("txt", Seq("v"))
          case "attn.to_out.0"        => ("img", Seq("proj"))
          case "attn.to_add_out"      => ("txt", Seq("proj"))
          case "ff.linear_in"         => ("img", Seq("gate", "up"))
          case "ff.linear_out"        => ("img", Seq("down"))
          case "ff_context.linear_in" => ("txt", Seq("gate", "up"))
          case _                      => ("txt", Seq("down"))
        }
        parts(s"double_blocks.$n.$stream", names*)
      case SingleOriginal(n, part)  => single(n, part == "linear1")
      case SingleDiffusers(n, part) => single(n, part == "to_qkv_mlp_proj")
      case "norm_out.linear"        => Seq("final_layer.adaLN_modulation.1")
      case Kohya(name) => kohyaNames.get(name).toSeq.flatMap(placement)
      case other       => Wholes.get(other).toSeq
    }
  }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read BFL's and ComfyUI's names, diffusers' and kohya's. A fused
    * weight's update is split by the rows of its `up`. Returns the targets that
    * matched nothing or a weight of another shape, left unapplied.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    madeForLoras.foreach(ops.release)
    madeForLoras = Nil
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        val sites = placement(target).filter(siteWeights.contains)
        val weights = sites.map(siteWeights)
        val fits = sites.nonEmpty &&
          weights.map(_.shape.dimensions.head).sum ==
          pair.up.shape.dimensions.head &&
          weights.forall(_.shape.last == pair.down.shape.last)
        (
          target,
          pair.copy(scale = pair.scale * multiplier),
          sites.filter(_ => fits)
        )
      }
    )
    val made = scala.collection.mutable.ArrayBuffer.empty[Tensor]
    val bySite = placed.flatMap { (target, pair, sites) =>
      // diffusers' final modulation is (scale ; shift), BFL's (shift ; scale)
      val up =
        if (target != "norm_out.linear" || sites.isEmpty) pair.up
        else {
          val swapped = ops.allocate(pair.up.dtype, pair.up.shape)
          made += swapped
          ops.copy(pair.up.rows(width, width), swapped.rows(0, width))
          ops.copy(pair.up.rows(0, width), swapped.rows(width, width))
          swapped
        }
      val rows = sites.map(siteWeights(_).shape.dimensions.head)
      sites.zip(rows.scanLeft(0L)(_ + _)).zip(rows).map {
        case ((site, first), count) =>
          site -> pair.copy(up = up.rows(first, count))
      }
    }
    madeForLoras = made.toSeq
    updates.use(bySite.groupMap(_._1)(_._2))
    placed.collect { case (target, _, Nil) => target }.distinct
  }

  private def lora(x: Tensor, site: String, out: Tensor): Unit =
    updates(x, site, out)

  private val rope = Rope(
    Flux2Config.Theta,
    c.headDimension,
    RopeLayout.Interleaved,
    RopeSections.Axes(Seq.fill(4)(c.headDimension / 8))
  )
  private val attention = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

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
    val joined: Tensor = flat(DType.F32, wide + mlp)
    val positions: Tensor = flat(DType.I32, 4L * tokens)
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

  /** The prompt's text for every step: the encoder's taps side by side (`[L,
    * textWidth]`) into `out` (`[L, hidden]`).
    */
  def embedText(features: Tensor, out: Tensor): Unit = {
    val tokens = features.shape.dimensions.head
    require(
      features.shape == Shape.of(tokens, c.textWidth) &&
        out.shape == Shape.of(tokens, width),
      s"embedText: features ${features.shape}, out ${out.shape}"
    )
    ops.linear(features, textIn, out)
    lora(features, "txt_in", out)
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (the target's packed
    * latents on `grid`), given the prompt's `text` (`[L, hidden]` from
    * `embedText`), the references' packed latents on their grids and the
    * distilled `guidance` scale (read only when the model embeds one), into
    * `out` (like `latents`).
    */
  def velocity(
      latents: Tensor,
      grid: Grid,
      references: Seq[(Tensor, Grid)],
      text: Tensor,
      timestep: Float,
      guidance: Float,
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
    val small = mutable.ArrayBuffer.empty[Tensor]
    def vector(elements: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(1, elements))
      small += tensor
      tensor
    }
    try {
      // the timestep (and the guidance scale): a sinusoid (cosines first)
      // through its MLP, the two summed, then SiLU once for every modulation
      val half = c.timeChannels / 2
      def embed(value: Float, first: Tensor, second: Tensor, site: String) = {
        val sinusoid = Array.tabulate(2 * half) { i =>
          val frequency =
            math.exp(-math.log(10000) * (i % half) / half).toFloat
          val angle = value * 1000f * frequency
          if (i < half) math.cos(angle).toFloat else math.sin(angle).toFloat
        }
        val embedded = ops.fromFloats(Shape.of(1, 2L * half), sinusoid)
        small += embedded
        val (inner, result) = (vector(width), vector(width))
        ops.linear(embedded, first, inner)
        lora(embedded, s"$site.in_layer", inner)
        ops.activation(Activation.Silu, inner, inner)
        ops.linear(inner, second, result)
        lora(inner, s"$site.out_layer", result)
        result
      }
      val time = embed(timestep, timeIn, timeOut, "time_in")
      for {
        first <- guidanceIn
        second <- guidanceOut
      } ops.add(time, embed(guidance, first, second, "guidance_in"), time)
      ops.activation(Activation.Silu, time, time)
      val (imageMod, textMod, singleMod, finalMod) =
        (
          vector(6 * width),
          vector(6 * width),
          vector(3 * width),
          vector(2 * width)
        )
      Seq(
        (imageModulation, imageMod, "double_stream_modulation_img.lin"),
        (textModulation, textMod, "double_stream_modulation_txt.lin"),
        (singleModulation, singleMod, "single_stream_modulation.lin"),
        (finalModulation, finalMod, "final_layer.adaLN_modulation.1")
      ).foreach { (weight, modulation, site) =>
        ops.linear(time, weight, modulation)
        lora(time, site, modulation)
      }
      def part(modulation: Tensor, i: Int) =
        modulation
          .view(modulation.shape.last / width, width)
          .rows(i, 1)
          .view(width)

      // [text ; target ; references]
      val stream = w.x.prefix(tokens, width)
      ops.copy(text, stream.rows(0, length))
      ops.linear(latents, imageIn, stream.rows(length, grid.tokens))
      lora(latents, "img_in", stream.rows(length, grid.tokens))
      references.foldLeft(length + grid.tokens) { case (at, (reference, g)) =>
        ops.linear(reference, imageIn, stream.rows(at, g.tokens))
        lora(reference, "img_in", stream.rows(at, g.tokens))
        at + g.tokens
      }
      val axes = Seq.tabulate(4)(_ => mutable.ArrayBuilder.make[Int])
      def place(t: Int, ys: Int => Int, xs: Int => Int, l: Int => Int, n: Int) =
        (0 until n).foreach { i =>
          axes(0) += t
          axes(1) += ys(i)
          axes(2) += xs(i)
          axes(3) += l(i)
        }
      place(0, _ => 0, _ => 0, identity, length)
      (grid +: references.map(_._2)).zipWithIndex.foreach { (g, k) =>
        place(10 * k, _ / g.width, _ % g.width, _ => 0, g.tokens)
      }
      val positions = w.positions.prefix(4, tokens)
      ops.writeInts(
        positions.view(4L * tokens),
        axes.flatMap(_.result()).toArray
      )

      val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
      def headNorm(x: Tensor, weight: Tensor) = {
        val rows = x.shape.dimensions.head * heads
        ops.rmsNorm(x.view(rows, d), weight, epsilon, 0f, x.view(rows, d))
      }
      // q, k, v of every token → attended, all [tokens, width]
      def attend(q: Tensor, k: Tensor, v: Tensor, attended: Tensor): Unit = {
        val rotatedQueries = w.normed.prefix(tokens, heads, d)
        val rotatedKeys = w.rotatedKeys.prefix(tokens, heads, d)
        ops.rope(q.view(tokens, heads, d), positions, rope, rotatedQueries)
        ops.rope(k.view(tokens, heads, d), positions, rope, rotatedKeys)
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

      doubles.foreach { block =>
        val streams = Seq((block.text, textMod), (block.image, imageMod))
        streams.zip(spans).foreach { case ((s, mod), (first, n)) =>
          def rows(t: Tensor) = t.rows(first, n)
          val normed = rows(w.normed.prefix(tokens, width))
          ops.layerNorm(rows(stream), None, None, epsilon, normed)
          ops.modulate(normed, part(mod, 1), part(mod, 0), normed)
          ops.linears(
            normed,
            Seq(s.q, s.k, s.v),
            Seq(rows(q), rows(k), rows(v))
          )
          Seq("q" -> q, "k" -> k, "v" -> v)
            .foreach((part, y) => lora(normed, s"${s.prefix}.$part", rows(y)))
          headNorm(rows(q), s.qNorm)
          headNorm(rows(k), s.kNorm)
        }
        attend(q, k, v, attended)
        streams.zip(spans).foreach { case ((s, mod), (first, n)) =>
          def rows(t: Tensor) = t.rows(first, n)
          val x = rows(stream)
          ops.linear(rows(attended), s.proj, rows(projected))
          lora(rows(attended), s"${s.prefix}.proj", rows(projected))
          ops.gatedAdd(x, rows(projected), part(mod, 2))
          val normed = rows(w.normed.prefix(tokens, width))
          ops.layerNorm(x, None, None, epsilon, normed)
          ops.modulate(normed, part(mod, 4), part(mod, 3), normed)
          val (gate, up) = (
            w.mlpGate.prefix(n, c.intermediate),
            w.mlpUp.prefix(n, c.intermediate)
          )
          ops.linears(normed, Seq(s.mlpGate, s.mlpUp), Seq(gate, up))
          lora(normed, s"${s.prefix}.gate", gate)
          lora(normed, s"${s.prefix}.up", up)
          ops.gated(Activation.Silu, gate, up, gate)
          ops.linear(gate, s.mlpDown, rows(projected))
          lora(gate, s"${s.prefix}.down", rows(projected))
          ops.gatedAdd(x, rows(projected), part(mod, 5))
        }
      }

      singles.foreach { block =>
        val normed = w.normed.prefix(tokens, width)
        ops.layerNorm(stream, None, None, epsilon, normed)
        ops.modulate(normed, part(singleMod, 1), part(singleMod, 0), normed)
        val (gate, up) = (
          w.mlpGate.prefix(tokens, c.intermediate),
          w.mlpUp.prefix(tokens, c.intermediate)
        )
        ops.linears(
          normed,
          Seq(block.q, block.k, block.v, block.mlpGate, block.mlpUp),
          Seq(q, k, v, gate, up)
        )
        Seq("q" -> q, "k" -> k, "v" -> v, "gate" -> gate, "up" -> up)
          .foreach((part, y) => lora(normed, s"${block.prefix}.$part", y))
        headNorm(q, block.qNorm)
        headNorm(k, block.kNorm)
        attend(q, k, v, attended)
        ops.gated(Activation.Silu, gate, up, gate)
        val joined = w.joined.prefix(tokens, width + c.intermediate)
        ops.concatColumns(Seq(attended, gate), joined)
        ops.linear(joined, block.linear2, projected)
        lora(joined, s"${block.prefix}.linear2", projected)
        ops.gatedAdd(stream, projected, part(singleMod, 2))
      }

      // the final layer on the target's tokens: rows (shift ; scale)
      val target = stream.rows(length, grid.tokens)
      val normed = w.normed.prefix(grid.tokens, width)
      ops.layerNorm(target, None, None, epsilon, normed)
      ops.modulate(normed, part(finalMod, 1), part(finalMod, 0), normed)
      ops.linear(normed, finalOut, out)
      lora(normed, "final_layer.linear", out)
    } finally small.foreach(ops.release)
  }

  def close(): Unit = {
    updates.close()
    madeForLoras.foreach(ops.release)
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object Flux2 {

  /** A packed image's grid: its tokens are `height × width`, row-major. */
  final case class Grid(height: Int, width: Int) {
    def tokens: Int = height * width
  }

  def open(ops: Ops, path: Path): Flux2 = {
    val source = WeightSource.open(ops, path)
    try new Flux2(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
