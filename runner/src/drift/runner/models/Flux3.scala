package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.lora.{Lora, LoraUpdates}
import drift.runner.models.Flux2.Grid
import drift.runner.ops.*
import drift.runner.tensor.*
import drift.runner.weights.WeightSource

import java.nio.file.Path

/** FLUX 3 Image Klein's diffusion transformer, its sizes read from the weights
  * (5.6B: 3072 wide, 24 heads of 128, 5 blocks per mode then 28 shared ones, an
  * MLP 3 times as wide, text features 20480 wide, 128 latent features).
  * `flowMap`: the step-distilled model, which is also told the noise level each
  * step jumps to; the base model is not.
  */
final case class Flux3Config(
    hidden: Int,
    heads: Int,
    headDimension: Int,
    modeBlocks: Int,
    sharedBlocks: Int,
    intermediate: Int,
    textWidth: Int,
    latentChannels: Int,
    timeChannels: Int,
    flowMap: Boolean
)

object Flux3Config {
  val Theta = 10000.0
  val Epsilon = 1e-6f

  /** Where the images sit on the RoPE's height and width axes: half their
    * latent row and column.
    */
  val ImagePositionScale = 0.5

  /** Whether the weights are a FLUX 3 transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("early_stream_modulations.image.lin.weight")

  def of(source: WeightSource): Flux3Config = {
    if (!holds(source))
      throw new FormatException(
        "no FLUX 3 transformer (early_stream_modulations.image.lin.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    def count(blocks: String) =
      Iterator
        .from(0)
        .takeWhile(i => source.has(s"$blocks.$i.q_proj.weight"))
        .size
    val Seq(hidden, latentChannels) = dimensions("emb_in.image.weight")
    val headDimension =
      dimensions("single_blocks.0.norm.query_norm.scale").head
    Flux3Config(
      hidden = hidden,
      heads = hidden / headDimension,
      headDimension = headDimension,
      modeBlocks = count("txt_mode_blocks"),
      sharedBlocks = count("single_blocks"),
      intermediate = dimensions("single_blocks.0.mlp_in.weight").head / 2,
      textWidth = dimensions("txt_in.weight").last,
      latentChannels = latentChannels,
      timeChannels = dimensions("time_in.in_layer.weight").last,
      flowMap = source.has("ref_time_in.in_layer.weight")
    )
  }
}

/** FLUX 3 Image Klein's diffusion transformer (BFL's `flux3.dit.DiT`), from the
  * original names.
  *   - Tokens of three modes: the target's packed latents through
  *     `emb_in.image`, every reference's through `emb_in.image_cond`, the text
  *     encoder's taps side by side (`[L, 20480]`) through `txt_in`, once per
  *     prompt (`embedText`).
  *   - Positions on four RoPE axes (theta 10000, 16 pairs each, pairs
  *     adjacent), fractional: the text `(0, 0, 0, l)`, the target `(0, y / 2, x
  *     / 2, 0)`, reference `k` (from 0) `(0, y / 2, x / 2, −1 − k)`.
  *   - One time vector per mode: a sinusoid (cosines first) of the mode's
  *     timestep × 1000, in BF16 as BFL's pack holds it, through `time_in` — σ
  *     for the target, 0 for the references and the text; the flow map adds the
  *     level the step jumps to through `ref_time_in` to all three.
  *   - Every block is one kind: layer norm, a shift and scale, q, k and v (RMS
  *     norms on the heads, then RoPE), attention, and beside it a SwiGLU MLP;
  *     the attention's projection and the MLP's output summed into one gated
  *     residual. First each mode through its own blocks, attending to its own
  *     tokens alone (all the references together), with the mode's early
  *     modulation; then every token through the shared blocks, each mode's
  *     tokens with their mode's modulation.
  *   - The final layer on the target's tokens. No biases anywhere; the weights
  *     are used as stored, the MLP's input split into row views.
  */
final class Flux3 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: Flux3Config = Flux3Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = Flux3Config.Epsilon
  private val width = c.hidden.toLong

  private def matrix(name: String): Tensor = source(name)

  private def headNormWeight(name: String): Tensor =
    weights.floats(name, name, Shape.of(c.headDimension))

  final private class Block(val prefix: String) {
    val q: Tensor = matrix(s"$prefix.q_proj.weight")
    val k: Tensor = matrix(s"$prefix.k_proj.weight")
    val v: Tensor = matrix(s"$prefix.v_proj.weight")
    val qNorm: Tensor = headNormWeight(s"$prefix.norm.query_norm.scale")
    val kNorm: Tensor = headNormWeight(s"$prefix.norm.key_norm.scale")
    val attentionOut: Tensor = matrix(s"$prefix.attn_out.weight")
    private val mlpIn = matrix(s"$prefix.mlp_in.weight")
    val (mlpGate, mlpUp) = (
      mlpIn.rows(0, c.intermediate),
      mlpIn.rows(c.intermediate, c.intermediate)
    )
    val mlpOut: Tensor = matrix(s"$prefix.mlp_out.weight")
  }

  /** The three modes, in the order their tokens take: the target, the
    * references, the text. `name` is the weights', `modulation` its
    * modulations'.
    */
  private enum Mode(val name: String, val modulation: String) {
    case Image extends Mode("image", "image")
    case Condition extends Mode("image_cond", "image_cond")
    case Text extends Mode("text", "txt")
  }

  private def modeBlocks(prefix: String) =
    (0 until c.modeBlocks).map(i => new Block(s"$prefix.$i"))

  private val earlyBlocks: Map[Mode, Seq[Block]] = Map(
    Mode.Image -> modeBlocks("content_mode_blocks.image"),
    Mode.Condition -> modeBlocks("content_mode_blocks.image_cond"),
    Mode.Text -> modeBlocks("txt_mode_blocks")
  )
  private val sharedBlocks =
    (0 until c.sharedBlocks).map(i => new Block(s"single_blocks.$i"))
  private val imageIn = matrix("emb_in.image.weight")
  private val conditionIn = matrix("emb_in.image_cond.weight")
  private val textIn = matrix("txt_in.weight")
  private val timeIn = matrix("time_in.in_layer.weight")
  private val timeOut = matrix("time_in.out_layer.weight")
  private val targetTimeIn =
    Option.when(c.flowMap)(matrix("ref_time_in.in_layer.weight"))
  private val targetTimeOut =
    Option.when(c.flowMap)(matrix("ref_time_in.out_layer.weight"))
  private def modulations(kind: String) =
    Mode.values.toSeq
      .map(mode =>
        mode -> matrix(
          s"${kind}_stream_modulations.${mode.modulation}.lin.weight"
        )
      )
      .toMap
  private val earlyModulations = modulations("early")
  private val sharedModulations = modulations("single")
  private val finalModulation = matrix(
    "final_layer.image.adaLN_modulation.1.weight"
  )
  private val finalOut = matrix("final_layer.image.linear.weight")

  // ---- LoRAs ---------------------------------------------------------------------

  /** The active LoRAs' updates, by site: a weight's name without `.weight`, the
    * MLP's input as its `<block>.mlp_in.gate` and `.up` halves.
    */
  private val updates = new LoraUpdates(ops)

  private val allBlocks: Seq[Block] =
    Mode.values.toSeq.flatMap(earlyBlocks) ++ sharedBlocks

  /** Every site's weight (the MLP input's halves as row views). */
  private lazy val siteWeights: Map[String, Tensor] = {
    val blocks = allBlocks.flatMap { b =>
      Seq(
        "q_proj" -> b.q,
        "k_proj" -> b.k,
        "v_proj" -> b.v,
        "attn_out" -> b.attentionOut,
        "mlp_in.gate" -> b.mlpGate,
        "mlp_in.up" -> b.mlpUp,
        "mlp_out" -> b.mlpOut
      ).map((part, weight) => s"${b.prefix}.$part" -> weight)
    }
    val modes = Mode.values.toSeq.flatMap(mode =>
      Seq(
        s"early_stream_modulations.${mode.modulation}.lin" -> earlyModulations(
          mode
        ),
        s"single_stream_modulations.${mode.modulation}.lin" -> sharedModulations(
          mode
        )
      )
    )
    (blocks ++ modes ++ Seq(
      "emb_in.image" -> imageIn,
      "emb_in.image_cond" -> conditionIn,
      "txt_in" -> textIn,
      "time_in.in_layer" -> timeIn,
      "time_in.out_layer" -> timeOut,
      "final_layer.image.adaLN_modulation.1" -> finalModulation,
      "final_layer.image.linear" -> finalOut
    ) ++ targetTimeIn.map("ref_time_in.in_layer" -> _) ++
      targetTimeOut.map("ref_time_in.out_layer" -> _)).toMap
  }

  /** The weights a LoRA names, by BFL's names: the MLP's input covers its two
    * halves in turn.
    */
  private lazy val targets: Map[String, Seq[String]] =
    siteWeights.keys.toSeq
      .map(site =>
        if (site.endsWith(".mlp_in.gate") || site.endsWith(".mlp_in.up")) {
          val weight = site.substring(0, site.lastIndexOf('.'))
          weight -> Seq(s"$weight.gate", s"$weight.up")
        } else site -> Seq(site)
      )
      .toMap

  /** kohya's names (`lora_unet_single_blocks_0_q_proj`): BFL's with every dot
    * an underscore.
    */
  private lazy val kohyaNames: Map[String, String] =
    targets.keys.map(name => name.replace('.', '_') -> name).toMap

  private val Kohya = """lora_unet_(.+)""".r

  private def placement(target: String): Seq[String] = target match {
    case Kohya(name) => kohyaNames.get(name).toSeq.flatMap(targets)
    case other       => targets.getOrElse(other, Nil)
  }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read BFL's names and kohya's. The MLP input's update is split by the
    * rows of its `up`. Returns the targets that matched nothing or a weight of
    * another shape, left unapplied.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        val sites = placement(target)
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
    val bySite = placed.flatMap { (_, pair, sites) =>
      val rows = sites.map(siteWeights(_).shape.dimensions.head)
      sites.zip(rows.scanLeft(0L)(_ + _)).zip(rows).map {
        case ((site, first), count) =>
          site -> pair.copy(up = pair.up.rows(first, count))
      }
    }
    updates.use(bySite.groupMap(_._1)(_._2))
    placed.collect { case (target, _, Nil) => target }.distinct
  }

  private def lora(x: Tensor, site: String, out: Tensor): Unit =
    updates(x, site, out)

  private val attention =
    new PlainAttention(ops, c.heads, c.heads, c.headDimension)

  /** Flat buffers for `tokens` rows, grown when too small; each use takes the
    * `prefix` it needs.
    */
  final private class Buffers(val tokens: Int) extends WorkBuffers(ops) {
    private val wide = tokens * width
    private val mlp = tokens.toLong * c.intermediate
    val x: Tensor = flat(DType.F32, wide)
    val normed: Tensor = flat(DType.F32, wide)
    val q: Tensor = flat(DType.F32, wide)
    val k: Tensor = flat(DType.F32, wide)
    val v: Tensor = flat(DType.F32, wide)
    val rotatedQueries: Tensor = flat(DType.F32, wide)
    val rotatedKeys: Tensor = flat(DType.F32, wide)
    val attended: Tensor = flat(DType.F32, wide)
    val projected: Tensor = flat(DType.F32, wide)
    val mlpGate: Tensor = flat(DType.F32, mlp)
    val mlpUp: Tensor = flat(DType.F32, mlp)
  }

  private val buffers = new GrownBuffers(new Buffers(_))

  /** The RoPE tables of the last layout asked for (the target's grid, the
    * references', the text's length): the same at every step.
    */
  private var tables = Option.empty[((Grid, Seq[Grid], Int), Tensor, Tensor)]

  private def releaseTables(): Unit = tables.foreach { (_, cosines, sines) =>
    ops.release(cosines)
    ops.release(sines)
  }

  private def tablesFor(
      grid: Grid,
      references: Seq[Grid],
      textTokens: Int
  ): (Tensor, Tensor) = {
    val layout = (grid, references, textTokens)
    tables.filter(_._1 == layout).getOrElse {
      releaseTables()
      tables = None
      val (cosines, sines) = angles(grid, references, textTokens)
      val shape =
        Shape.of(cosines.length / (c.headDimension / 2), c.headDimension / 2)
      val made =
        (layout, ops.fromFloats(shape, cosines), ops.fromFloats(shape, sines))
      tables = Some(made)
      made
    } match { case (_, cosines, sines) => (cosines, sines) }
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

  /** Each token's RoPE angles, `[tokens, headDimension / 2]` cosines and sines
    * (BFL's `rope`: per axis `position × theta^(−2j / axis)`, in double), the
    * target, the references then the text.
    */
  private def angles(
      grid: Grid,
      references: Seq[Grid],
      textTokens: Int
  ): (Array[Float], Array[Float]) = {
    val pairs = c.headDimension / 2
    val perAxis = pairs / 4
    val frequencies = Array.tabulate(perAxis)(j =>
      1.0 / math.pow(Flux3Config.Theta, 2.0 * j / (2 * perAxis))
    )
    val positions = Seq.newBuilder[(Double, Double, Double, Double)]
    def image(g: Grid, l: Double) =
      (0 until g.tokens).foreach { i =>
        positions += ((
          0.0,
          Flux3Config.ImagePositionScale * (i / g.width),
          Flux3Config.ImagePositionScale * (i % g.width),
          l
        ))
      }
    image(grid, 0.0)
    references.zipWithIndex.foreach((g, k) => image(g, -1.0 - k))
    (0 until textTokens).foreach(l =>
      positions += ((0.0, 0.0, 0.0, l.toDouble))
    )
    val all = positions.result()
    val (cosines, sines) =
      (new Array[Float](all.size * pairs), new Array[Float](all.size * pairs))
    all.zipWithIndex.foreach { case ((t, y, x, l), token) =>
      Seq(t, y, x, l).zipWithIndex.foreach { (position, axis) =>
        frequencies.indices.foreach { j =>
          val angle = position * frequencies(j)
          val at = token * pairs + axis * perAxis + j
          cosines(at) = math.cos(angle).toFloat
          sines(at) = math.sin(angle).toFloat
        }
      }
    }
    (cosines, sines)
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (the target's packed
    * latents on `grid`), given the prompt's `text` (`[L, hidden]` from
    * `embedText`) and the references' packed latents on their grids, into `out`
    * (like `latents`). `next`: the level the step jumps to, which the flow map
    * (and it alone) is told.
    */
  def velocity(
      latents: Tensor,
      grid: Grid,
      references: Seq[(Tensor, Grid)],
      text: Tensor,
      timestep: Float,
      next: Option[Float],
      out: Tensor
  ): Unit = {
    val length = text.shape.dimensions.head.toInt
    val conditions = references.map(_._2.tokens).sum
    val tokens = grid.tokens + conditions + length
    require(
      latents.shape == Shape.of(grid.tokens, c.latentChannels) &&
        out.shape == latents.shape &&
        references.forall((r, g) =>
          r.shape == Shape.of(g.tokens, c.latentChannels)
        ),
      s"velocity: latents ${latents.shape}, out ${out.shape} on $grid, references ${references
          .map((r, g) => s"${r.shape} on $g")}"
    )
    require(
      next.isDefined == c.flowMap,
      if (c.flowMap) "the flow map is told the level each step jumps to"
      else "only the flow map is told the level a step jumps to"
    )
    val w = buffers(tokens)
    val small = new TensorScope(ops)
    def vector(elements: Long) =
      small.allocate(DType.F32, Shape.of(1, elements))
    try {
      // a timestep as BFL's pack holds it (BF16), × 1000 in BF16, through a
      // sinusoid (cosines first) cast to BF16, then the MLP
      val half = c.timeChannels / 2
      def embed(value: Float, first: Tensor, second: Tensor, site: String) = {
        val scaled = Bf16.round(1000f * Bf16.round(value))
        val sinusoid = TimestepSinusoid(
          scaled,
          half,
          singleFrequencies = true,
          bf16 = true
        )
        val embedded = small.fromFloats(Shape.of(1, 2L * half), sinusoid)
        val (inner, result) = (vector(width), vector(width))
        ops.linear(embedded, first, inner)
        lora(embedded, s"$site.in_layer", inner)
        ops.activation(Activation.Silu, inner, inner)
        ops.linear(inner, second, result)
        lora(inner, s"$site.out_layer", result)
        result
      }
      val target = for {
        level <- next
        first <- targetTimeIn
        second <- targetTimeOut
      } yield embed(level, first, second, "ref_time_in")
      def timeVector(value: Float) = {
        val time = embed(value, timeIn, timeOut, "time_in")
        target.foreach(ops.add(time, _, time))
        ops.activation(Activation.Silu, time, time)
        time
      }
      val imageTime = timeVector(timestep)
      // the references and the text both sit at 0
      val restTime = timeVector(0f)
      val times = Map(
        Mode.Image -> imageTime,
        Mode.Condition -> restTime,
        Mode.Text -> restTime
      )
      // (shift, scale, gate) of each mode, early and shared
      def modulate(weights: Map[Mode, Tensor], site: String) =
        Mode.values.toSeq.map { mode =>
          val modulation = vector(3 * width)
          ops.linear(times(mode), weights(mode), modulation)
          lora(
            times(mode),
            s"${site}_stream_modulations.${mode.modulation}.lin",
            modulation
          )
          mode -> modulation
        }.toMap
      val early = modulate(earlyModulations, "early")
      val shared = modulate(sharedModulations, "single")
      val finalModulated = vector(2 * width)
      ops.linear(imageTime, finalModulation, finalModulated)
      lora(imageTime, "final_layer.image.adaLN_modulation.1", finalModulated)
      def part(modulation: Tensor, i: Int) =
        modulation
          .view(modulation.shape.last / width, width)
          .rows(i, 1)
          .view(width)

      // [target ; references ; text]
      val stream = w.x.prefix(tokens, width)
      val spans = Map(
        Mode.Image -> (0, grid.tokens),
        Mode.Condition -> (grid.tokens, conditions),
        Mode.Text -> (grid.tokens + conditions, length)
      )
      ops.linear(latents, imageIn, stream.rows(0, grid.tokens))
      lora(latents, "emb_in.image", stream.rows(0, grid.tokens))
      references.foldLeft(grid.tokens) { case (at, (reference, g)) =>
        ops.linear(reference, conditionIn, stream.rows(at, g.tokens))
        lora(reference, "emb_in.image_cond", stream.rows(at, g.tokens))
        at + g.tokens
      }
      ops.copy(text, stream.rows(grid.tokens + conditions, length))
      val (cosines, sines) = tablesFor(grid, references.map(_._2), length)

      val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
      def headNorm(x: Tensor, weight: Tensor) = {
        val rows = x.shape.dimensions.head * heads
        ops.rmsNorm(x.view(rows, d), weight, epsilon, 0f, x.view(rows, d))
      }

      /** `block` on the stream's rows `[first, first + n)`, as one attention,
        * each of `segments` (`(first, n, modulation)` within them) modulated by
        * its own.
        */
      def run(
          block: Block,
          first: Int,
          n: Int,
          segments: Seq[(Int, Int, Tensor)]
      ): Unit = {
        def rows(t: Tensor) = t.prefix(tokens, width).rows(first, n)
        val (x, normed) = (rows(w.x), rows(w.normed))
        segments.foreach { (at, count, modulation) =>
          ops.layerNorm(
            x.rows(at, count),
            None,
            None,
            epsilon,
            normed.rows(at, count)
          )
          ops.modulate(
            normed.rows(at, count),
            part(modulation, 1),
            part(modulation, 0),
            normed.rows(at, count)
          )
        }
        val (q, k, v) = (rows(w.q), rows(w.k), rows(w.v))
        val (gate, up) = (
          w.mlpGate.prefix(n, c.intermediate),
          w.mlpUp.prefix(n, c.intermediate)
        )
        ops.linears(
          normed,
          Seq(block.q, block.k, block.v, block.mlpGate, block.mlpUp),
          Seq(q, k, v, gate, up)
        )
        Seq(
          "q_proj" -> q,
          "k_proj" -> k,
          "v_proj" -> v,
          "mlp_in.gate" -> gate,
          "mlp_in.up" -> up
        ).foreach((part, y) => lora(normed, s"${block.prefix}.$part", y))
        headNorm(q, block.qNorm)
        headNorm(k, block.kNorm)
        val (queries, keys) = (
          w.rotatedQueries.prefix(n, heads, d),
          w.rotatedKeys.prefix(n, heads, d)
        )
        val (cos, sin) = (cosines.rows(first, n), sines.rows(first, n))
        ops.ropeTable(q.view(n, heads, d), cos, sin, queries)
        ops.ropeTable(k.view(n, heads, d), cos, sin, keys)
        val (attended, projected) = (rows(w.attended), rows(w.projected))
        attention(queries.view(n, width), keys.view(n, width), v, attended)
        ops.linear(attended, block.attentionOut, projected)
        lora(attended, s"${block.prefix}.attn_out", projected)
        ops.gated(Activation.Silu, gate, up, gate)
        // the MLP's output into `normed`, no longer read
        ops.linear(gate, block.mlpOut, normed)
        lora(gate, s"${block.prefix}.mlp_out", normed)
        ops.add(projected, normed, projected)
        segments.foreach { (at, count, modulation) =>
          ops.gatedAdd(
            x.rows(at, count),
            projected.rows(at, count),
            part(modulation, 2)
          )
        }
      }

      (0 until c.modeBlocks).foreach { depth =>
        Mode.values.foreach { mode =>
          val (first, n) = spans(mode)
          if (n > 0)
            run(earlyBlocks(mode)(depth), first, n, Seq((0, n, early(mode))))
        }
      }
      val segments = Mode.values.toSeq.collect {
        case mode if spans(mode)._2 > 0 =>
          (spans(mode)._1, spans(mode)._2, shared(mode))
      }
      sharedBlocks.foreach(run(_, 0, tokens, segments))

      // the final layer on the target's tokens: rows (shift ; scale)
      val image = stream.rows(0, grid.tokens)
      val normed = w.normed.prefix(grid.tokens, width)
      ops.layerNorm(image, None, None, epsilon, normed)
      ops.modulate(
        normed,
        part(finalModulated, 1),
        part(finalModulated, 0),
        normed
      )
      ops.linear(normed, finalOut, out)
      lora(normed, "final_layer.image.linear", out)
    } finally small.close()
  }

  def close(): Unit = {
    updates.close()
    releaseTables()
    attention.close()
    buffers.release()
    weights.release()
    source.close()
  }
}

object Flux3 {

  def open(ops: Ops, path: Path): Flux3 = {
    val source = WeightSource.open(ops, path)
    try new Flux3(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
