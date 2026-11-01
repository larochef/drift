package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.lora.{Lora, LoraUpdates}
import drift.runner.models.Flux2.Grid
import drift.runner.ops.*
import drift.runner.tensor.*
import drift.runner.weights.WeightSource

import java.nio.file.Path
import scala.collection.mutable

/** MOSS-Image's diffusion transformer, its sizes read from the weights (5B:
  * 3072 wide, 24 heads of 128, 42 blocks, an MLP of 8192, Qwen3.5-9B's 4096
  * text features, 128 latent features). `fusedQkv`: q, k and v stored as one
  * matrix (ComfyUI's and leejet's files) rather than three (OpenMOSS's).
  */
final case class MossImageConfig(
    hidden: Int,
    heads: Int,
    headDimension: Int,
    layers: Int,
    intermediate: Int,
    textWidth: Int,
    latentChannels: Int,
    timeChannels: Int,
    fusedQkv: Boolean
)

object MossImageConfig {
  val Theta = 10000.0
  val Epsilon = 1e-6f

  /** The text refiner's blocks, before the main ones. */
  val TextBlocks = 2

  /** Whether the weights are a MOSS-Image transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("shared_block_modulation.weight") &&
      source.has("video_head_modulation.weight")

  def of(source: WeightSource): MossImageConfig = {
    if (!holds(source))
      throw new FormatException(
        "no MOSS-Image transformer (shared_block_modulation.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    val Seq(hidden, latentChannels) = dimensions("video_embedder.weight")
    val headDimension = dimensions("layers.0.q_norm.weight").head
    MossImageConfig(
      hidden = hidden,
      heads = hidden / headDimension,
      headDimension = headDimension,
      layers = Iterator
        .from(0)
        .takeWhile(i => source.has(s"layers.$i.norm1.weight"))
        .size,
      intermediate = dimensions("layers.0.proj.ffn_down.weight").last,
      textWidth = dimensions("text_embedder.proj.weight").last,
      latentChannels = latentChannels,
      timeChannels = dimensions("time_embedding.0.weight").last,
      fusedQkv = source.has("layers.0.proj.qkv.weight")
    )
  }
}

/** MOSS-Image's diffusion transformer (OpenMOSS's `MOSSImageTransformer`), from
  * its original names.
  *   - The text: Qwen3.5's last hidden state through `text_embedder.proj`, then
  *     two refiner blocks over the text alone, once per prompt (`embedText`).
  *   - Positions on four RoPE axes (theta 10000, 16 frequencies each,
  *     `rotate_half`'s halves, all sines then all cosines): the image `(0, L, y
  *     − (H − 1) / 2, x − (W − 1) / 2)` for `L` text tokens, the text `(0, l,
  *     0, 0)`. The image's tokens come first.
  *   - One time vector: a sinusoid (cosines first) of the timestep (σ × 1000)
  *     through `time_embedding`, with no activation after it, gives every
  *     block's six modulations at once (`shared_block_modulation`: shift, scale
  *     and gate for the attention, then for the MLP) and the head's two.
  *   - Every block: an RMS norm (weighted) modulated, q, k and v (RMS norms on
  *     the heads, then RoPE), attention over [image ; text], a gated residual;
  *     then the MLP's own: SiTU-GLU (`Activation.Situ`) of the fused up and
  *     gate. No biases in the blocks. `skipLayers` leaves blocks out: the
  *     negative branch runs without block 22.
  *   - The head on the image's tokens: RMS norm, modulated, a linear.
  */
final class MossImage private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: MossImageConfig = MossImageConfig.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = MossImageConfig.Epsilon
  private val width = c.hidden.toLong

  private def matrix(name: String): Tensor = source(name)

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  /** A linear layer with a bias, by its original name (its LoRA site). */
  private def affineLayer(prefix: String, outputs: Long): Affine =
    Affine(matrix(s"$prefix.weight"), floats(s"$prefix.bias", outputs), prefix)

  /** The `count` equal row blocks of a fused matrix, each by its own site. */
  private def split(
      weight: Tensor,
      sites: Seq[String]
  ): Seq[(String, Tensor)] = {
    val rows = weight.shape.dimensions.head / sites.size
    sites.zipWithIndex.map((site, i) => site -> weight.rows(i * rows, rows))
  }

  /** A text refiner block: RMS norms, a fused qkv with a bias, a projection and
    * the MLP's gate, up and down, all with biases.
    */
  final private class TextBlock(val prefix: String) {
    val norm1: Tensor = floats(s"$prefix.norm1.weight", width)
    val norm2: Tensor = floats(s"$prefix.norm2.weight", width)
    val qNorm: Tensor = floats(s"$prefix.q_norm.weight", c.headDimension)
    val kNorm: Tensor = floats(s"$prefix.k_norm.weight", c.headDimension)
    private val qkvBias = floats(s"$prefix.qkv.bias", 3 * width)
    val Seq(q, k, v) = split(
      matrix(s"$prefix.qkv.weight"),
      Seq("q", "k", "v").map(part => s"$prefix.qkv.$part")
    ).zipWithIndex.map { case ((site, weight), i) =>
      Affine(weight, qkvBias.view(3, width).rows(i, 1).view(width), site)
    }
    val proj: Affine = affineLayer(s"$prefix.proj", width)
    val gate: Affine = affineLayer(s"$prefix.ffn_gate", c.intermediate)
    val up: Affine = affineLayer(s"$prefix.ffn_up", c.intermediate)
    val down: Affine = affineLayer(s"$prefix.ffn_down", width)
  }

  /** A main block: its weights by LoRA site (`<prefix>.proj.q` …, the MLP's
    * fused input as its `.up` and `.gate` halves).
    */
  final private class Block(val prefix: String) {
    val norm1: Tensor = floats(s"$prefix.norm1.weight", width)
    val norm2: Tensor = floats(s"$prefix.norm2.weight", width)
    val qNorm: Tensor = floats(s"$prefix.q_norm.weight", c.headDimension)
    val kNorm: Tensor = floats(s"$prefix.k_norm.weight", c.headDimension)
    private val p = s"$prefix.proj"
    val Seq(q, k, v) =
      if (c.fusedQkv)
        split(matrix(s"$p.qkv.weight"), Seq(s"$p.q", s"$p.k", s"$p.v"))
      else
        Seq("q", "k", "v").map(part =>
          s"$p.$part" -> matrix(s"$p.$part.weight")
        )
    val o: (String, Tensor) = s"$p.o" -> matrix(s"$p.o.weight")
    val Seq(up, gate) = split(
      matrix(s"$p.ffn_up_gate.weight"),
      Seq(s"$p.ffn_up_gate.up", s"$p.ffn_up_gate.gate")
    )
    val down: (String, Tensor) = s"$p.ffn_down" -> matrix(s"$p.ffn_down.weight")
    def sites: Seq[(String, Tensor)] = Seq(q, k, v, o, up, gate, down)
  }

  private val textIn = affineLayer("text_embedder.proj", width)
  private val textBlocks = (0 until MossImageConfig.TextBlocks).map(i =>
    new TextBlock(s"text_embedder.blocks.$i")
  )
  private val blocks = (0 until c.layers).map(i => new Block(s"layers.$i"))
  private val imageIn = affineLayer("video_embedder", width)
  private val timeIn = affineLayer("time_embedding.0", width)
  private val timeOut = affineLayer("time_embedding.2", width)
  private val blockModulation =
    affineLayer("shared_block_modulation", 6 * width)
  private val headModulation = affineLayer("video_head_modulation", 2 * width)
  private val finalNorm = floats("final_norm_video.weight", width)
  private val finalOut = matrix("final_linear_video.weight")

  // ---- LoRAs ---------------------------------------------------------------------

  /** The active LoRAs' updates, by site. */
  private val updates = new LoraUpdates(ops)

  /** Every site's weight (fused matrices as row views). */
  private lazy val siteWeights: Map[String, Tensor] = {
    val affines = Seq(
      textIn,
      imageIn,
      timeIn,
      timeOut,
      blockModulation,
      headModulation
    ) ++ textBlocks.flatMap(b =>
      Seq(b.q, b.k, b.v, b.proj, b.gate, b.up, b.down)
    )
    (blocks.flatMap(_.sites) ++ affines.map(a => a.site -> a.weight) :+
      ("final_linear_video" -> finalOut)).toMap
  }

  /** The weights a LoRA names, by OpenMOSS's names: a fused one (`proj.qkv`,
    * `proj.ffn_up_gate`, a text block's `qkv`) covers its parts in the order
    * their rows take.
    */
  private lazy val targets: Map[String, Seq[String]] = {
    val fused = blocks.flatMap { b =>
      Seq(
        s"${b.prefix}.proj.qkv" -> Seq(b.q._1, b.k._1, b.v._1),
        s"${b.prefix}.proj.ffn_up_gate" -> Seq(b.up._1, b.gate._1)
      )
    } ++ textBlocks.map(b =>
      s"${b.prefix}.qkv" -> Seq(b.q.site, b.k.site, b.v.site)
    )
    siteWeights.keys.map(site => site -> Seq(site)).toMap ++ fused
  }

  /** kohya's names (`lora_unet_layers_0_proj_q`): OpenMOSS's with every dot an
    * underscore.
    */
  private lazy val kohyaNames: Map[String, String] =
    targets.keys.map(name => name.replace('.', '_') -> name).toMap

  private val Kohya = """lora_unet_(.+)""".r

  private def placement(target: String): Seq[String] = target match {
    case Kohya(name) => kohyaNames.get(name).toSeq.flatMap(targets)
    case other       => targets.getOrElse(other, Nil)
  }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read OpenMOSS's names and kohya's. A fused weight's update is split
    * by the rows of its `up`. Returns the targets that matched nothing or a
    * weight of another shape, left unapplied.
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

  private def affine(layer: Affine, x: Tensor, out: Tensor): Unit =
    layer(ops, x, out, updates)

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

  /** The RoPE tables of the layouts last asked for (the image's grid, the
    * text's length), the same at every step: a prompt's and its negative's.
    */
  private val tables =
    mutable.LinkedHashMap.empty[(Grid, Int), (Tensor, Tensor)]
  private val KeptTables = 4

  private def releaseTables(): Unit = {
    tables.values.foreach { (cosines, sines) =>
      ops.release(cosines)
      ops.release(sines)
    }
    tables.clear()
  }

  /** The tables of `textTokens` text tokens after `grid`'s image (an empty grid
    * for the text alone).
    */
  private def tablesFor(grid: Grid, textTokens: Int): (Tensor, Tensor) =
    tables.getOrElse(
      (grid, textTokens), {
        if (tables.size >= KeptTables) {
          val (oldest, (cosines, sines)) = tables.head
          ops.release(cosines)
          ops.release(sines)
          tables.remove(oldest)
        }
        val (cosines, sines) = MossImage.angles(c, grid, textTokens)
        val shape =
          Shape.of(cosines.length / (c.headDimension / 2), c.headDimension / 2)
        val made =
          (ops.fromFloats(shape, cosines), ops.fromFloats(shape, sines))
        tables((grid, textTokens)) = made
        made
      }
    )

  private val (heads, d) = (c.heads.toLong, c.headDimension.toLong)

  private def headNorm(x: Tensor, weight: Tensor): Unit = {
    val rows = x.shape.dimensions.head * heads
    ops.rmsNorm(x.view(rows, d), weight, epsilon, 0f, x.view(rows, d))
  }

  /** q and k (`[n, hidden]`) normed per head and turned by `cosines` and
    * `sines`, then attention over the `n` tokens into `attended`.
    */
  private def attend(
      w: Buffers,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      qNorm: Tensor,
      kNorm: Tensor,
      cosines: Tensor,
      sines: Tensor,
      attended: Tensor
  ): Unit = {
    val n = q.shape.dimensions.head
    headNorm(q, qNorm)
    headNorm(k, kNorm)
    val (queries, keys) = (
      w.rotatedQueries.prefix(n, heads, d),
      w.rotatedKeys.prefix(n, heads, d)
    )
    ops.ropeTable(q.view(n, heads, d), cosines, sines, queries, halves = true)
    ops.ropeTable(k.view(n, heads, d), cosines, sines, keys, halves = true)
    attention(queries.view(n, width), keys.view(n, width), v, attended)
  }

  /** The prompt's text for every step: Qwen3.5's last hidden state (`[L,
    * textWidth]`) through the projection and the refiner blocks into `out`
    * (`[L, hidden]`).
    */
  def embedText(features: Tensor, out: Tensor): Unit = {
    val n = features.shape.dimensions.head
    require(
      features.shape == Shape.of(n, c.textWidth) &&
        out.shape == Shape.of(n, width),
      s"embedText: features ${features.shape}, out ${out.shape}"
    )
    val w = buffers(n.toInt)
    val (cosines, sines) = tablesFor(Grid(0, 0), n.toInt)
    def rows(t: Tensor) = t.prefix(n, width)
    affine(textIn, features, out)
    textBlocks.foreach { b =>
      val (normed, q, k, v) = (rows(w.normed), rows(w.q), rows(w.k), rows(w.v))
      ops.rmsNorm(out, b.norm1, epsilon, 0f, normed)
      affine(b.q, normed, q)
      affine(b.k, normed, k)
      affine(b.v, normed, v)
      val (attended, projected) = (rows(w.attended), rows(w.projected))
      attend(w, q, k, v, b.qNorm, b.kNorm, cosines, sines, attended)
      affine(b.proj, attended, projected)
      ops.add(out, projected, out)
      ops.rmsNorm(out, b.norm2, epsilon, 0f, normed)
      val (gate, up) = (
        w.mlpGate.prefix(n, c.intermediate),
        w.mlpUp.prefix(n, c.intermediate)
      )
      affine(b.gate, normed, gate)
      affine(b.up, normed, up)
      ops.gated(Activation.Situ, gate, up, gate)
      affine(b.down, gate, projected)
      ops.add(out, projected, out)
    }
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (the image's latents
    * on `grid`, `[tokens, latentChannels]` row by row), given the prompt's
    * `text` (`[L, hidden]` from `embedText`), into `out` (like `latents`), the
    * blocks of `skipLayers` (from 0) left out.
    */
  def velocity(
      latents: Tensor,
      grid: Grid,
      text: Tensor,
      timestep: Float,
      skipLayers: Set[Int],
      out: Tensor
  ): Unit = {
    val length = text.shape.dimensions.head.toInt
    val tokens = grid.tokens + length
    require(
      latents.shape == Shape.of(grid.tokens, c.latentChannels) &&
        out.shape == latents.shape,
      s"velocity: latents ${latents.shape}, out ${out.shape} on $grid"
    )
    val w = buffers(tokens)
    val small = new TensorScope(ops)
    def vector(elements: Long) =
      small.allocate(DType.F32, Shape.of(1, elements))
    try {
      // the scheduler's timestep (σ × 1000 in F32) through a sinusoid in
      // double, cosines first, then the MLP
      val sinusoid =
        TimestepSinusoid((timestep * 1000f).toDouble, c.timeChannels / 2)
      val embedded =
        small.fromFloats(Shape.of(1, c.timeChannels.toLong), sinusoid)
      val (inner, time) = (vector(width), vector(width))
      affine(timeIn, embedded, inner)
      ops.activation(Activation.Silu, inner, inner)
      affine(timeOut, inner, time)
      val modulation = vector(6 * width)
      affine(blockModulation, time, modulation)
      val head = vector(2 * width)
      affine(headModulation, time, head)
      def part(table: Tensor, i: Int) =
        table.view(table.shape.last / width, width).rows(i, 1).view(width)

      // [image ; text]
      val stream = w.x.prefix(tokens, width)
      affine(imageIn, latents, stream.rows(0, grid.tokens))
      ops.copy(text, stream.rows(grid.tokens, length))
      val (cosines, sines) = tablesFor(grid, length)
      def rows(t: Tensor) = t.prefix(tokens, width)
      val (normed, q, k, v) = (rows(w.normed), rows(w.q), rows(w.k), rows(w.v))
      val (attended, projected) = (rows(w.attended), rows(w.projected))
      val (gate, up) = (
        w.mlpGate.prefix(tokens, c.intermediate),
        w.mlpUp.prefix(tokens, c.intermediate)
      )
      def linear(x: Tensor, site: (String, Tensor), into: Tensor) = {
        ops.linear(x, site._2, into)
        lora(x, site._1, into)
      }
      blocks.zipWithIndex.foreach { (b, index) =>
        if (!skipLayers(index)) {
          ops.rmsNorm(stream, b.norm1, epsilon, 0f, normed)
          ops.modulate(normed, part(modulation, 1), part(modulation, 0), normed)
          ops.linears(normed, Seq(b.q._2, b.k._2, b.v._2), Seq(q, k, v))
          Seq(b.q -> q, b.k -> k, b.v -> v).foreach((site, y) =>
            lora(normed, site._1, y)
          )
          attend(w, q, k, v, b.qNorm, b.kNorm, cosines, sines, attended)
          linear(attended, b.o, projected)
          ops.gatedAdd(stream, projected, part(modulation, 2))
          ops.rmsNorm(stream, b.norm2, epsilon, 0f, normed)
          ops.modulate(normed, part(modulation, 4), part(modulation, 3), normed)
          ops.linears(normed, Seq(b.up._2, b.gate._2), Seq(up, gate))
          lora(normed, b.up._1, up)
          lora(normed, b.gate._1, gate)
          ops.gated(Activation.Situ, gate, up, gate)
          linear(gate, b.down, projected)
          ops.gatedAdd(stream, projected, part(modulation, 5))
        }
      }

      // the head on the image's tokens: (shift ; scale)
      val image = stream.rows(0, grid.tokens)
      val imageNormed = w.normed.prefix(grid.tokens, width)
      ops.rmsNorm(image, finalNorm, epsilon, 0f, imageNormed)
      ops.modulate(imageNormed, part(head, 1), part(head, 0), imageNormed)
      ops.linear(imageNormed, finalOut, out)
      lora(imageNormed, "final_linear_video", out)
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

object MossImage {

  def open(ops: Ops, path: Path): MossImage = {
    val source = WeightSource.open(ops, path)
    try new MossImage(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }

  /** Each token's RoPE angles, `[tokens, headDimension / 2]` cosines and sines
    * (OpenMOSS's `_RoPE`, in F32 as it computes them): frequency `j` turns axis
    * `j / bands` by `theta^(−(j % bands) / bands)`, `bands` an eighth of a
    * head; `grid`'s image first, then `textTokens` of text.
    */
  def angles(
      c: MossImageConfig,
      grid: Grid,
      textTokens: Int
  ): (Array[Float], Array[Float]) = {
    val pairs = c.headDimension / 2
    val bands = c.headDimension / 8
    val frequencies = Array.tabulate(bands)(j =>
      (1.0 / math
        .pow(MossImageConfig.Theta, (j.toFloat / bands).toDouble)).toFloat
    )
    val (rowCentre, columnCentre) =
      ((grid.height - 1) / 2.0, (grid.width - 1) / 2.0)
    val positions =
      (0 until grid.tokens).map(i =>
        Seq(
          0f,
          textTokens.toFloat,
          (i / grid.width - rowCentre).toFloat,
          (i % grid.width - columnCentre).toFloat
        )
      ) ++ (0 until textTokens).map(l => Seq(0f, l.toFloat, 0f, 0f))
    val (cosines, sines) = (
      new Array[Float](positions.size * pairs),
      new Array[Float](positions.size * pairs)
    )
    positions.zipWithIndex.foreach { (position, token) =>
      (0 until pairs).foreach { j =>
        val angle = position(j / bands) * frequencies(j % bands)
        cosines(token * pairs + j) = math.cos(angle).toFloat
        sines(token * pairs + j) = math.sin(angle).toFloat
      }
    }
    (cosines, sines)
  }
}
