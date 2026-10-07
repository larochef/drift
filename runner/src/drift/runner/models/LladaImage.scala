package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.models.Flux2.Grid
import drift.runner.ops.*
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** LLaDA-Image's transformer, read off its weights (released: 3840 wide, 30
  * heads of 128, 30 blocks after 2 refiner blocks of each kind, MLPs of 10240,
  * 128 latent features, caption features of 2560, semantic features of 4096).
  */
final case class LladaImageConfig(
    hidden: Int,
    heads: Int,
    headDimension: Int,
    layers: Int,
    refiners: Int,
    intermediate: Int,
    latentChannels: Int,
    captionWidth: Int,
    semanticWidth: Int,
    timeWidth: Int,
    epsilon: Float
) {

  /** RoPE's three axes (sequence, row, column) in pairs: 32/48/48 values for
    * heads of 128.
    */
  def ropeAxes: RopeSections.Axes = {
    val sequence = headDimension / 8
    val side = (headDimension / 2 - sequence) / 2
    RopeSections.Axes(Seq(sequence, side, side))
  }
}

object LladaImageConfig {
  val Theta = 256f

  /** Sequences are padded to a multiple of this many tokens. */
  val SequenceMultiple = 32

  /** Whether the weights are a LLaDA-Image transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("all_x_embedder.1-1.weight") &&
      source.has("sigvq_embedder.1.weight")

  def of(source: WeightSource): LladaImageConfig = {
    if (!holds(source))
      throw new FormatException(
        "no LLaDA-Image transformer (all_x_embedder.1-1.weight, sigvq_embedder.1.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    def count(prefix: String) =
      Iterator
        .from(0)
        .takeWhile(i => source.has(s"$prefix.$i.attention.to_q.weight"))
        .size
    val Seq(hidden, latentChannels) = dimensions("all_x_embedder.1-1.weight")
    val configured = source.config
    val heads = configured
      .filter(_.has("n_heads"))
      .fold(hidden / 128)(_.int("n_heads"))
    LladaImageConfig(
      hidden = hidden,
      heads = heads,
      headDimension = hidden / heads,
      layers = count("layers"),
      refiners = count("noise_refiner"),
      intermediate = dimensions("layers.0.feed_forward.w1.weight").head,
      latentChannels = latentChannels,
      captionWidth = dimensions("cap_embedder.1.weight").last,
      semanticWidth = dimensions("sigvq_embedder.1.weight").last,
      timeWidth = dimensions("t_embedder.mlp.0.weight").last,
      epsilon = configured
        .filter(_.has("norm_eps"))
        .fold(1e-5f)(_.double("norm_eps").toFloat)
    )
  }
}

/** A prompt (and, editing, the source image's semantic tokens) as the main
  * blocks take them, which no step changes: the refined caption tokens
  * (`caption`, two copies of them when editing) and semantic tokens, each
  * padded to whole multiples of 32, with their positions (`[3, tokens]`,
  * axis-major).
  */
final class LladaCondition private[models] (
    ops: Ops,
    val caption: Tensor,
    val captionPositions: Array[Int],
    /** The caption tokens of one copy, padding included. */
    val captionLength: Int,
    /** The caption tokens supplied. */
    val captionSupplied: Int,
    val semantic: Option[(Tensor, Array[Int])],
    /** Whether the sequence holds a source image (two caption copies). */
    val editing: Boolean
) extends AutoCloseable {
  def close(): Unit = (caption +: semantic.map(_._1).toSeq).foreach(ops.release)
}

/** LLaDA-Image's diffusion transformer (inclusionAI's
  * `LLaDAImageTransformer2DModel`, a Lumina 2 / Z-Image NextDiT; sd-cpp's
  * `llada_image.hpp`), from the released names.
  *   - A block: RMS norm (no weights), attention (no biases, RMS norms on q and
  *     k per head, RoPE on three axes, theta 256), an RMS norm of what the
  *     attention gives, a residual; the same around a SwiGLU MLP (`w2(silu(w1
  *     x) × w3 x)`). The modulated blocks scale each norm's output and gate
  *     each residual (`tanh`) from the timestep; the refiners of the conditions
  *     are not modulated.
  *   - Text to image: the caption features through an RMS norm, a linear and
  *     the context refiner; the latents (one token each) through a linear and
  *     the noise refiner; then [image ; caption] through the main blocks, and a
  *     final layer on the image's tokens. Every sequence is padded to a
  *     multiple of 32 tokens with a learned pad token. Positions: caption token
  *     `i` at `(1 + i, 0, 0)`, the image at `(1 + caption length, row,
  *     column)`, pads at the origin.
  *   - Editing: the caption twice (the second copy two positions after the
  *     first), the source's latents then the target's, each image right after
  *     its caption copy on the sequence axis, and the source's semantic tokens
  *     (SigVQ's) through their own refiner; [captions ; images ; semantic]
  *     through the main blocks. The first caption copy, the source and the
  *     semantic tokens are modulated as at timestep 0 (clean), the second copy
  *     and the target as at the step's.
  *   - The timestep: a sinusoid of `1000 σ` (cosines first) through an MLP; the
  *     released pipeline runs in BF16, and σ, its product and the sinusoid are
  *     rounded as it rounds them. The model gives the velocity's opposite,
  *     which `velocity` turns back.
  */
final class LladaImage private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: LladaImageConfig = LladaImageConfig.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = false)
  private val width = c.hidden.toLong
  private val Multiple = LladaImageConfig.SequenceMultiple

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))
  private def matrix(name: String): Tensor = source(s"$name.weight")

  final private case class Affine(weight: Tensor, bias: Tensor)
  private def affineLayer(name: String): Affine = {
    val weight = matrix(name)
    Affine(weight, floats(s"$name.bias", weight.shape.dimensions.head))
  }
  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  final private class Block(prefix: String, modulated: Boolean) {
    val query: Tensor = matrix(s"$prefix.attention.to_q")
    val key: Tensor = matrix(s"$prefix.attention.to_k")
    val value: Tensor = matrix(s"$prefix.attention.to_v")
    val output: Tensor = matrix(s"$prefix.attention.to_out.0")
    val gate: Tensor = matrix(s"$prefix.feed_forward.w1")
    val down: Tensor = matrix(s"$prefix.feed_forward.w2")
    val up: Tensor = matrix(s"$prefix.feed_forward.w3")
    val modulation: Option[Affine] =
      Option.when(modulated)(affineLayer(s"$prefix.adaLN_modulation.0"))
  }
  private def blocks(prefix: String, count: Int, modulated: Boolean) =
    (0 until count).map(i => new Block(s"$prefix.$i", modulated))

  private val noiseRefiner = blocks("noise_refiner", c.refiners, true)
  private val contextRefiner = blocks("context_refiner", c.refiners, false)
  private val semanticRefiner = blocks("sigvq_refiner", c.refiners, false)
  private val layers = blocks("layers", c.layers, true)
  private val modulatedBlocks = noiseRefiner ++ layers
  private val imageIn = affineLayer("all_x_embedder.1-1")
  private val captionIn = affineLayer("cap_embedder.1")
  private val semanticIn = affineLayer("sigvq_embedder.1")
  private val timeIn = affineLayer("t_embedder.mlp.0")
  private val timeOut = affineLayer("t_embedder.mlp.2")
  private val finalModulation = affineLayer(
    "all_final_layer.1-1.adaLN_modulation.1"
  )
  private val finalOut = affineLayer("all_final_layer.1-1.linear")
  private val imagePad = floats("x_pad_token", width)
  private val captionPad = floats("cap_pad_token", width)
  private val semanticPad = floats("sigvq_pad_token", width)

  private def constant(count: Long, value: Float) =
    weights.keep(
      ops.fromFloats(Shape.of(count), Array.fill(count.toInt)(value))
    )
  private val ones = constant(width, 1f)
  private val headOnes = constant(c.headDimension, 1f)
  private val zero = constant(width, 0f)
  private val widestOnes =
    constant(math.max(c.captionWidth, c.semanticWidth), 1f)

  private val rope = Rope(
    LladaImageConfig.Theta,
    c.headDimension,
    RopeLayout.Interleaved,
    c.ropeAxes
  )
  private val attention =
    new PlainAttention(ops, c.heads, c.heads, c.headDimension)

  private def padded(count: Int) = (count + Multiple - 1) / Multiple * Multiple

  /** Buffers for sequences of up to `tokens`. */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def rows(dtype: DType, count: Long, columns: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(count, columns))
      held += tensor
      tensor
    }
    val x: Tensor = rows(DType.F32, tokens, width)
    val normed: Tensor = rows(DType.F32, tokens, width)
    val q: Tensor = rows(DType.F32, tokens, width)
    val k: Tensor = rows(DType.F32, tokens, width)
    val v: Tensor = rows(DType.F32, tokens, width)
    val rotatedQueries: Tensor = rows(DType.F32, tokens, width)
    val rotatedKeys: Tensor = rows(DType.F32, tokens, width)
    val attended: Tensor = rows(DType.F32, tokens, width)
    val projected: Tensor = rows(DType.F32, tokens, width)
    val mlpGate: Tensor = rows(DType.F32, tokens, c.intermediate)
    val mlpUp: Tensor = rows(DType.F32, tokens, c.intermediate)
    val positions: Tensor = rows(DType.I32, 3, tokens)
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

  /** A modulation of the rows `[first, first + count)`: four vectors (scale,
    * gate, scale, gate), the gates through `tanh` already.
    */
  final private case class Span(first: Int, count: Int, parts: Tensor)

  /** One block over the first `tokens` rows of `w.x` at `positions`. */
  private def run(
      block: Block,
      w: Buffers,
      tokens: Int,
      positions: Tensor,
      spans: Seq[Span]
  ): Unit = {
    def first(t: Tensor, columns: Long = width) =
      t.view(t.shape.elementCount).prefix(tokens, columns)
    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    val (x, normed, projected) =
      (first(w.x), first(w.normed), first(w.projected))
    def scaled(part: Int): Unit = {
      ops.rmsNorm(x, ones, c.epsilon, 0f, normed)
      spans.foreach(s =>
        ops.modulate(
          normed.rows(s.first, s.count),
          s.parts.rows(part, 1).view(width),
          zero,
          normed.rows(s.first, s.count)
        )
      )
    }
    def residual(part: Int): Unit = {
      ops.rmsNorm(projected, ones, c.epsilon, 0f, projected)
      if (block.modulation.isEmpty) ops.add(x, projected, x)
      else
        spans.foreach(s =>
          ops.gatedAdd(
            x.rows(s.first, s.count),
            projected.rows(s.first, s.count),
            s.parts.rows(part, 1).view(width)
          )
        )
    }
    scaled(0)
    val (q, k, v) = (first(w.q), first(w.k), first(w.v))
    ops.linears(normed, Seq(block.query, block.key, block.value), Seq(q, k, v))
    Seq(q, k).foreach(t =>
      ops.rmsNorm(
        t.view(tokens * heads, d),
        headOnes,
        c.epsilon,
        0f,
        t.view(tokens * heads, d)
      )
    )
    val (rotatedQueries, rotatedKeys) =
      (first(w.rotatedQueries), first(w.rotatedKeys))
    ops.rope(
      q.view(tokens, heads, d),
      positions,
      rope,
      rotatedQueries.view(tokens, heads, d)
    )
    ops.rope(
      k.view(tokens, heads, d),
      positions,
      rope,
      rotatedKeys.view(tokens, heads, d)
    )
    attention(rotatedQueries, rotatedKeys, v, first(w.attended))
    ops.linear(first(w.attended), block.output, projected)
    residual(1)
    scaled(2)
    val (gate, up) =
      (first(w.mlpGate, c.intermediate), first(w.mlpUp, c.intermediate))
    ops.linears(normed, Seq(block.gate, block.up), Seq(gate, up))
    ops.gated(Activation.Silu, gate, up, gate)
    ops.linear(gate, block.down, projected)
    residual(3)
  }

  /** `features` through an RMS norm and `layer` into `out`'s first rows, its
    * other rows the pad token.
    */
  private def embedded(
      features: Tensor,
      layer: Affine,
      pad: Tensor,
      out: Tensor
  ): Unit = {
    val supplied = features.shape.dimensions.head
    val total = out.shape.dimensions.head
    val normed = ops.allocate(DType.F32, features.shape)
    try {
      ops.rmsNorm(
        features,
        widestOnes.rows(0, features.shape.last),
        c.epsilon,
        0f,
        normed
      )
      affine(normed, layer, out.rows(0, supplied))
      (supplied until total).foreach(i =>
        ops.copy(pad, out.rows(i, 1).view(width))
      )
    } finally ops.release(normed)
  }

  /** Positions `[3, tokens]`, axis-major, as a tensor of the buffers'. */
  private def place(w: Buffers, positions: Array[Int], tokens: Int): Tensor = {
    val out = w.positions.view(3L * w.tokens).prefix(3, tokens)
    ops.writeInts(out.view(3L * tokens), positions)
    out
  }

  /** `count` tokens along the sequence axis from `start`, then pads at the
    * origin up to `total`: positions `[3, total]`, axis-major.
    */
  private def along(start: Int, count: Int, total: Int): Array[Int] =
    Array.tabulate(3 * total)(i => if (i < count) start + i else 0)

  /** An image's positions on `grid` at `frame`, padded to `total`. */
  private def onGrid(frame: Int, grid: Grid, total: Int): Array[Int] =
    Array.tabulate(3 * total) { i =>
      val (axis, token) = (i / total, i % total)
      if (token >= grid.tokens) 0
      else if (axis == 0) frame
      else if (axis == 1) token / grid.width
      else token % grid.width
    }

  /** The prompt's caption features (`[n, captionWidth]`) for every step; with
    * `editing`, for an image of `imageTokens` latents beside its source, and
    * with `semantic` (`[m, semanticWidth]`, the source's SigVQ features; the
    * unguided side of an edit has none).
    */
  def condition(
      caption: Tensor,
      semantic: Option[Tensor],
      imageTokens: Int,
      editing: Boolean
  ): LladaCondition = {
    val supplied = caption.shape.dimensions.head.toInt
    val length = padded(supplied)
    require(editing || semantic.isEmpty, "semantic tokens are an edit's")
    val copies = if (editing) 2 else 1
    val w = buffersFor(
      math.max(
        copies * length,
        semantic.fold(0)(s => padded(s.shape.dimensions.head.toInt))
      )
    )
    // the copies' positions: the second starts two after the first's end
    val starts = Seq(1, 1 + supplied + 2).take(copies)
    val positions = {
      val each = starts.map(along(_, supplied, length))
      Array.tabulate(3 * copies * length) { i =>
        val (axis, token) = (i / (copies * length), i % (copies * length))
        each(token / length)(axis * length + token % length)
      }
    }
    def refined(
        tokens: Int,
        places: Array[Int],
        refiner: Seq[Block]
    ): Tensor = {
      val at = place(w, places, tokens)
      refiner.foreach(run(_, w, tokens, at, Nil))
      val out = ops.allocate(DType.F32, Shape.of(tokens, width))
      ops.copy(w.x.view(w.tokens * width).prefix(tokens, width), out)
      out
    }
    val stream = w.x.view(w.tokens * width).prefix(copies * length, width)
    embedded(caption, captionIn, captionPad, stream.rows(0, length))
    if (copies == 2)
      ops.copy(stream.rows(0, length), stream.rows(length, length))
    val captions = refined(copies * length, positions, contextRefiner)
    val semantics = semantic.map { features =>
      val count = features.shape.dimensions.head.toInt
      val total = padded(count)
      val places =
        along(copies * length + 2 * padded(imageTokens) + 1, count, total)
      embedded(
        features,
        semanticIn,
        semanticPad,
        w.x.view(w.tokens * width).prefix(total, width)
      )
      (refined(total, places, semanticRefiner), places)
    }
    new LladaCondition(
      ops,
      captions,
      positions,
      length,
      supplied,
      semantics,
      editing
    )
  }

  private val small = mutable.ArrayBuffer.empty[Tensor]
  private def vector(rows: Long, elements: Long): Tensor = {
    val tensor = ops.allocate(DType.F32, Shape.of(rows, elements))
    small += tensor
    tensor
  }
  private val timeInner = vector(1, timeIn.bias.shape.elementCount)
  private val time = vector(1, c.timeWidth)
  private val timeSilu = vector(1, c.timeWidth)
  private val raw = vector(modulatedBlocks.size, 4 * width)
  private val finalScale = vector(1, width)

  /** Every modulated block's four vectors at `timestep` (`[4 × blocks,
    * hidden]`, the gates through `tanh`) and the final layer's scale.
    */
  private def modulations(timestep: Float): (Tensor, Tensor) = {
    val sinusoid = ops.fromFloats(
      Shape.of(1, c.timeWidth),
      LladaImage.sinusoid(timestep, c.timeWidth)
    )
    try affine(sinusoid, timeIn, timeInner)
    finally ops.release(sinusoid)
    ops.activation(Activation.Silu, timeInner, timeInner)
    affine(timeInner, timeOut, time)
    modulatedBlocks.zipWithIndex.foreach((block, i) =>
      affine(time, block.modulation.get, raw.rows(i, 1))
    )
    ops.activation(Activation.Silu, time, timeSilu)
    affine(timeSilu, finalModulation, finalScale)
    val parts = ops.toFloats(raw)
    parts.indices.foreach { i =>
      if (i / c.hidden % 2 == 1) parts(i) = math.tanh(parts(i)).toFloat
    }
    val last = ops.allocate(DType.F32, Shape.of(width))
    ops.copy(finalScale.view(width), last)
    (ops.fromFloats(Shape.of(4L * modulatedBlocks.size, width), parts), last)
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (`[grid.tokens,
    * latentChannels]`, row-major) under `condition`, into `out` (like
    * `latents`); editing, `source` is the source image's latents on the same
    * grid.
    */
  def velocity(
      latents: Tensor,
      grid: Grid,
      condition: LladaCondition,
      source: Option[Tensor],
      timestep: Float,
      out: Tensor
  ): Unit = {
    require(
      latents.shape == Shape.of(grid.tokens, c.latentChannels) &&
        out.shape == latents.shape && source.forall(_.shape == latents.shape) &&
        source.isDefined == condition.editing,
      s"velocity: latents ${latents.shape}, out ${out.shape} on $grid, source ${source
          .map(_.shape)}, editing ${condition.editing}"
    )
    val each = padded(grid.tokens)
    val images = if (condition.editing) 2 else 1
    val captions = condition.caption.shape.dimensions.head.toInt
    val semantics =
      condition.semantic.fold(0)(_._1.shape.dimensions.head.toInt)
    val tokens = images * each + captions + semantics
    val w = buffersFor(tokens)
    val made = mutable.ArrayBuffer.empty[Tensor]
    try {
      val (noisy, noisyFinal) = modulations(timestep)
      made ++= Seq(noisy, noisyFinal)
      val clean = Option.when(condition.editing) {
        val both = modulations(0f)
        made ++= Seq(both._1, both._2)
        both
      }
      def parts(of: Tensor, block: Int) = of.rows(4L * block, 4)
      // a block's modulation over spans of (rows, noisy or clean)
      def spans(block: Int, layout: Seq[(Int, Int, Boolean)]) =
        layout.map((first, count, isNoisy) =>
          Span(
            first,
            count,
            parts(if (isNoisy) noisy else clean.get._1, block)
          )
        )

      // the images: [source ; target] editing, each padded
      val stream = w.x.view(w.tokens * width).prefix(tokens, width)
      val supplied = source.toSeq :+ latents
      val frames =
        if (condition.editing)
          Seq(
            1 + condition.captionSupplied,
            1 + condition.captionSupplied + 2 + condition.captionSupplied
          )
        else Seq(1 + condition.captionLength)
      supplied.zipWithIndex.foreach { (image, i) =>
        affine(image, imageIn, stream.rows(i.toLong * each, grid.tokens))
        (grid.tokens until each).foreach(p =>
          ops.copy(imagePad, stream.rows(i.toLong * each + p, 1).view(width))
        )
      }
      val imagePositions = {
        val perImage = frames.map(onGrid(_, grid, each))
        Array.tabulate(3 * images * each) { i =>
          val (axis, token) = (i / (images * each), i % (images * each))
          perImage(token / each)(axis * each + token % each)
        }
      }
      val imageLayout =
        if (condition.editing) Seq((0, each, false), (each, each, true))
        else Seq((0, each, true))
      val imagesAt = place(w, imagePositions, images * each)
      noiseRefiner.indices.foreach(i =>
        run(noiseRefiner(i), w, images * each, imagesAt, spans(i, imageLayout))
      )

      // the whole sequence: [image ; caption], or editing [captions ; images ;
      // semantic]
      def joined(groups: Seq[Array[Int]], counts: Seq[Int]): Array[Int] = {
        val total = counts.sum
        val starts = counts.scanLeft(0)(_ + _)
        Array.tabulate(3 * total) { i =>
          val (axis, token) = (i / total, i % total)
          val g = starts.lastIndexWhere(_ <= token).min(groups.size - 1)
          groups(g)(axis * counts(g) + token - starts(g))
        }
      }
      val (positions, layout, target) =
        if (condition.editing) {
          val semanticPositions =
            condition.semantic.fold(Array.empty[Int])(_._2)
          // the refined images move after the captions
          ops.copy(
            stream.rows(0, images.toLong * each),
            w.projected
              .view(w.tokens * width)
              .prefix(images.toLong * each, width)
          )
          ops.copy(condition.caption, stream.rows(0, captions))
          ops.copy(
            w.projected
              .view(w.tokens * width)
              .prefix(images.toLong * each, width),
            stream.rows(captions, images.toLong * each)
          )
          condition.semantic.foreach((semantic, _) =>
            ops.copy(
              semantic,
              stream.rows(captions + images.toLong * each, semantics)
            )
          )
          val half = condition.captionLength
          (
            joined(
              Seq(condition.captionPositions, imagePositions, semanticPositions)
                .take(if (semantics > 0) 3 else 2),
              Seq(captions, images * each, semantics)
                .take(if (semantics > 0) 3 else 2)
            ),
            Seq(
              (0, half, false),
              (half, half, true),
              (captions, each, false),
              (captions + each, each, true),
              (captions + 2 * each, semantics, false)
            ).filter(_._2 > 0),
            captions + each
          )
        } else {
          ops.copy(condition.caption, stream.rows(each, captions))
          (
            joined(
              Seq(imagePositions, condition.captionPositions),
              Seq(each, captions)
            ),
            Seq((0, tokens, true)),
            0
          )
        }
      val at = place(w, positions, tokens)
      layers.indices.foreach(i =>
        run(layers(i), w, tokens, at, spans(c.refiners + i, layout))
      )

      // the final layer on the target's tokens, and the sign turned
      val rows = stream.rows(target, grid.tokens)
      val normed = w.normed.view(w.tokens * width).prefix(grid.tokens, width)
      ops.layerNorm(rows, None, None, 1e-6f, normed)
      ops.modulate(normed, noisyFinal, zero, normed)
      affine(normed, finalOut, out)
      ops.scale(out, -1f, out)
    } finally made.foreach(ops.release)
  }

  def close(): Unit = {
    attention.close()
    small.foreach(ops.release)
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object LladaImage {

  private def bf16(value: Float): Float = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    java.lang.Float.intBitsToFloat(
      ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16
    )
  }

  /** The timestep's sinusoid as the released pipeline computes it in BF16: σ
    * rounded, times 1000 rounded, the product with each frequency in F32, its
    * cosine (first half) and sine (second) rounded.
    */
  def sinusoid(timestep: Float, channels: Int): Array[Float] = {
    val half = channels / 2
    val level = bf16(bf16(timestep) * 1000f)
    Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(10000) * (i % half) / half).toFloat
      val angle = (level * frequency).toDouble
      bf16((if (i < half) math.cos(angle) else math.sin(angle)).toFloat)
    }
  }

  def open(ops: Ops, path: Path): LladaImage = {
    val source = WeightSource.open(ops, path)
    try new LladaImage(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
