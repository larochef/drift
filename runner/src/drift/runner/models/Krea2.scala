package drift.runner.models

import drift.runner.diffusion.*
import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.nio.file.Path
import scala.collection.mutable

/** Krea 2's transformer, read off its weights: `hidden` wide, `blocks` deep,
  * heads of `headDimension` (queries over fewer key-value heads), and the text
  * encoder's `textLayers` taps of `textWidth` fused by `textHeads` heads.
  */
final case class Krea2Config(
    hidden: Int,
    blocks: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    intermediate: Int,
    textWidth: Int,
    textLayers: Int,
    textHeads: Int,
    textIntermediate: Int,
    layerwiseBlocks: Int,
    refinerBlocks: Int,
    latentChannels: Int
) {

  /** RoPE's three axes (frame, row, column) in pairs, sd-cpp's rule
    * `{d − 12u, 6u, 6u}` values with `u = d / 16`: 32/48/48 for heads of 128.
    */
  def ropeAxes: RopeSections.Axes = {
    val u = headDimension / 16
    RopeSections.Axes(Seq((headDimension - 12 * u) / 2, 3 * u, 3 * u))
  }
}

object Krea2Config {
  val Theta = 1000f
  val Epsilon = 1e-5f

  def of(source: WeightSource): Krea2Config = {
    def dimensions(name: String) = source(name).shape.dimensions.map(_.toInt)
    def count(prefix: String) =
      Iterator
        .from(0)
        .takeWhile(i => source.has(s"$prefix.$i.prenorm.scale"))
        .size
    val Seq(hidden, latentChannels) = dimensions("first.weight")
    val headDimension = dimensions("blocks.0.attn.qknorm.qnorm.scale").head
    val textHead =
      dimensions("txtfusion.refiner_blocks.0.attn.qknorm.qnorm.scale").head
    val textWidth = dimensions("txtmlp.1.weight").last
    Krea2Config(
      hidden = hidden,
      blocks = count("blocks"),
      heads = hidden / headDimension,
      kvHeads = dimensions("blocks.0.attn.wk.weight").head / headDimension,
      headDimension = headDimension,
      intermediate = dimensions("blocks.0.mlp.gate.weight").head,
      textWidth = textWidth,
      textLayers = dimensions("txtfusion.projector.weight").last,
      textHeads = textWidth / textHead,
      textIntermediate =
        dimensions("txtfusion.refiner_blocks.0.mlp.gate.weight").head,
      layerwiseBlocks = count("txtfusion.layerwise_blocks"),
      refinerBlocks = count("txtfusion.refiner_blocks"),
      latentChannels = latentChannels
    )
  }
}

/** Krea 2's diffusion transformer (diffusers' `Krea2Transformer2DModel`,
  * sd-cpp's `krea2.hpp`), from a single-file checkpoint's original names.
  *   - The text: the encoder's layer taps fused per token (attention across the
  *     taps, a learned sum, attention across the tokens), then an MLP to the
  *     model's width. It depends on the prompt only: `encodeText` once per
  *     image.
  *   - The image: single-stream blocks over [text ; image] tokens with
  *     AdaLN-single modulation (one timestep vector of six, plus each block's
  *     table): gated attention with q/k RMS norm and a three-axis RoPE (text at
  *     position 0), and a SwiGLU MLP. `velocity` per denoising step. RMS norms
  *     take `1 + w`; the weights are used as stored (BF16 through BF16 GEMMs),
  *     their small vectors decoded to F32 at load.
  */
final class Krea2 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: Krea2Config = Krea2Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = Krea2Config.Epsilon

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  /** As `floats`, but always a copy of its own: a table a LoRA rewrites. */
  private def ownedFloats(name: String, count: Long): Tensor = {
    val copy = weights.keep(ops.allocate(DType.F32, Shape.of(count)))
    ops.copy(floats(name, count), copy)
    copy
  }

  /** A linear layer with a bias, by its original name. */
  final private case class Affine(name: String, weight: Tensor, bias: Tensor)

  private def affineLayer(prefix: String, outputs: Long): Affine =
    Affine(prefix, source(s"$prefix.weight"), floats(s"$prefix.bias", outputs))

  /** A block's attention and MLP weights (a text-fusion block, or a main block
    * but for its modulation table), with heads of `head`.
    */
  final private class Block(val prefix: String, width: Int, head: Int) {
    val prenorm: Tensor = floats(s"$prefix.prenorm.scale", width)
    val postnorm: Tensor = floats(s"$prefix.postnorm.scale", width)
    val q: Tensor = source(s"$prefix.attn.wq.weight")
    val k: Tensor = source(s"$prefix.attn.wk.weight")
    val v: Tensor = source(s"$prefix.attn.wv.weight")
    val gate: Tensor = source(s"$prefix.attn.gate.weight")
    val o: Tensor = source(s"$prefix.attn.wo.weight")
    val qNorm: Tensor = floats(s"$prefix.attn.qknorm.qnorm.scale", head)
    val kNorm: Tensor = floats(s"$prefix.attn.qknorm.knorm.scale", head)
    val mlpGate: Tensor = source(s"$prefix.mlp.gate.weight")
    val mlpUp: Tensor = source(s"$prefix.mlp.up.weight")
    val mlpDown: Tensor = source(s"$prefix.mlp.down.weight")
  }

  private val textHead = c.textWidth / c.textHeads
  private val layerwise = (0 until c.layerwiseBlocks).map(i =>
    new Block(s"txtfusion.layerwise_blocks.$i", c.textWidth, textHead)
  )
  private val refiner = (0 until c.refinerBlocks).map(i =>
    new Block(s"txtfusion.refiner_blocks.$i", c.textWidth, textHead)
  )

  /** The learned sum over the taps: one weight per tap, on the host (a LoRA's
    * delta added when one targets it).
    */
  private val projectorBase: Array[Float] =
    source("txtfusion.projector.weight").dtype.decode(
      MemorySegment.ofArray(weights.hostBytes("txtfusion.projector.weight")),
      c.textLayers
    )
  private var projector: Array[Float] = projectorBase
  private val textNorm = floats("txtmlp.0.scale", c.textWidth)
  private val textIn = affineLayer("txtmlp.1", c.hidden)
  private val textOut = affineLayer("txtmlp.3", c.hidden)
  private val imageIn = affineLayer("first", c.hidden)
  private val timeIn = affineLayer("tmlp.0", c.hidden)
  private val timeOut = affineLayer("tmlp.2", c.hidden)
  private val timeModulation = affineLayer("tproj.1", 6L * c.hidden)
  private val blocks = (0 until c.blocks).map(i =>
    (
      new Block(s"blocks.$i", c.hidden, c.headDimension),
      ownedFloats(s"blocks.$i.mod.lin", 6L * c.hidden)
    )
  )
  private val finalNorm = floats("last.norm.scale", c.hidden)
  private val finalModulation =
    ownedFloats("last.modulation.lin", 2L * c.hidden)
  private val finalOut = affineLayer("last.linear", c.latentChannels)

  private val rope = Rope(
    Krea2Config.Theta,
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

  /** Flat buffers for `tokens` rows of the widest operand, grown when too
    * small; each use takes the `prefix` it needs.
    */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(dtype: DType, elements: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(elements))
      held += tensor
      tensor
    }
    private val widest = tokens.toLong * math.max(c.hidden, c.textWidth)
    private val widestMlp =
      tokens.toLong * math.max(c.intermediate, c.textIntermediate)
    val x: Tensor = flat(DType.F32, widest)
    val normed: Tensor = flat(DType.F32, widest)
    val q: Tensor = flat(DType.F32, widest)
    val k: Tensor = flat(DType.F32, widest)
    val v: Tensor = flat(DType.F32, widest)
    val gate: Tensor = flat(DType.F32, widest)
    val attended: Tensor = flat(DType.F32, widest)
    val projected: Tensor = flat(DType.F32, widest)
    val rotatedKeys: Tensor = flat(DType.F32, widest)
    val mlpGate: Tensor = flat(DType.F32, widestMlp)
    val mlpUp: Tensor = flat(DType.F32, widestMlp)
    val positions: Tensor = flat(DType.I32, 3L * tokens)
    // attention's keys and values, one page for the whole sequence (pages
    // hold a multiple of 16 tokens)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, c.kvHeads, c.headDimension)
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

  /** `out = x · weight + bias`, and the active LoRAs' updates. */
  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    lora(x, layer.name, out)
    ops.addRow(out, layer.bias, out)
  }

  // ---- LoRAs ---------------------------------------------------------------------

  /** The active LoRAs' updates of each linear layer, by its original name. */
  private val updates = new LoraUpdates(ops)

  /** Tables a LoRA may target (not linears): their values as loaded. */
  private lazy val tables: Map[String, (Tensor, Array[Float])] =
    (("last.modulation.lin" -> finalModulation) +: blocks.zipWithIndex.map(
      (b, i) => s"blocks.$i.mod.lin" -> b._2
    ))
      .map((name, tensor) => name -> (tensor, ops.toFloats(tensor)))
      .toMap

  private def linearNames: Set[String] =
    (Seq(textIn, textOut, imageIn, timeIn, timeOut, timeModulation, finalOut)
      .map(_.name) ++
      (layerwise ++ refiner ++ blocks.map(_._1)).flatMap(b =>
        Seq(
          "attn.wq",
          "attn.wk",
          "attn.wv",
          "attn.gate",
          "attn.wo",
          "mlp.gate",
          "mlp.up",
          "mlp.down"
        )
          .map(part => s"${b.prefix}.$part")
      )).toSet

  /** Makes `loras` (each at its multiplier) the active set, replacing the last:
    * linears take their updates at run time, tables (the modulations, the text
    * projector) are rebuilt from their loaded values plus the deltas. Returns
    * the targets that matched nothing, left unapplied.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val all = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map((target, pair) =>
        Krea2.originalName(target) -> pair.copy(scale = pair.scale * multiplier)
      )
    )
    val linears = linearNames
    updates.use(
      all.filter((target, _) => linears.contains(target)).groupMap(_._1)(_._2)
    )
    val byTable =
      all.filter((target, _) => tables.contains(target)).groupMap(_._1)(_._2)
    tables.foreach { case (name, (tensor, base)) =>
      val values = byTable
        .getOrElse(name, Nil)
        .foldLeft(base)((values, pair) => plus(values, pair))
      val uploaded = ops.fromFloats(tensor.shape, values)
      try ops.copy(uploaded, tensor)
      finally ops.release(uploaded)
    }
    projector = all
      .filter(_._1 == "txtfusion.projector")
      .map(_._2)
      .foldLeft(projectorBase)((values, pair) => plus(values, pair))
    all
      .map(_._1)
      .distinct
      .filterNot(t =>
        linears.contains(t) || tables.contains(t) || t == "txtfusion.projector"
      )
  }

  /** `values + scale × up · down`, the product computed on the host. */
  private def plus(values: Array[Float], pair: LoraPair): Array[Float] = {
    val (down, up) = (floatsOf(pair.down), floatsOf(pair.up))
    val (rank, in) = (pair.rank, pair.down.shape.dimensions.last.toInt)
    Array.tabulate(values.length) { i =>
      val (row, column) = (i / in, i % in)
      values(i) + pair.scale * (0 until rank)
        .map(r => up(row * rank + r) * down(r * in + column))
        .sum
    }
  }

  private def floatsOf(tensor: Tensor): Array[Float] =
    if (tensor.dtype == DType.F32) ops.toFloats(tensor)
    else {
      val floats = ops.allocate(DType.F32, tensor.shape)
      try {
        ops.convert(tensor, floats)
        ops.toFloats(floats)
      } finally ops.release(floats)
    }

  private def lora(x: Tensor, target: String, out: Tensor): Unit =
    updates(x, target, out)

  /** A block's attention and MLP on `x` (`[rows, width]`, in place). With a
    * `modulation` (a main block's six rows of `width`: prescale, preshift,
    * pregate, postscale, postshift, postgate) the norms are modulated and the
    * residuals gated. `attend(q, k, v, out)` runs the attention itself.
    */
  private def block(
      b: Block,
      x: Tensor,
      heads: Int,
      kvHeads: Int,
      head: Int,
      intermediate: Int,
      modulation: Option[Tensor],
      w: Buffers,
      attend: (Tensor, Tensor, Tensor, Tensor) => Unit
  ): Unit = {
    val Seq(rows, width) = x.shape.dimensions
    val kvWidth = kvHeads.toLong * head
    def vector(i: Int) =
      modulation.map(_.prefix(6 * width).view(6, width).rows(i, 1).view(width))
    def residual(y: Tensor, gate: Option[Tensor]) = gate match {
      case Some(g) => ops.gatedAdd(x, y, g)
      case None    => ops.add(x, y, x)
    }
    val normed = w.normed.prefix(rows, width)
    ops.rmsNorm(x, b.prenorm, epsilon, 1f, normed)
    for {
      scale <- vector(0)
      shift <- vector(1)
    }
      ops.modulate(normed, scale, shift, normed)
    val (q, k, v, gate) = (
      w.q.prefix(rows, width),
      w.k.prefix(rows, kvWidth),
      w.v.prefix(rows, kvWidth),
      w.gate.prefix(rows, width)
    )
    ops.linears(normed, Seq(b.q, b.k, b.v, b.gate), Seq(q, k, v, gate))
    Seq("attn.wq" -> q, "attn.wk" -> k, "attn.wv" -> v, "attn.gate" -> gate)
      .foreach((part, y) => lora(normed, s"${b.prefix}.$part", y))
    ops.rmsNorm(
      q.view(rows * heads, head),
      b.qNorm,
      epsilon,
      1f,
      q.view(rows * heads, head)
    )
    ops.rmsNorm(
      k.view(rows * kvHeads, head),
      b.kNorm,
      epsilon,
      1f,
      k.view(rows * kvHeads, head)
    )
    val attended = w.attended.prefix(rows, width)
    attend(q, k, v, attended)
    ops.activation(Activation.Sigmoid, gate, gate)
    ops.mul(attended, gate, attended)
    val projected = w.projected.prefix(rows, width)
    ops.linear(attended, b.o, projected)
    lora(attended, s"${b.prefix}.attn.wo", projected)
    residual(projected, vector(2))
    ops.rmsNorm(x, b.postnorm, epsilon, 1f, normed)
    for {
      scale <- vector(3)
      shift <- vector(4)
    }
      ops.modulate(normed, scale, shift, normed)
    val (mlpGate, mlpUp) =
      (w.mlpGate.prefix(rows, intermediate), w.mlpUp.prefix(rows, intermediate))
    ops.linears(normed, Seq(b.mlpGate, b.mlpUp), Seq(mlpGate, mlpUp))
    lora(normed, s"${b.prefix}.mlp.gate", mlpGate)
    lora(normed, s"${b.prefix}.mlp.up", mlpUp)
    ops.gated(Activation.Silu, mlpGate, mlpUp, mlpGate)
    ops.linear(mlpGate, b.mlpDown, projected)
    lora(mlpGate, s"${b.prefix}.mlp.down", projected)
    residual(projected, vector(5))
  }

  /** The prompt's text for every step: the encoder's taps (`taps`, tap-major
    * `[layers × L, width]` as `Qwen3.encode` gives them, overwritten) fused and
    * projected into `out` (`[L, hidden]`).
    */
  def encodeText(taps: Tensor, out: Tensor): Unit = {
    val layers = c.textLayers.toLong
    val tokens = taps.shape.dimensions.head / layers
    require(
      taps.shape == Shape.of(layers * tokens, c.textWidth) &&
        out.shape == Shape.of(tokens, c.hidden),
      s"encodeText: taps ${taps.shape}, out ${out.shape}"
    )
    val w = buffersFor((layers * tokens).toInt)
    val (heads, head) = (c.textHeads.toLong, textHead.toLong)
    // `sequences` short sequences, sequence-major rows
    def across(
        sequences: Long
    )(q: Tensor, k: Tensor, v: Tensor, out: Tensor) = {
      val positions = q.shape.dimensions.head / sequences
      def view(t: Tensor) = t.view(positions, sequences, heads, head)
      ops.shortAttention(
        view(q),
        view(k),
        view(v),
        (1 / math.sqrt(head.toDouble)).toFloat,
        view(out)
      )
    }
    def fusion(b: Block, x: Tensor, sequences: Long) =
      block(
        b,
        x,
        c.textHeads,
        c.textHeads,
        textHead,
        c.textIntermediate,
        None,
        w,
        across(sequences)
      )
    // across each token's taps: the rows are tap-major, a tap is a position
    layerwise.foreach(fusion(_, taps, tokens))
    // the learned sum over the taps
    val text = w.x.prefix(tokens, c.textWidth)
    val scaled = w.normed.prefix(tokens, c.textWidth)
    ops.zero(text)
    (0 until c.textLayers).foreach { n =>
      ops.scale(taps.rows(n * tokens, tokens), projector(n), scaled)
      ops.add(text, scaled, text)
    }
    // across the tokens: one sequence
    refiner.foreach(fusion(_, text, 1))
    val normed = w.projected.prefix(tokens, c.textWidth)
    ops.rmsNorm(text, textNorm, epsilon, 1f, normed)
    val inner = w.q.prefix(tokens, c.hidden)
    affine(normed, textIn, inner)
    ops.activation(Activation.GeluTanh, inner, inner)
    affine(inner, textOut, out)
  }

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (`[gridHeight ×
    * gridWidth, channels]`, the packed patches row-major), given the prompt's
    * `text` (`[L, hidden]` from `encodeText`), into `out` (like `latents`).
    */
  def velocity(
      latents: Tensor,
      text: Tensor,
      timestep: Float,
      gridHeight: Int,
      gridWidth: Int,
      out: Tensor
  ): Unit = {
    val (length, images) =
      (text.shape.dimensions.head.toInt, gridHeight * gridWidth)
    val tokens = length + images
    require(
      latents.shape == Shape.of(images, c.latentChannels) &&
        out.shape == latents.shape,
      s"velocity: latents ${latents.shape}, out ${out.shape} for a $gridHeight × $gridWidth grid"
    )
    val w = buffersFor(tokens)
    val f = c.hidden.toLong
    // the timestep: a sinusoid (cosines first) through the time MLP
    val half = timeIn.weight.shape.last.toInt / 2
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
      val (modulation, blockModulation) = (vector(6 * f), vector(6 * f))
      affine(embedded, timeIn, inner)
      ops.activation(Activation.GeluTanh, inner, inner)
      affine(inner, timeOut, time)
      ops.activation(Activation.GeluTanh, time, inner)
      affine(inner, timeModulation, modulation)
      // [text ; image]; positions (frame, row, column), the text's all 0
      val stream = w.x.prefix(tokens, f)
      ops.copy(text, stream.rows(0, length))
      affine(latents, imageIn, stream.rows(length, images))
      val positions = w.positions.prefix(3, tokens)
      ops.writeInts(
        positions.view(3L * tokens),
        new Array[Int](tokens) ++
          (Array.fill(length)(0) ++ Array.tabulate(images)(_ / gridWidth)) ++
          (Array.fill(length)(0) ++ Array.tabulate(images)(_ % gridWidth))
      )
      val (heads, kvHeads, d) =
        (c.heads.toLong, c.kvHeads.toLong, c.headDimension.toLong)
      def attend(q: Tensor, k: Tensor, v: Tensor, attended: Tensor): Unit = {
        // the rotated queries in place of the normed input, which is spent
        val rotatedQueries = w.normed.prefix(tokens, heads, d)
        val rotatedKeys = w.rotatedKeys.prefix(tokens, kvHeads, d)
        ops.rope(q.view(tokens, heads, d), positions, rope, rotatedQueries)
        ops.rope(k.view(tokens, kvHeads, d), positions, rope, rotatedKeys)
        ops.cacheWrite(
          rotatedKeys,
          v.view(tokens, kvHeads, d),
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
      blocks.foreach { (b, table) =>
        ops.add(modulation, table.view(1, 6 * f), blockModulation)
        block(
          b,
          stream,
          c.heads,
          c.kvHeads,
          c.headDimension,
          c.intermediate,
          Some(blockModulation),
          w,
          attend
        )
      }
      // the final layer on the image tokens, modulated by the time MLP's output
      val normed = w.normed.prefix(images, f)
      ops.rmsNorm(stream.rows(length, images), finalNorm, epsilon, 1f, normed)
      val (scale, shift) = (vector(f), vector(f))
      val table = finalModulation.view(2, f)
      ops.add(table.rows(0, 1), time, scale)
      ops.add(table.rows(1, 1), time, shift)
      ops.modulate(normed, scale.view(f), shift.view(f), normed)
      affine(normed, finalOut, out)
    } finally small.foreach(ops.release)
  }

  def close(): Unit = {
    updates.close()
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object Krea2 {

  /** diffusers' Krea 2 names → the original ones. */
  private val Renames: Seq[(String, String)] = Seq(
    "^transformer_blocks\\." -> "blocks.",
    "^text_fusion\\." -> "txtfusion.",
    "^img_in$" -> "first",
    "^time_embed\\.linear_1$" -> "tmlp.0",
    "^time_embed\\.linear_2$" -> "tmlp.2",
    "^time_mod_proj$" -> "tproj.1",
    "^txt_in\\.linear_1$" -> "txtmlp.1",
    "^txt_in\\.linear_2$" -> "txtmlp.3",
    "^final_layer\\.linear$" -> "last.linear",
    "^final_layer\\.scale_shift_table$" -> "last.modulation.lin",
    "\\.attn\\.to_q$" -> ".attn.wq",
    "\\.attn\\.to_k$" -> ".attn.wk",
    "\\.attn\\.to_v$" -> ".attn.wv",
    "\\.attn\\.to_gate$" -> ".attn.gate",
    "\\.attn\\.to_out\\.0$" -> ".attn.wo",
    "\\.ff\\.(gate|up|down)$" -> ".mlp.$1",
    "\\.scale_shift_table$" -> ".mod.lin"
  )

  /** A LoRA target in the original naming, as the file's or diffusers'. */
  def originalName(target: String): String =
    Renames.foldLeft(target)((name, rename) =>
      name.replaceAll(rename._1, rename._2)
    )

  def open(ops: Ops, path: Path): Krea2 = {
    val source = WeightSource.open(ops, path)
    try {
      if (!source.has("txtfusion.projector.weight"))
        throw new FormatException(
          s"$path is no Krea 2 checkpoint (no txtfusion.projector.weight)"
        )
      new Krea2(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
