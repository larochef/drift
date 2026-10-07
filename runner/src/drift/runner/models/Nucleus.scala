package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import java.util.stream.IntStream
import scala.collection.mutable

/** Nucleus-Image's transformer, read off its weights (released: 2048 wide, 32
  * blocks, 16 query heads over 4 key-value heads of 128, text features of 4096,
  * 64 packed latent features, 64 routed experts of 1344 from the fourth block
  * on). What the weights do not say comes from the `config.json` beside them,
  * else as released: each routed block's `capacityFactors` (4 for the first
  * two, then 2) and the `routeScale` (2.5).
  */
final case class NucleusConfig(
    hidden: Int,
    blocks: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    textWidth: Int,
    latentChannels: Int,
    outChannels: Int,
    experts: Int,
    capacityFactors: Seq[Double],
    routeScale: Float
) {

  /** RoPE's three axes (image, row, column) in pairs: 16/56/56 for heads of
    * 128.
    */
  def ropeAxes: RopeSections.Axes = {
    val image = headDimension / 16
    val side = (headDimension / 2 - image) / 2
    RopeSections.Axes(Seq(image, side, side))
  }
}

object NucleusConfig {
  val Theta = 10000f
  val Epsilon = 1e-6f

  /** Whether the weights are a Nucleus-Image transformer's. */
  def holds(source: WeightSource): Boolean =
    source.has("transformer_blocks.0.encoder_proj.weight") &&
      source.has("transformer_blocks.0.img_mod.1.weight")

  /** Whether block `i` routes its tokens to experts. */
  def routed(source: WeightSource, i: Int): Boolean =
    source.has(s"transformer_blocks.$i.img_mlp.experts.gate_up_proj")

  def of(source: WeightSource): NucleusConfig = {
    if (!holds(source))
      throw new FormatException(
        "no Nucleus-Image transformer (transformer_blocks.0.encoder_proj.weight)"
      )
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    val Seq(hidden, latentChannels) = dimensions("img_in.weight")
    val headDimension =
      dimensions("transformer_blocks.0.attn.norm_q.weight").head
    val blocks = Iterator
      .from(0)
      .takeWhile(i => source.has(s"transformer_blocks.$i.attn.to_q.weight"))
      .size
    val routedBlocks = (0 until blocks).filter(routed(source, _))
    val configured = source.config.filter(_.has("capacity_factors"))
    if (
      configured.exists(c => c.has("use_sigmoid") && c.boolean("use_sigmoid"))
    )
      throw new FormatException(
        "a Nucleus-Image router of sigmoids is not supported"
      )
    NucleusConfig(
      hidden = hidden,
      blocks = blocks,
      heads = hidden / headDimension,
      kvHeads = dimensions(
        "transformer_blocks.0.attn.to_k.weight"
      ).head / headDimension,
      headDimension = headDimension,
      textWidth = dimensions("txt_norm.weight").head,
      latentChannels = latentChannels,
      outChannels = dimensions("proj_out.weight").head,
      experts = routedBlocks.headOption.fold(0)(i =>
        dimensions(s"transformer_blocks.$i.img_mlp.gate.weight").head
      ),
      capacityFactors = configured match {
        case Some(config) =>
          config.json("capacity_factors") match {
            case ujson.Arr(values) => values.map(_.num).toSeq
            case one               => Seq.fill(blocks)(one.num)
          }
        case None =>
          (0 until blocks).map(i =>
            if (!routedBlocks.contains(i)) 0.0
            else if (routedBlocks.indexOf(i) < 2) 4.0
            else 2.0
          )
      },
      routeScale = configured
        .filter(_.has("route_scale"))
        .fold(2.5f)(_.double("route_scale").toFloat)
    )
  }
}

/** A prompt's text as every block's attention takes it, for an image of one
  * size: each block's keys (normed and turned) and values, `[length, kvHeads ×
  * headDimension]`, which do not change from step to step; `largest` the
  * largest value of each block.
  */
final class NucleusText private[models] (
    ops: Ops,
    val length: Int,
    val keys: Seq[Tensor],
    val values: Seq[Tensor],
    val largest: Seq[Float],
    private[models] val scaled: Tensor
) extends AutoCloseable {
  def close(): Unit = (keys ++ values :+ scaled).foreach(ops.release)
}

/** Nucleus-Image's diffusion transformer (diffusers'
  * `NucleusMoEImageTransformer2DModel`), from diffusers' names.
  *   - Image tokens alone run through the blocks: the packed latents through
  *     `img_in`. The text never does: after an RMS norm each block projects it
  *     (`encoder_proj`) to keys and values of its own, which the image's
  *     queries see beside the image's (`text`, once per prompt).
  *   - Attention: 16 query heads over 4 key-value heads, RMS norms on q and k
  *     per head, RoPE on three axes (theta 10000, 16/56/56): the image at (0,
  *     row, column), both centred; text token `i` at `max(h, w) / 2 + i` on all
  *     three. No biases but `img_in`, `encoder_proj`, the timestep's and the
  *     modulations'.
  *   - The timestep: a sinusoid of `1000 σ` as wide as the model (cosines
  *     first) through an MLP and an RMS norm; the released pipeline runs in
  *     BF16, so σ and the sinusoid are rounded as it rounds them. Each block
  *     takes two scales and two gates from it (no shift), the gates held to ±2
  *     and through `tanh`.
  *   - The MLP: the first blocks a dense SwiGLU; the others a mixture of
  *     experts where the experts choose. A router scores every token for every
  *     expert from the timestep and the token before its modulation (a softmax
  *     over the experts); each expert takes its `capacity` best tokens
  *     (`⌈factor × tokens / experts⌉`), weighted by its score over the sum of
  *     the scores of the experts that took the token, times `routeScale`; a
  *     shared expert takes every token. The experts' weights are stored as
  *     `[experts, in, out]` and transposed at load into the layout the GEMMs
  *     take.
  *   - The final layer: a layer norm modulated by (scale ; shift), `proj_out`.
  *     The model gives the velocity's opposite, which `velocity` turns back.
  */
final class Nucleus private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: NucleusConfig = NucleusConfig.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = false)
  private val epsilon = NucleusConfig.Epsilon
  private val width = c.hidden.toLong
  private val kvWidth = c.kvHeads.toLong * c.headDimension

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))
  private def matrix(name: String): Tensor = source(s"$name.weight")

  /** A linear layer with a bias. */
  final private case class Affine(weight: Tensor, bias: Tensor)
  private def affineLayer(name: String, outputs: Long): Affine =
    Affine(matrix(name), floats(s"$name.bias", outputs))

  /** A SwiGLU as diffusers' `FeedForward` stores it: `net.0.proj` holds the
    * values then their gates.
    */
  final private class SwiGlu(prefix: String) {
    private val in = matrix(s"$prefix.net.0.proj")
    val inner: Long = in.shape.dimensions.head / 2
    val up: Tensor = in.rows(0, inner)
    val gate: Tensor = in.rows(inner, inner)
    val down: Tensor = matrix(s"$prefix.net.2")
  }

  /** Floats as BF16 (round to nearest even), uploaded. */
  private def bf16(shape: Shape, values: Array[Float]): Tensor = {
    val bytes = new Array[Byte](2 * values.length)
    values.indices.foreach { i =>
      val bits = java.lang.Float.floatToRawIntBits(values(i))
      val rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16
      bytes(2 * i) = rounded.toByte
      bytes(2 * i + 1) = (rounded >>> 8).toByte
    }
    weights.upload(DType.BF16, shape, bytes)
  }

  /** A mixture of experts: the router's halves (the timestep's columns, the
    * token's), the experts' weights transposed (`[experts, 2 × inner, hidden]`
    * gates then values, `[experts, hidden, inner]`), the shared expert.
    */
  final private class Mixture(prefix: String, val capacityFactor: Double) {
    private val router = weights.hostFloats(s"$prefix.gate.weight")
    private def half(first: Int) =
      bf16(
        Shape.of(c.experts, width),
        Array.tabulate(c.experts * c.hidden)(i =>
          router(i / c.hidden * 2 * c.hidden + first + i % c.hidden)
        )
      )
    val routerTime: Tensor = half(0)
    val routerToken: Tensor = half(c.hidden)

    /** `[experts, in, out]` as stored into `[experts, out, in]`. */
    private def transposed(name: String): Tensor = {
      val stored = source(name)
      val Seq(experts, in, out) = stored.shape.dimensions
      val made = weights.keep(
        ops.allocate(stored.dtype, Shape.of(experts, out, in))
      )
      (0L until experts).foreach(e =>
        ops.transpose(
          stored.view(experts, in * out).rows(e, 1).view(in, out),
          made.view(experts, out * in).rows(e, 1).view(out, in)
        )
      )
      made
    }
    private val allGateUp = transposed(s"$prefix.experts.gate_up_proj")
    private val allDown = transposed(s"$prefix.experts.down_proj")
    val inner: Long = allDown.shape.last
    def gateUp(expert: Int): Tensor =
      allGateUp
        .view(c.experts, 2 * inner * width)
        .rows(expert, 1)
        .view(2 * inner, width)
    def down(expert: Int): Tensor =
      allDown.view(c.experts, width * inner).rows(expert, 1).view(width, inner)
    val shared = new SwiGlu(s"$prefix.shared_expert")
  }

  final private class Block(index: Int) {
    private val prefix = s"transformer_blocks.$index"
    val modulation: Affine = affineLayer(s"$prefix.img_mod.1", 4 * width)
    val textIn: Affine = affineLayer(s"$prefix.encoder_proj", width)
    val textKey: Tensor = matrix(s"$prefix.attn.add_k_proj")
    val textValue: Tensor = matrix(s"$prefix.attn.add_v_proj")
    val textKeyNorm: Tensor =
      floats(s"$prefix.attn.norm_added_k.weight", c.headDimension)
    val query: Tensor = matrix(s"$prefix.attn.to_q")
    val key: Tensor = matrix(s"$prefix.attn.to_k")
    val value: Tensor = matrix(s"$prefix.attn.to_v")
    val output: Tensor = matrix(s"$prefix.attn.to_out.0")
    val queryNorm: Tensor =
      floats(s"$prefix.attn.norm_q.weight", c.headDimension)
    val keyNorm: Tensor = floats(s"$prefix.attn.norm_k.weight", c.headDimension)
    val mlp: Either[SwiGlu, Mixture] =
      if (NucleusConfig.routed(source, index))
        Right(new Mixture(s"$prefix.img_mlp", c.capacityFactors(index)))
      else Left(new SwiGlu(s"$prefix.img_mlp"))
  }

  private val blocks = (0 until c.blocks).map(new Block(_))
  private val textNorm = floats("txt_norm.weight", c.textWidth)
  private val imageIn = affineLayer("img_in", width)
  private val timeIn = affineLayer(
    "time_text_embed.timestep_embedder.linear_1",
    matrix("time_text_embed.timestep_embedder.linear_1").shape.dimensions.head
  )
  private val timeOut =
    affineLayer("time_text_embed.timestep_embedder.linear_2", width)
  private val timeNorm = floats("time_text_embed.norm.weight", width)
  private val finalModulation = affineLayer("norm_out.linear", 2 * width)
  private val finalOut = matrix("proj_out")

  private val rope = Rope(
    NucleusConfig.Theta,
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
  private val zero = weights.keep(
    ops.fromFloats(Shape.of(width), new Array[Float](c.hidden))
  )

  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  private def headNorm(x: Tensor, heads: Long, weight: Tensor): Unit = {
    val rows = x.shape.dimensions.head * heads
    val d = c.headDimension.toLong
    ops.rmsNorm(x.view(rows, d), weight, epsilon, 0f, x.view(rows, d))
  }

  /** The attention's values go through an F16 cache (largest 65504): values
    * beyond this are scaled down by a power of two before it and the output
    * back, as for Mage-Flow.
    */
  private val ValueLimit = 8192f
  private def factorFor(largest: Float): Float =
    if (largest <= ValueLimit) 1f
    else
      math
        .pow(2, math.ceil(math.log(largest / ValueLimit) / math.log(2)))
        .toFloat

  /** The prompt's text for every step of an image on a `gridHeight × gridWidth`
    * grid: the encoder's hidden state (`[L, textWidth]`) as each block's keys
    * and values.
    */
  def text(features: Tensor, gridHeight: Int, gridWidth: Int): NucleusText = {
    val length = features.shape.dimensions.head
    require(
      features.shape == Shape.of(length, c.textWidth),
      s"text: features ${features.shape}"
    )
    val d = c.headDimension.toLong
    val temporaries = Seq(
      ops.allocate(DType.F32, features.shape),
      ops.allocate(DType.F32, Shape.of(length, width)),
      ops.allocate(DType.F32, Shape.of(length, kvWidth))
    )
    val Seq(normed, context, key) = temporaries
    val first = math.max(gridHeight / 2, gridWidth / 2)
    val positions = ops.fromInts(
      Shape.of(3, length),
      Array.tabulate(3 * length.toInt)(i => first + i % length.toInt)
    )
    try {
      ops.rmsNorm(features, textNorm, epsilon, 0f, normed)
      val made = blocks.map { block =>
        affine(normed, block.textIn, context)
        val keys = ops.allocate(DType.F32, Shape.of(length, kvWidth))
        val values = ops.allocate(DType.F32, Shape.of(length, kvWidth))
        ops.linears(
          context,
          Seq(block.textKey, block.textValue),
          Seq(key, values)
        )
        headNorm(key, c.kvHeads, block.textKeyNorm)
        ops.rope(
          key.view(length, c.kvHeads, d),
          positions,
          rope,
          keys.view(length, c.kvHeads, d)
        )
        (keys, values, ops.maxAbs(values))
      }
      new NucleusText(
        ops,
        length.toInt,
        made.map(_._1),
        made.map(_._2),
        made.map(_._3),
        ops.allocate(DType.F32, Shape.of(length, kvWidth))
      )
    } finally (temporaries :+ positions).foreach(ops.release)
  }

  /** Buffers for `tokens` image tokens after `length` text ones, remade when
    * either grows.
    */
  final private class Buffers(val tokens: Int, val length: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def rows(dtype: DType, count: Long, columns: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(count, columns))
      held += tensor
      tensor
    }
    private val widestMlp = blocks.map(_.mlp.fold(_.inner, _.shared.inner)).max
    private val mixtures = blocks.flatMap(_.mlp.toOption)
    val slots: Long = mixtures
      .map(m => c.experts.toLong * capacity(m, tokens))
      .maxOption
      .getOrElse(0L)
    private val expertInner = mixtures.map(_.inner).maxOption.getOrElse(0L)
    val x: Tensor = rows(DType.F32, tokens, width)
    val normed: Tensor = rows(DType.F32, tokens, width)
    val modulated: Tensor = rows(DType.F32, tokens, width)
    val q: Tensor = rows(DType.F32, tokens, width)
    val rotatedQueries: Tensor = rows(DType.F32, tokens, width)
    val k: Tensor = rows(DType.F32, tokens, kvWidth)
    val v: Tensor = rows(DType.F32, tokens, kvWidth)
    val rotatedKeys: Tensor = rows(DType.F32, tokens, kvWidth)
    val attended: Tensor = rows(DType.F32, tokens, width)
    val projected: Tensor = rows(DType.F32, tokens, width)
    val mlpGate: Tensor = rows(DType.F32, tokens, widestMlp)
    val mlpUp: Tensor = rows(DType.F32, tokens, widestMlp)
    val logits: Tensor = rows(DType.F32, tokens, math.max(1, c.experts))
    val gathered: Tensor = rows(DType.F32, math.max(1, slots), width)
    val slotOut: Tensor = rows(DType.F32, math.max(1, slots), width)
    val slotBoth: Tensor =
      rows(DType.F32, math.max(1, slots), 2 * math.max(1, expertInner))
    val slotGate: Tensor =
      rows(DType.F32, math.max(1, slots), math.max(1, expertInner))
    val slotUp: Tensor =
      rows(DType.F32, math.max(1, slots), math.max(1, expertInner))
    val positions: Tensor = rows(DType.I32, 3, tokens)
    val cache: KvCache = ops.allocateCache(
      1,
      (length + tokens + 15) / 16 * 16,
      c.kvHeads,
      c.headDimension
    )
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

  private def capacity(mixture: Mixture, tokens: Int): Int =
    math.min(
      tokens,
      math.max(1, math.ceil(mixture.capacityFactor * tokens / c.experts).toInt)
    )

  private var buffers = Option.empty[Buffers]
  private def buffersFor(tokens: Int, length: Int): Buffers =
    buffers.filter(b => b.tokens == tokens && b.length >= length).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens, length)
      buffers = Some(created)
      created
    }

  private val small = mutable.ArrayBuffer.empty[Tensor]
  private def vector(rows: Long, elements: Long): Tensor = {
    val tensor = ops.allocate(DType.F32, Shape.of(rows, elements))
    small += tensor
    tensor
  }
  private val timeInner = vector(1, timeIn.bias.shape.elementCount)
  private val time = vector(1, width)
  private val timeSilu = vector(1, width)
  private val rawModulations = vector(c.blocks, 4 * width)
  private val finalParts = vector(1, 2 * width)
  private val routerBias = vector(1, math.max(1, c.experts))

  /** The velocity at `timestep` (σ in [0, 1]) of `latents` (`[gridHeight ×
    * gridWidth, latentChannels]`, the packed patches row-major), given the
    * prompt's `text` (made for this grid), into `out` (`[tokens,
    * outChannels]`).
    */
  def velocity(
      latents: Tensor,
      text: NucleusText,
      timestep: Float,
      gridHeight: Int,
      gridWidth: Int,
      out: Tensor
  ): Unit = {
    val tokens = gridHeight * gridWidth
    require(
      latents.shape == Shape.of(tokens, c.latentChannels) &&
        out.shape == Shape.of(tokens, c.outChannels),
      s"velocity: latents ${latents.shape}, out ${out.shape} for a $gridHeight × $gridWidth grid"
    )
    val w = buffersFor(tokens, text.length)
    val (heads, kvHeads, d) =
      (c.heads.toLong, c.kvHeads.toLong, c.headDimension.toLong)

    // the timestep through its MLP and norm; every block's modulation from it
    val sinusoid = ops.fromFloats(
      Shape.of(1, width),
      Nucleus.sinusoid(timestep, c.hidden)
    )
    try affine(sinusoid, timeIn, timeInner)
    finally ops.release(sinusoid)
    ops.activation(Activation.Silu, timeInner, timeInner)
    affine(timeInner, timeOut, time)
    ops.rmsNorm(time, timeNorm, epsilon, 0f, time)
    ops.activation(Activation.Silu, time, timeSilu)
    blocks.zipWithIndex.foreach((block, i) =>
      affine(timeSilu, block.modulation, rawModulations.rows(i, 1))
    )
    affine(timeSilu, finalModulation, finalParts)
    // (scale, gate, scale, gate) per block: the gates held to ±2, through tanh
    val parts = ops.toFloats(rawModulations)
    parts.indices.foreach { i =>
      if (i / c.hidden % 2 == 1)
        parts(i) = math.tanh(math.max(-2f, math.min(2f, parts(i)))).toFloat
    }
    val modulations = ops.fromFloats(Shape.of(4L * c.blocks, width), parts)
    val perStep = mutable.ArrayBuffer(modulations)
    def part(block: Int, i: Int) =
      modulations.rows(4L * block + i, 1).view(width)

    try {
      affine(latents, imageIn, w.x)
      val (top, left) =
        (gridHeight - gridHeight / 2, gridWidth - gridWidth / 2)
      ops.writeInts(
        w.positions.view(3L * tokens),
        new Array[Int](tokens) ++
          Array.tabulate(tokens)(_ / gridWidth - top) ++
          Array.tabulate(tokens)(_ % gridWidth - left)
      )

      blocks.zipWithIndex.foreach { (block, b) =>
        ops.layerNorm(w.x, None, None, epsilon, w.normed)
        ops.modulate(w.normed, part(b, 0), zero, w.modulated)
        ops.linears(
          w.modulated,
          Seq(block.query, block.key, block.value),
          Seq(w.q, w.k, w.v)
        )
        headNorm(w.q, heads, block.queryNorm)
        headNorm(w.k, kvHeads, block.keyNorm)
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
        // keys and values: the text's, then the image's
        val factor =
          factorFor(math.max(ops.maxAbs(w.v), text.largest(b)))
        val textValues =
          if (factor == 1f) text.values(b)
          else {
            ops.scale(text.values(b), 1 / factor, text.scaled)
            ops.scale(w.v, 1 / factor, w.v)
            text.scaled
          }
        ops.cacheWrite(
          text.keys(b).view(text.length, kvHeads, d),
          textValues.view(text.length, kvHeads, d),
          w.cache,
          w.pageTable,
          0
        )
        ops.cacheWrite(
          w.rotatedKeys.view(tokens, kvHeads, d),
          w.v.view(tokens, kvHeads, d),
          w.cache,
          w.pageTable,
          text.length
        )
        ops.attention(
          w.rotatedQueries.view(tokens, heads, d),
          w.cache,
          w.pageTable,
          text.length,
          text.length + tokens,
          attention,
          w.attended.view(tokens, heads, d)
        )
        if (factor != 1f) ops.scale(w.attended, factor, w.attended)
        ops.linear(w.attended, block.output, w.projected)
        ops.gatedAdd(w.x, w.projected, part(b, 1))

        ops.layerNorm(w.x, None, None, epsilon, w.normed)
        ops.modulate(w.normed, part(b, 2), zero, w.modulated)
        def swiGlu(layer: SwiGlu): Unit = {
          val gate = w.mlpGate
            .view(tokens.toLong * w.mlpGate.shape.last)
            .prefix(tokens, layer.inner)
          val up = w.mlpUp
            .view(tokens.toLong * w.mlpUp.shape.last)
            .prefix(tokens, layer.inner)
          ops.linears(w.modulated, Seq(layer.gate, layer.up), Seq(gate, up))
          ops.gated(Activation.Silu, gate, up, gate)
          ops.linear(gate, layer.down, w.projected)
        }
        block.mlp match {
          case Left(dense)    => swiGlu(dense)
          case Right(mixture) =>
            // the router: the timestep's part a row for every token
            ops.linear(time, mixture.routerTime, routerBias)
            ops.linear(w.normed, mixture.routerToken, w.logits)
            ops.addRow(w.logits, routerBias.view(c.experts), w.logits)
            val each = capacity(mixture, tokens)
            val (chosen, shares) = Nucleus.route(
              ops.toFloats(w.logits),
              tokens,
              c.experts,
              each,
              c.routeScale
            )
            val slots = c.experts.toLong * each
            val ids = ops.fromInts(Shape.of(slots), chosen)
            val gates = ops.fromFloats(Shape.of(slots), shares)
            perStep ++= Seq(ids, gates)
            def slotRows(t: Tensor, columns: Long) =
              t.view(t.shape.elementCount).prefix(slots, columns)
            val gathered = slotRows(w.gathered, width)
            val both = slotRows(w.slotBoth, 2 * mixture.inner)
            val (gate, up) =
              (
                slotRows(w.slotGate, mixture.inner),
                slotRows(w.slotUp, mixture.inner)
              )
            val slotOut = slotRows(w.slotOut, width)
            ops.embedding(w.modulated, ids, gathered)
            (0 until c.experts).foreach(e =>
              ops.linear(
                gathered.rows(e.toLong * each, each),
                mixture.gateUp(e),
                both.rows(e.toLong * each, each)
              )
            )
            ops.splitHalves(both, gate, up)
            ops.gated(Activation.Silu, gate, up, gate)
            (0 until c.experts).foreach(e =>
              ops.linear(
                gate.rows(e.toLong * each, each),
                mixture.down(e),
                slotOut.rows(e.toLong * each, each)
              )
            )
            swiGlu(mixture.shared)
            ops.scatterAddRows(slotOut, ids, gates, w.projected)
        }
        ops.gatedAdd(w.x, w.projected, part(b, 3))
      }

      // the final layer: (scale ; shift), and the sign turned
      ops.layerNorm(w.x, None, None, epsilon, w.normed)
      ops.modulate(
        w.normed,
        finalParts.view(2, width).rows(0, 1).view(width),
        finalParts.view(2, width).rows(1, 1).view(width),
        w.normed
      )
      ops.linear(w.normed, finalOut, out)
      ops.scale(out, -1f, out)
    } finally perStep.foreach(ops.release)
  }

  def close(): Unit = {
    small.foreach(ops.release)
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object Nucleus {

  private def bf16(value: Float): Float = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    java.lang.Float.intBitsToFloat(
      ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16
    )
  }

  /** The timestep's sinusoid as the released pipeline computes it in BF16: the
    * scheduler's timestep (`1000 σ`) and its thousandth each rounded, the
    * product with each frequency times 1000 in F32, its cosine (first half) and
    * sine (second) rounded.
    */
  def sinusoid(timestep: Float, channels: Int): Array[Float] = {
    val half = channels / 2
    val level = bf16(bf16(timestep * 1000f) / 1000f)
    Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(10000) * (i % half) / half).toFloat
      val angle = (level * frequency * 1000f).toDouble
      bf16((if (i < half) math.cos(angle) else math.sin(angle)).toFloat)
    }
  }

  /** Expert-choice routing: from the router's `logits` (`[tokens, experts]`),
    * each expert's `capacity` best tokens by its share of the token's softmax,
    * expert after expert, and each choice's weight: the share over the sum of
    * the shares of the experts that chose the token, times `scale`.
    */
  def route(
      logits: Array[Float],
      tokens: Int,
      experts: Int,
      capacity: Int,
      scale: Float
  ): (Array[Int], Array[Float]) = {
    val shares = new Array[Float](tokens * experts)
    IntStream
      .range(0, tokens)
      .parallel()
      .forEach { t =>
        val at = t * experts
        var largest = Float.NegativeInfinity
        var e = 0
        while (e < experts) {
          largest = math.max(largest, logits(at + e))
          e += 1
        }
        var sum = 0.0
        e = 0
        while (e < experts) {
          sum += math.exp((logits(at + e) - largest).toDouble)
          e += 1
        }
        e = 0
        while (e < experts) {
          shares(at + e) =
            (math.exp((logits(at + e) - largest).toDouble) / sum).toFloat
          e += 1
        }
      }
    val chosen = new Array[Int](experts * capacity)
    IntStream
      .range(0, experts)
      .parallel()
      .forEach { e =>
        val column = Array.tabulate(tokens)(t => shares(t * experts + e))
        val sorted = column.clone()
        java.util.Arrays.sort(sorted)
        val threshold = sorted(tokens - capacity)
        var (count, t) = (0, 0)
        while (t < tokens) {
          if (column(t) > threshold) {
            chosen(e * capacity + count) = t
            count += 1
          }
          t += 1
        }
        t = 0
        while (count < capacity) {
          if (column(t) == threshold) {
            chosen(e * capacity + count) = t
            count += 1
          }
          t += 1
        }
      }
    val sums = new Array[Double](tokens)
    chosen.indices.foreach(slot =>
      sums(chosen(slot)) += shares(chosen(slot) * experts + slot / capacity)
    )
    val weights = Array.tabulate(chosen.length) { slot =>
      val token = chosen(slot)
      (shares(token * experts + slot / capacity) / (sums(
        token
      ) + 1e-12) * scale).toFloat
    }
    (chosen, weights)
  }

  def open(ops: Ops, path: Path): Nucleus = {
    val source = WeightSource.open(ops, path)
    try new Nucleus(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
