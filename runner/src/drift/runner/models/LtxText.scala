package drift.runner.models

import drift.runner.diffusion.LoraUpdates
import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.util.stream.IntStream
import scala.collection.mutable

/** LTX 2.5's text features (diffusers' `LTX2TextConnectors`, its LTX 2.3+
  * branch): each of Gemma 4's hidden states RMS-normed per token, side by side
  * per token (hidden-major: value `h × states + s`, as diffusers flattens
  * them), scaled by √(width / 3840) and projected to the video's and the
  * audio's widths (`text_embedding_projection.*_aggregate_embed`, in the text
  * encoder's file). The projections' columns are regrouped state-major at load,
  * so the states go in as they come.
  */
final class LtxTextFeatures(
    ops: Ops,
    source: WeightSource,
    states: Int,
    hidden: Int
) {

  private val weights = new HybridWeights(ops, source, gguf = false)

  final case class Projection(weight: Tensor, bias: Tensor, scale: Float) {
    def width: Long = bias.shape.elementCount
  }

  private def projection(name: String): Projection = {
    val weight = s"$name.weight"
    val Seq(width, inputs) = source.shape(weight).dimensions.map(_.toInt)
    require(
      inputs == states * hidden,
      s"$name takes $inputs, not $states × $hidden"
    )
    val plain =
      Set(DType.F32, DType.F16, DType.BF16).contains(source.storedAs(weight))
    Projection(
      if (plain) regroupedBytes(weight, width, inputs)
      else regroupedFloats(weight, width, inputs),
      weights.floats(s"$name.bias", s"$name.bias", Shape.of(width)),
      math.sqrt(width.toDouble / hidden).toFloat
    )
  }

  /** `move(from, to)` for every value of `rows` rows: the one at `h × states +
    * s` of its row goes to `s × hidden + h`. A row a task.
    */
  private def regroup(rows: Int, inputs: Int)(
      move: (Int, Int) => Unit
  ): Unit =
    IntStream.range(0, rows).parallel().forEach { row =>
      val base = row * inputs
      var h = 0
      while (h < hidden) {
        var s = 0
        while (s < states) {
          move(base + h * states + s, base + s * hidden + h)
          s += 1
        }
        h += 1
      }
    }

  /** A projection stored in a plain type, regrouped as it is stored. */
  private def regroupedBytes(
      weight: String,
      width: Int,
      inputs: Int
  ): Tensor = {
    val dtype = source.storedAs(weight)
    val element = dtype.byteSize(1).toInt
    val bytes = weights.hostBytes(weight)
    val regrouped = new Array[Byte](bytes.length)
    // a row's values are counted from the file's start: Long offsets
    IntStream.range(0, width).parallel().forEach { row =>
      val base = row.toLong * inputs * element
      var h = 0
      while (h < hidden) {
        var s = 0
        while (s < states) {
          val from = (base + (h.toLong * states + s) * element).toInt
          val to = (base + (s.toLong * hidden + h) * element).toInt
          var byte = 0
          while (byte < element) {
            regrouped(to + byte) = bytes(from + byte)
            byte += 1
          }
          s += 1
        }
        h += 1
      }
    }
    weights.upload(dtype, Shape.of(width, inputs), regrouped)
  }

  /** A projection stored quantized (fp8, ComfyUI's int8): decoded on the host a
    * few million values at a time, regrouped, and kept in BF16.
    */
  private def regroupedFloats(
      weight: String,
      width: Int,
      inputs: Int
  ): Tensor = {
    val out = weights.keep(ops.allocate(DType.BF16, Shape.of(width, inputs)))
    val chunk = math.max(1, (1 << 24) / inputs)
    var row = 0
    while (row < width) {
      val count = math.min(chunk, width - row)
      val values = source.hostRows(weight, row, count)
      val regrouped = new Array[Float](values.length)
      regroup(count, inputs)((from, to) => regrouped(to) = values(from))
      val part = ops.fromFloats(Shape.of(count, inputs), regrouped)
      try ops.convert(part, out.rows(row, count))
      finally ops.release(part)
      row += count
    }
    out
  }

  val video: Projection = projection(
    "text_embedding_projection.video_aggregate_embed"
  )
  val audio: Projection = projection(
    "text_embedding_projection.audio_aggregate_embed"
  )

  /** `hiddenStates` (`[states × L, hidden]`, state-major, from `Gemma4Text`) as
    * the video's and the audio's features, `[L, width]` each, the caller's to
    * release.
    */
  def apply(hiddenStates: Tensor): (Tensor, Tensor) = {
    val tokens = hiddenStates.shape.dimensions.head / states
    val normed = ops.allocate(DType.F32, hiddenStates.shape)
    val side = ops.allocate(DType.F32, Shape.of(tokens, states.toLong * hidden))
    val ones = ops.fromFloats(Shape.of(hidden), Array.fill(hidden)(1f))
    try {
      ops.rmsNorm(hiddenStates, ones, 1e-6f, 0f, normed)
      ops.concatColumns(
        (0 until states).map(s => normed.rows(s * tokens, tokens)),
        side
      )
      def through(p: Projection) = {
        val out = ops.allocate(DType.F32, Shape.of(tokens, p.width))
        ops.linear(side, p.weight, out)
        // (x × scale) · W + b
        ops.scale(out, p.scale, out)
        ops.addRow(out, p.bias, out)
        out
      }
      (through(video), through(audio))
    } finally Seq(normed, side, ones).foreach(ops.release)
  }

  def release(): Unit = weights.release()
}

/** LTX 2's text connectors (diffusers' `LTX2ConnectorTransformer1d`, the
  * transformer file's `*_embeddings_connector`): the prompt's features
  * front-aligned in a sequence of `Length` rows, the rest filled by the
  * learnable registers (row `p` takes register `p mod count`); blocks of gated
  * self-attention (each head's output × 2σ(logit)) with RMS norms across the
  * heads and the split 1-D RoPE over `p / 4096`, and a GELU (tanh) MLP,
  * pre-norm without weights; a final norm. The features' order is the stream's
  * own: video 32 heads of 128, audio 32 of 64.
  */
final class LtxConnector(
    ops: Ops,
    source: WeightSource,
    prefix: String,
    updates: Option[LoraUpdates]
) {

  private def dimensions(name: String) =
    source.shape(s"$prefix.$name").dimensions.map(_.toInt)

  private val Seq(registerCount, width) = dimensions("learnable_registers")
  private val heads = dimensions(
    "transformer_1d_blocks.0.attn1.to_gate_logits.weight"
  ).head
  private val head = width / heads
  private val intermediate = dimensions(
    "transformer_1d_blocks.0.ff.net.0.proj.weight"
  ).head
  private val Epsilon = 1e-6f
  private val weights = new HybridWeights(ops, source, gguf = false)

  private def floats(name: String, count: Long) =
    weights.floats(s"$prefix.$name", s"$prefix.$name", Shape.of(count))

  /** A linear layer and its site: its name in the file, a LoRA's target. */
  final private case class Affine(site: String, weight: Tensor, bias: Tensor)
  private val affines = mutable.ArrayBuffer.empty[Affine]
  private def affine(name: String, outputs: Long) = {
    val made = Affine(
      s"$prefix.$name",
      source.linear(s"$prefix.$name.weight"),
      floats(s"$name.bias", outputs)
    )
    affines += made
    made
  }

  final private class Block(i: Int) {
    private val at = s"transformer_1d_blocks.$i"
    val q: Affine = affine(s"$at.attn1.to_q", width)
    val k: Affine = affine(s"$at.attn1.to_k", width)
    val v: Affine = affine(s"$at.attn1.to_v", width)
    val o: Affine = affine(s"$at.attn1.to_out.0", width)
    val gate: Affine = affine(s"$at.attn1.to_gate_logits", heads)
    val qNorm: Tensor = floats(s"$at.attn1.q_norm.weight", width)
    val kNorm: Tensor = floats(s"$at.attn1.k_norm.weight", width)
    val up: Affine = affine(s"$at.ff.net.0.proj", intermediate)
    val down: Affine = affine(s"$at.ff.net.2", width)
  }

  private val blocks = Iterator
    .from(0)
    .takeWhile(i =>
      source.has(s"$prefix.transformer_1d_blocks.$i.attn1.to_q.weight")
    )
    .map(new Block(_))
    .toSeq
  private val registers = weights.floats(
    s"$prefix.learnable_registers",
    s"$prefix.learnable_registers",
    Shape.of(registerCount, width)
  )
  private val ones =
    weights.keep(ops.fromFloats(Shape.of(width), Array.fill(width)(1f)))
  private val attention =
    Attention((1 / math.sqrt(head)).toFloat, causal = false, None, None, None)

  /** Every linear's weight by its site. */
  def sites: Map[String, Tensor] =
    affines.map(layer => layer.site -> layer.weight).toMap

  private def run(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    updates.foreach(_(x, layer.site, out))
    ops.addRow(out, layer.bias, out)
  }

  /** The split 1-D RoPE's per-head tables over `rows` positions. */
  private def tables(rows: Int): (Tensor, Tensor) = {
    val pairs = width / 2
    val angles = Array.tabulate(rows * pairs) { i =>
      val (p, j) = (i / pairs, i % pairs)
      val frequency = math.pow(10000.0, j.toDouble / (pairs - 1)) * math.Pi / 2
      (2.0 * p / 4096 - 1) * frequency
    }
    (
      ops.fromFloats(
        Shape.of(rows, heads, head / 2),
        angles.map(a => math.cos(a).toFloat)
      ),
      ops.fromFloats(
        Shape.of(rows, heads, head / 2),
        angles.map(a => math.sin(a).toFloat)
      )
    )
  }

  /** `features` (`[L, width]`) through the connector over `rows` positions:
    * `[rows, width]`, the caller's to release.
    */
  def apply(features: Tensor, rows: Int): Tensor = {
    val tokens = features.shape.dimensions.head.toInt
    require(
      tokens <= rows && rows % registerCount == 0,
      s"$tokens tokens over $rows rows"
    )
    val n = rows.toLong
    val held = mutable.ArrayBuffer.empty[Tensor]
    def allocate(columns: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(n, columns))
      held += tensor
      tensor
    }
    val x = ops.allocate(DType.F32, Shape.of(n, width.toLong))
    try {
      ops.copy(features, x.rows(0, tokens))
      (tokens until rows).foreach(p =>
        ops.copy(registers.rows(p % registerCount, 1), x.rows(p, 1))
      )
      val (cos, sin) = tables(rows)
      held ++= Seq(cos, sin)
      val (normed, q, k, v, rq, rk, attended, gated, projected) = (
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width),
        allocate(width)
      )
      val (logits, up) = (allocate(heads), allocate(intermediate))
      val pageTable = ops.fromInts(Shape.of(1), Array(0))
      held += pageTable
      val cache: KvCache =
        ops.allocateCache(1, (rows + 15) / 16 * 16, heads, head)
      held ++= Seq(cache.keys, cache.values)
      val (h, d) = (heads.toLong, head.toLong)
      blocks.foreach { b =>
        ops.rmsNorm(x, ones, Epsilon, 0f, normed)
        run(normed, b.gate, logits)
        run(normed, b.q, q)
        run(normed, b.k, k)
        run(normed, b.v, v)
        ops.rmsNorm(q, b.qNorm, Epsilon, 0f, q)
        ops.rmsNorm(k, b.kNorm, Epsilon, 0f, k)
        ops.ropeTable(
          q.view(n, h, d),
          cos,
          sin,
          rq.view(n, h, d),
          halves = true
        )
        ops.ropeTable(
          k.view(n, h, d),
          cos,
          sin,
          rk.view(n, h, d),
          halves = true
        )
        ops.cacheWrite(rk.view(n, h, d), v.view(n, h, d), cache, pageTable, 0)
        ops.attention(
          rq.view(n, h, d),
          cache,
          pageTable,
          0,
          rows,
          attention,
          attended.view(n, h, d)
        )
        // each head's output × 2σ(its logit)
        ops.activation(Activation.Sigmoid, logits, logits)
        ops.scale(logits, 2f, logits)
        ops.zero(gated)
        ops.rowGatedAdd(
          gated.view(n * h, d),
          attended.view(n * h, d),
          logits.view(n * h)
        )
        run(gated, b.o, projected)
        ops.add(x, projected, x)
        ops.rmsNorm(x, ones, Epsilon, 0f, normed)
        run(normed, b.up, up)
        ops.activation(Activation.GeluTanh, up, up)
        run(up, b.down, projected)
        ops.add(x, projected, x)
      }
      ops.rmsNorm(x, ones, Epsilon, 0f, x)
      x
    } catch {
      case error: Throwable =>
        ops.release(x)
        throw error
    } finally held.foreach(ops.release)
  }

  def release(): Unit = weights.release()
}

object LtxConnector {

  /** The connector at `prefix`; its linears take `updates` when given (the
    * transformer's LoRAs).
    */
  def apply(
      ops: Ops,
      source: WeightSource,
      prefix: String,
      updates: Option[LoraUpdates] = None
  ): LtxConnector = {
    if (!source.has(s"$prefix.learnable_registers"))
      throw new FormatException(s"no LTX 2 text connector at $prefix")
    new LtxConnector(ops, source, prefix, updates)
  }
}
