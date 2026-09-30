package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.tensor.*

import scala.collection.mutable

/** Gemma 4's text model as a text encoder (transformers'
  * `Gemma4UnifiedTextModel`; LTX 2.5's, the 12B), from transformers' names
  * under `prefix`, its shape read off the weights:
  *   - sliding layers (heads of `head`, their own key and value heads, RoPE θ
  *     10⁴ on the whole head) and global ones (heads twice as wide over one key
  *     head that is also the value, `attention_k_eq_v`; the proportional RoPE,
  *     θ 10⁶ on the first quarter of each head's pairs);
  *   - RMS norms of plain weights, on the input, the attention's output, the
  *     MLP's input and output; RMS norms on each query and key head (weighted)
  *     and each value head (not); attention unscaled; a gated GELU (tanh) MLP;
  *     each layer's output times its `layer_scalar`; embeddings × √hidden (in
  *     the table's type).
  * `encode` gives every hidden state (the embeddings, then each layer's output,
  * the last one through the final norm, as transformers' `hidden_states`) of a
  * prompt's tokens, causal; a prompt shorter than the sliding window (1024)
  * attends to all of itself, as the attention here does.
  */
final class Gemma4Text private (ops: Ops, source: WeightSource, prefix: String)
    extends AutoCloseable {

  private def dimensions(name: String) =
    source.shape(s"$prefix$name").dimensions.map(_.toInt)

  val hidden: Int = dimensions("embed_tokens.weight").last
  val layers: Int = Iterator
    .from(0)
    .takeWhile(i => source.has(s"${prefix}layers.$i.input_layernorm.weight"))
    .size
  private val intermediate = dimensions("layers.0.mlp.gate_proj.weight").head
  private val Epsilon = 1e-6f

  private val weights = new HybridWeights(ops, source, gguf = false)
  private def floats(name: String, count: Long) =
    weights.floats(s"$prefix$name", s"$prefix$name", Shape.of(count))

  final private class Layer(i: Int) {
    private def at(part: String) = s"layers.$i.$part"
    val head: Int = dimensions(at("self_attn.q_norm.weight")).head
    val heads: Int = dimensions(at("self_attn.q_proj.weight")).head / head
    val kvHeads: Int = dimensions(at("self_attn.k_proj.weight")).head / head

    /** A global layer: its values are its keys, before their norm. */
    val keysAreValues: Boolean =
      !source.has(s"$prefix${at("self_attn.v_proj.weight")}")
    val inputNorm: Tensor = floats(at("input_layernorm.weight"), hidden)
    val q: Tensor = source(s"$prefix${at("self_attn.q_proj.weight")}")
    val k: Tensor = source(s"$prefix${at("self_attn.k_proj.weight")}")
    val v: Option[Tensor] =
      Option.unless(keysAreValues)(
        source(s"$prefix${at("self_attn.v_proj.weight")}")
      )
    val qNorm: Tensor = floats(at("self_attn.q_norm.weight"), head)
    val kNorm: Tensor = floats(at("self_attn.k_norm.weight"), head)
    val o: Tensor = source(s"$prefix${at("self_attn.o_proj.weight")}")
    val postAttentionNorm: Tensor =
      floats(at("post_attention_layernorm.weight"), hidden)
    val preMlpNorm: Tensor =
      floats(at("pre_feedforward_layernorm.weight"), hidden)
    val postMlpNorm: Tensor =
      floats(at("post_feedforward_layernorm.weight"), hidden)
    val gate: Tensor = source(s"$prefix${at("mlp.gate_proj.weight")}")
    val up: Tensor = source(s"$prefix${at("mlp.up_proj.weight")}")
    val down: Tensor = source(s"$prefix${at("mlp.down_proj.weight")}")
    val scalar: Float = weights.hostFloats(s"$prefix${at("layer_scalar")}").head
  }

  private val embedding = source(s"${prefix}embed_tokens.weight")
  private val stack = (0 until layers).map(new Layer(_))
  private val finalNorm = floats("norm.weight", hidden)

  /** The narrowest layer's heads define "sliding": the global ones are wider.
    */
  private val slidingHead = stack.map(_.head).min

  /** √hidden, rounded to the embedding table's type as transformers does. */
  private val embedScale: Float = {
    val scale = math.sqrt(hidden.toDouble).toFloat
    if (embedding.dtype == DType.BF16) {
      val bits = java.lang.Float.floatToRawIntBits(scale)
      java.lang.Float.intBitsToFloat(
        ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16
      )
    } else scale
  }

  /** RoPE's cosines and sines (`[L, head / 2]`) of a layer's kind: the sliding
    * layers' default RoPE, the global layers' proportional one (a quarter of
    * the pairs turning, θ 10⁶ over the whole head width).
    */
  private def rope(
      tokens: Int,
      head: Int,
      global: Boolean
  ): (Tensor, Tensor) = {
    val pairs = head / 2
    val (theta, turning) =
      if (global) (1000000.0, pairs / 4) else (10000.0, pairs)
    val angles = Array.tabulate(tokens * pairs) { i =>
      val (t, m) = (i / pairs, i % pairs)
      if (m < turning) t / math.pow(theta, 2.0 * m / head) else 0.0
    }
    (
      ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.cos(a).toFloat)
      ),
      ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.sin(a).toFloat)
      )
    )
  }

  /** Every hidden state of `ids`: `[(layers + 1) × L, hidden]`, state-major,
    * the caller's to release.
    */
  def encode(ids: Array[Int]): Tensor = {
    val tokens = ids.length
    val l = tokens.toLong
    val out = ops.allocate(DType.F32, Shape.of((layers + 1) * l, hidden.toLong))
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    def allocate(columns: Long) = keep(
      ops.allocate(DType.F32, Shape.of(l, columns))
    )
    try {
      val idTensor = keep(ops.fromInts(Shape.of(l), ids))
      val x = allocate(hidden)
      ops.embedding(embedding, idTensor, x)
      ops.scale(x, embedScale, x)
      ops.copy(x, out.rows(0, l))
      val widest = stack.map(layer => layer.heads.toLong * layer.head).max
      val (normed, projected) = (allocate(hidden), allocate(hidden))
      val (q, k, v, rotated, attended) =
        (
          allocate(widest),
          allocate(widest),
          allocate(widest),
          allocate(widest),
          allocate(widest)
        )
      val rotatedKeys = allocate(widest)
      val (gate, up) = (allocate(intermediate), allocate(intermediate))
      // the causal mask as an additive bias, per head count
      val masks = mutable.Map.empty[Int, Tensor]
      def mask(heads: Int) = masks.getOrElseUpdate(
        heads,
        keep(
          ops.fromFloats(
            Shape.of(heads, l, l),
            Array.tabulate(heads * tokens * tokens) { i =>
              val pair = i % (tokens * tokens)
              if (pair % tokens > pair / tokens) -1e30f else 0f
            }
          )
        )
      )
      val tables = mutable.Map.empty[Boolean, (Tensor, Tensor)]
      stack.zipWithIndex.foreach { (layer, i) =>
        val global = layer.head > slidingHead
        val (cos, sin) = tables.getOrElseUpdate(
          global, {
            val (c, s) = rope(tokens, layer.head, global)
            keep(c)
            keep(s)
            (c, s)
          }
        )
        val (h, kv, d) =
          (layer.heads.toLong, layer.kvHeads.toLong, layer.head.toLong)
        ops.rmsNorm(x, layer.inputNorm, Epsilon, 0f, normed)
        val (queries, keys, values) =
          (q.prefix(l, h * d), k.prefix(l, kv * d), v.prefix(l, kv * d))
        ops.linears(
          normed,
          Seq(layer.q, layer.k) ++ layer.v,
          Seq(queries, keys) ++ layer.v.map(_ => values)
        )
        if (layer.keysAreValues) ops.copy(keys, values)
        ops.rmsNorm(
          queries.view(l * h, d),
          layer.qNorm,
          Epsilon,
          0f,
          queries.view(l * h, d)
        )
        ops.rmsNorm(
          keys.view(l * kv, d),
          layer.kNorm,
          Epsilon,
          0f,
          keys.view(l * kv, d)
        )
        ops.rmsNorm(
          values.view(l * kv, d),
          ones(layer.head),
          Epsilon,
          0f,
          values.view(l * kv, d)
        )
        val (rq, rk) = (rotated.prefix(l, h, d), rotatedKeys.prefix(l, kv, d))
        ops.ropeTable(queries.view(l, h, d), cos, sin, rq, halves = true)
        ops.ropeTable(keys.view(l, kv, d), cos, sin, rk, halves = true)
        val att = attended.prefix(l, h * d)
        ops.shortAttention(
          rq.view(l, 1, h, d),
          rk.view(l, 1, kv, d),
          values.view(l, 1, kv, d),
          1f,
          att.view(l, 1, h, d),
          Some(mask(layer.heads))
        )
        ops.linear(att, layer.o, projected)
        ops.rmsNorm(projected, layer.postAttentionNorm, Epsilon, 0f, projected)
        ops.add(x, projected, x)
        ops.rmsNorm(x, layer.preMlpNorm, Epsilon, 0f, normed)
        ops.linears(normed, Seq(layer.gate, layer.up), Seq(gate, up))
        ops.gated(Activation.GeluTanh, gate, up, gate)
        ops.linear(gate, layer.down, projected)
        ops.rmsNorm(projected, layer.postMlpNorm, Epsilon, 0f, projected)
        ops.add(x, projected, x)
        if (layer.scalar != 1f) ops.scale(x, layer.scalar, x)
        val state = out.rows((i + 1) * l, l)
        if (i == layers - 1) ops.rmsNorm(x, finalNorm, Epsilon, 0f, state)
        else ops.copy(x, state)
      }
      out
    } catch {
      case error: Throwable =>
        ops.release(out)
        throw error
    } finally held.foreach(ops.release)
  }

  private val onesByWidth = mutable.Map.empty[Int, Tensor]
  private def ones(width: Int): Tensor =
    onesByWidth.getOrElseUpdate(
      width,
      weights.keep(ops.fromFloats(Shape.of(width), Array.fill(width)(1f)))
    )

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object Gemma4Text {

  /** Gemma 4's text model in `source` under `prefix` (`model.` in a text
    * encoder's single file); the source is the model's to close.
    */
  def apply(ops: Ops, source: WeightSource, prefix: String): Gemma4Text = {
    if (!source.has(s"${prefix}layers.0.layer_scalar"))
      throw new FormatException(
        s"no Gemma 4 text model under $prefix (no layer_scalar)"
      )
    new Gemma4Text(ops, source, prefix)
  }
}
