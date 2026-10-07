package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** The LLaDA2 text model's sizes, off its weights and the `config.json` beside
  * them when there is one (released: 2048 wide, 20 layers, 16 query heads over
  * 4 key-value heads of 128, half of each head rotated, 256 experts of 512 in 8
  * groups from the second layer on, 8 chosen per token among 4 groups).
  */
final case class Llada2Config(
    hidden: Int,
    layers: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    rotaryDimensions: Int,
    ropeTheta: Float,
    epsilon: Float,
    experts: Int,
    groups: Int,
    groupsChosen: Int,
    expertsChosen: Int,
    routeScale: Float
)

/** The LLaDA2 language model as LLaDA-Image's text encoder (inclusionAI's
  * `LLaDA2MoeModel`, `modeling_llada2uni_moe.py`): a bidirectional transformer
  * over embeddings (the prompt's tokens, then the QueryFormer's queries) to its
  * last hidden state.
  *   - A layer: RMS norm, attention, a residual; RMS norm, MLP, a residual.
  *     Attention from one fused projection (queries, then keys, then values),
  *     RMS norms on q and k per head, a rotary embedding on the first half of
  *     each head (halves paired), 16 query heads over 4. The prompt's tokens
  *     see the prompt alone, the queries everything.
  *   - The first layer's MLP is a dense SwiGLU; the others mix experts chosen
  *     token by token: sigmoid scores (plus a bias per expert for the choice
  *     alone), the best groups by the sum of their two best scores, the best
  *     experts within them, weighted by their scores' share times a scale; plus
  *     a shared expert. The choice is made on the host; each chosen expert runs
  *     its tokens' rows as a small GEMM.
  */
final class Llada2 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  private val prefix =
    if (source.has("model.language_model.norm.weight")) "model.language_model."
    else ""
  if (!source.has(s"${prefix}layers.0.attention.query_key_value.weight"))
    throw new FormatException(
      "no LLaDA2 text model (layers.0.attention.query_key_value.weight)"
    )
  // what the weights do not say: the `config.json` beside them, else the
  // released model's
  private val configured = source.config.filter(_.has("num_experts_per_tok"))
  private def setting(key: String, released: Double): Double =
    configured.fold(released)(_.double(key))

  val config: Llada2Config = {
    def dimensions(name: String) =
      source(prefix + name).shape.dimensions.map(_.toInt)
    val headDimension = setting("head_dim", 128).toInt
    Llada2Config(
      hidden = dimensions("norm.weight").head,
      layers = Iterator
        .from(0)
        .takeWhile(i =>
          source.has(s"${prefix}layers.$i.input_layernorm.weight")
        )
        .size,
      heads = setting("num_attention_heads", 16).toInt,
      kvHeads = setting("num_key_value_heads", 4).toInt,
      headDimension = headDimension,
      rotaryDimensions =
        (headDimension * setting("partial_rotary_factor", 0.5)).toInt,
      ropeTheta = setting("rope_theta", 600000).toFloat,
      epsilon = setting("rms_norm_eps", 1e-6).toFloat,
      experts = setting("num_experts", 256).toInt,
      groups = setting("n_group", 8).toInt,
      groupsChosen = setting("topk_group", 4).toInt,
      expertsChosen = setting("num_experts_per_tok", 8).toInt,
      routeScale = setting("routed_scaling_factor", 2.5).toFloat
    )
  }
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = false)
  private val width = c.hidden.toLong
  private val (queryWidth, kvWidth) =
    (c.heads.toLong * c.headDimension, c.kvHeads.toLong * c.headDimension)

  private def floats(name: String, count: Long): Tensor =
    weights.floats(prefix + name, prefix + name, Shape.of(count))
  private def matrix(name: String): Tensor = source(prefix + name)

  final private class SwiGlu(at: String) {
    val gate: Tensor = matrix(s"$at.gate_proj.weight")
    val up: Tensor = matrix(s"$at.up_proj.weight")
    val down: Tensor = matrix(s"$at.down_proj.weight")
    val inner: Long = gate.shape.dimensions.head
  }

  final private class Mixture(at: String) {
    val router: Tensor = matrix(s"$at.gate.weight")
    val bias: Array[Float] = weights.hostFloats(s"$prefix$at.gate.expert_bias")
    private val gates = matrix(s"$at.experts.gate_proj")
    private val ups = matrix(s"$at.experts.up_proj")
    private val downs = matrix(s"$at.experts.down_proj")
    val inner: Long = gates.shape.dimensions(1)
    private def of(all: Tensor, expert: Int, rows: Long, columns: Long) =
      all.view(c.experts, rows * columns).rows(expert, 1).view(rows, columns)
    def gate(expert: Int): Tensor = of(gates, expert, inner, width)
    def up(expert: Int): Tensor = of(ups, expert, inner, width)
    def down(expert: Int): Tensor = of(downs, expert, width, inner)
    val shared = new SwiGlu(s"$at.shared_experts")
  }

  final private class Layer(index: Int) {
    private val at = s"layers.$index"
    private val fused = matrix(s"$at.attention.query_key_value.weight")
    val query: Tensor = fused.rows(0, queryWidth)
    val key: Tensor = fused.rows(queryWidth, kvWidth)
    val value: Tensor = fused.rows(queryWidth + kvWidth, kvWidth)
    val queryNorm: Tensor =
      floats(s"$at.attention.query_layernorm.weight", c.headDimension)
    val keyNorm: Tensor =
      floats(s"$at.attention.key_layernorm.weight", c.headDimension)
    val output: Tensor = matrix(s"$at.attention.dense.weight")
    val inputNorm: Tensor = floats(s"$at.input_layernorm.weight", width)
    val mlpNorm: Tensor = floats(s"$at.post_attention_layernorm.weight", width)
    val mlp: Either[SwiGlu, Mixture] =
      if (source.has(s"$prefix$at.mlp.experts.gate_proj"))
        Right(new Mixture(s"$at.mlp"))
      else Left(new SwiGlu(s"$at.mlp"))
  }

  private val layers = (0 until c.layers).map(new Layer(_))
  private val finalNorm = floats("norm.weight", width)
  private val embeddings = matrix("word_embeddings.weight")
  private val rope = Rope(
    c.ropeTheta,
    c.rotaryDimensions,
    RopeLayout.Neox,
    RopeSections.Single
  )
  private val attention =
    new PlainAttention(ops, c.heads, c.kvHeads, c.headDimension)

  /** The embeddings of `ids`, `[tokens, hidden]`. */
  def embed(ids: Array[Int]): Tensor = {
    val tokens = ops.fromInts(Shape.of(ids.length), ids)
    try {
      val out = ops.allocate(DType.F32, Shape.of(ids.length, width))
      ops.embedding(embeddings, tokens, out)
      out
    } finally ops.release(tokens)
  }

  /** The last hidden state of `embeds` (`[tokens, hidden]`, positions 0 on):
    * the first `text` tokens see one another alone, the rest see everything.
    */
  def encode(embeds: Tensor, text: Int): Tensor = {
    val tokens = embeds.shape.dimensions.head.toInt
    val (heads, kvHeads, d) =
      (c.heads.toLong, c.kvHeads.toLong, c.headDimension.toLong)
    val held = mutable.ArrayBuffer.empty[Tensor]
    def rows(count: Long, columns: Long, dtype: DType = DType.F32) = {
      val tensor = ops.allocate(dtype, Shape.of(count, columns))
      held += tensor
      tensor
    }
    val widest = layers.map(_.mlp.fold(_.inner, _.shared.inner)).max
    val slots = tokens.toLong * c.expertsChosen
    val expertInner =
      layers.flatMap(_.mlp.toOption).map(_.inner).maxOption.getOrElse(1L)
    val x = ops.allocate(DType.F32, embeds.shape)
    try {
      ops.copy(embeds, x)
      val normed = rows(tokens, width)
      val (q, rotatedQueries, attended) =
        (
          rows(tokens, queryWidth),
          rows(tokens, queryWidth),
          rows(tokens, queryWidth)
        )
      val (k, rotatedKeys, v) =
        (rows(tokens, kvWidth), rows(tokens, kvWidth), rows(tokens, kvWidth))
      val projected = rows(tokens, width)
      val (mlpGate, mlpUp) = (rows(tokens, widest), rows(tokens, widest))
      val logits = rows(tokens, math.max(1, c.experts))
      val (gathered, slotOut) = (rows(slots, width), rows(slots, width))
      val (slotGate, slotUp) =
        (rows(slots, expertInner), rows(slots, expertInner))
      val positions = {
        val tensor =
          ops.fromInts(Shape.of(tokens), Array.tabulate(tokens)(identity))
        held += tensor
        tensor
      }
      val sight =
        if (text >= tokens) Nil
        else Seq((0, text, text), (text, tokens - text, tokens))
      def swiGlu(layer: SwiGlu): Unit = {
        val gate = mlpGate.view(tokens * widest).prefix(tokens, layer.inner)
        val up = mlpUp.view(tokens * widest).prefix(tokens, layer.inner)
        ops.linears(normed, Seq(layer.gate, layer.up), Seq(gate, up))
        ops.gated(Activation.Silu, gate, up, gate)
        ops.linear(gate, layer.down, projected)
      }
      layers.foreach { layer =>
        ops.rmsNorm(x, layer.inputNorm, c.epsilon, 0f, normed)
        ops.linears(
          normed,
          Seq(layer.query, layer.key, layer.value),
          Seq(q, k, v)
        )
        ops.rmsNorm(
          q.view(tokens * heads, d),
          layer.queryNorm,
          c.epsilon,
          0f,
          q.view(tokens * heads, d)
        )
        ops.rmsNorm(
          k.view(tokens * kvHeads, d),
          layer.keyNorm,
          c.epsilon,
          0f,
          k.view(tokens * kvHeads, d)
        )
        ops.rope(
          q.view(tokens, heads, d),
          positions,
          rope,
          rotatedQueries.view(tokens, heads, d)
        )
        ops.rope(
          k.view(tokens, kvHeads, d),
          positions,
          rope,
          rotatedKeys.view(tokens, kvHeads, d)
        )
        attention(rotatedQueries, rotatedKeys, v, attended, sight)
        ops.linear(attended, layer.output, projected)
        ops.add(x, projected, x)

        ops.rmsNorm(x, layer.mlpNorm, c.epsilon, 0f, normed)
        layer.mlp match {
          case Left(dense)    => swiGlu(dense)
          case Right(mixture) =>
            ops.linear(normed, mixture.router, logits)
            val routes = Llada2.route(ops.toFloats(logits), mixture.bias, c)
            val ids = ops.fromInts(Shape.of(slots), routes.tokens)
            val shares = ops.fromFloats(Shape.of(slots), routes.weights)
            try {
              ops.embedding(normed, ids, gathered)
              routes.runs.foreach { (expert, first, count) =>
                val gate = slotGate
                  .view(slots * expertInner)
                  .prefix(slots, mixture.inner)
                  .rows(first, count)
                val up = slotUp
                  .view(slots * expertInner)
                  .prefix(slots, mixture.inner)
                  .rows(first, count)
                ops.linears(
                  gathered.rows(first, count),
                  Seq(mixture.gate(expert), mixture.up(expert)),
                  Seq(gate, up)
                )
                ops.gated(Activation.Silu, gate, up, gate)
                ops.linear(
                  gate,
                  mixture.down(expert),
                  slotOut.rows(first, count)
                )
              }
              swiGlu(mixture.shared)
              ops.scatterAddRows(slotOut, ids, shares, projected)
            } finally {
              ops.release(ids)
              ops.release(shares)
            }
        }
        ops.add(x, projected, x)
      }
      ops.rmsNorm(x, finalNorm, c.epsilon, 0f, x)
      x
    } catch {
      case error: Throwable =>
        ops.release(x)
        throw error
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    attention.close()
    weights.release()
    source.close()
  }
}

object Llada2 {

  /** A layer's routing, expert after expert: each slot's token and weight, and
    * each chosen expert's run of slots `(expert, first, count)`.
    */
  final case class Routes(
      tokens: Array[Int],
      weights: Array[Float],
      runs: Seq[(Int, Int, Int)]
  )

  /** The experts of every token from the router's `logits` (`[tokens,
    * experts]`): sigmoid scores, plus `bias` for the choice alone; the
    * `groupsChosen` groups with the largest sum of their two best, then the
    * `expertsChosen` best experts within them, each weighted by its score over
    * the chosen ones' sum, times the scale.
    */
  def route(
      logits: Array[Float],
      bias: Array[Float],
      config: Llada2Config
  ): Routes = {
    val experts = config.experts
    val tokens = logits.length / experts
    val perGroup = experts / config.groups
    val chosen = (0 until tokens).map { t =>
      val scores =
        Array.tabulate(experts)(e =>
          1 / (1 + math.exp(-logits(t * experts + e).toDouble))
        )
      val routing = Array.tabulate(experts)(e => scores(e) + bias(e))
      val groups = (0 until config.groups)
        .sortBy { g =>
          val best =
            routing.slice(g * perGroup, (g + 1) * perGroup).sorted.takeRight(2)
          -best.sum
        }
        .take(config.groupsChosen)
        .toSet
      val picked = (0 until experts)
        .filter(e => groups(e / perGroup))
        .sortBy(e => -routing(e))
        .take(config.expertsChosen)
      val sum = picked.map(scores).sum + 1e-20
      picked.map(e => (e, t, (scores(e) / sum * config.routeScale).toFloat))
    }
    val slots = chosen.flatten.sortBy(_._1)
    val runs = slots.indices
      .groupBy(i => slots(i)._1)
      .toSeq
      .sortBy(_._1)
      .map((expert, indices) => (expert, indices.min, indices.size))
    Routes(slots.map(_._2).toArray, slots.map(_._3).toArray, runs)
  }

  def open(ops: Ops, path: Path): Llada2 = {
    val source = WeightSource.open(ops, path)
    try new Llada2(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
