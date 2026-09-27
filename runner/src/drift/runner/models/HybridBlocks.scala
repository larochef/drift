package drift.runner.models

import drift.runner.formats.GgufFile
import drift.runner.ops.*
import drift.runner.state.{KvCache, Sequence}
import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_BYTE
import scala.collection.mutable

/** A hybrid layer's feed-forward: routed experts or one dense SwiGLU. */
sealed trait FeedForward

/** `used` of `experts` routed experts of `intermediate`, and a gated shared
  * expert of `sharedIntermediate`.
  */
final case class RoutedExperts(
    experts: Int,
    used: Int,
    intermediate: Int,
    sharedIntermediate: Int
) extends FeedForward

/** One SwiGLU of `intermediate` (the dense Qwen 3.5 and 3.8, 27B). */
final case class DenseFeedForward(intermediate: Int) extends FeedForward

/** What the Qwen hybrids' blocks share (Qwen 3.5/3.6/3.8, dense or MoE, and
  * Qwen 3.8 Flash Next): gated-DeltaNet linear attention, gated full attention
  * with partial RoPE, and a routed mixture of experts with a gated shared
  * expert or a dense SwiGLU.
  */
final case class HybridShape(
    hidden: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    rotaryDimensions: Int,
    ropeTheta: Float,
    /** Plain RoPE, or mRoPE's interleaved sections (Qwen 3.5 and on). */
    ropeSections: RopeSections,
    rmsEpsilon: Float,
    deltaRule: DeltaRule,
    convTaps: Int,
    feedForward: FeedForward,
    /** 0 for a GGUF (the converter folds Qwen's `1 + w` into its norms), 1 for
      * transformers' weights.
      */
    normOffset: Float,
    /** What gates the DeltaNet's output norm: SiLU in 3.5/3.6, sigmoid in 3.8.
      */
    deltaGate: Activation
) {
  def convChannels: Int = deltaRule.qkvWidth
  def valueWidth: Int = deltaRule.valueHeads * deltaRule.dimension

  /** The routed experts, for the blocks that only take those. */
  def routed: RoutedExperts = feedForward match {
    case experts: RoutedExperts => experts
    case DenseFeedForward(_)    =>
      throw new IllegalStateException("a dense feed-forward has no experts")
  }

  /** A linear-attention layer's states: the convolution's last inputs and the
    * delta rule's matrix.
    */
  def recurrentShapes: Seq[Shape] = Seq(
    Shape.of(convTaps - 1, convChannels),
    Shape.of(deltaRule.valueHeads, deltaRule.dimension, deltaRule.dimension)
  )
}

object HybridShape {

  /** mRoPE's sections from a GGUF's `rope.dimension_sections` (llama.cpp writes
    * Qwen's interleaved ones as four, the last 0); plain RoPE without them.
    */
  def ggufRopeSections(file: GgufFile, architecture: String): RopeSections = {
    val key = s"$architecture.rope.dimension_sections"
    if (!file.metadata.contains(key)) RopeSections.Single
    else {
      val sections = file.longs(key).map(_.toInt)
      RopeSections.Interleaved(sections(0), sections(1), sections(2))
    }
  }
}

/** A model's tensors by their GGUF or transformers name, and the tensors made
  * at load (fused, stacked or converted weights), released at close.
  */
final class HybridWeights(
    ops: Ops,
    val source: WeightSource,
    val gguf: Boolean
) {
  private val made = mutable.ArrayBuffer.empty[Tensor]

  def apply(ggufName: String, huggingFaceName: String): Tensor =
    source(if (gguf) ggufName else huggingFaceName)

  def has(ggufName: String, huggingFaceName: String): Boolean =
    source.has(if (gguf) ggufName else huggingFaceName)

  /** The name these weights use. */
  def name(ggufName: String, huggingFaceName: String): String =
    if (gguf) ggufName else huggingFaceName

  /** A whole tensor decoded to floats on the host, for weights transformed at
    * load.
    */
  def hostFloats(name: String): Array[Float] = {
    val tensor = source(name)
    tensor.dtype.decode(
      MemorySegment.ofArray(hostBytes(name)),
      tensor.shape.elementCount.toInt
    )
  }

  /** A tensor's bytes on the host, for weights transformed at load. */
  def hostBytes(name: String): Array[Byte] =
    source.bytes(name).toArray(JAVA_BYTE)

  def upload(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor =
    keep(ops.fromBytes(dtype, shape, bytes))

  def keep(tensor: Tensor): Tensor = {
    made += tensor
    tensor
  }

  /** A tensor as F32 of `shape`, decoded at load when stored otherwise. */
  def floats(
      ggufName: String,
      huggingFaceName: String,
      shape: Shape
  ): Tensor = {
    val name = if (gguf) ggufName else huggingFaceName
    val stored = source(name)
    if (stored.dtype == DType.F32) stored.view(shape.dimensions*)
    else
      keep(
        ops.fromFloats(
          shape,
          stored.dtype.decode(
            MemorySegment.ofArray(hostBytes(name)),
            shape.elementCount.toInt
          )
        )
      )
  }

  /** Experts' weights as one `[experts, N, K]`: a GGUF holds them so;
    * transformers holds them fused (every expert's gate and up in one
    * `[experts, 2 × intermediate, hidden]`, split here) or `save_pretrained`
    * writes one tensor per expert, stacked here.
    */
  def stackedExperts(
      g: String,
      h: String,
      ggufName: String,
      projection: String,
      experts: Int
  ): Tensor =
    if (gguf) source(g + ggufName)
    else if (source.has(h + "experts.gate_up_proj")) {
      if (projection == "down_proj") source(h + "experts.down_proj")
      else {
        val name = h + "experts.gate_up_proj"
        val pair = source(name)
        val Seq(count, rows, cols) = pair.shape.dimensions
        val half = pair.dtype.byteSize(rows / 2 * cols).toInt
        val first = if (projection == "gate_proj") 0 else 1
        val bytes = hostBytes(name)
        upload(
          pair.dtype,
          Shape.of(count, rows / 2, cols),
          (0 until count.toInt)
            .flatMap(e =>
              bytes.slice((2 * e + first) * half, (2 * e + first + 1) * half)
            )
            .toArray
        )
      }
    } else
      stacked((0 until experts).map(e => s"${h}experts.$e.$projection.weight"))

  /** Tensors of one type stacked along a new first dimension, or joined along
    * their first when `join`.
    */
  def stacked(names: Seq[String], join: Boolean = false): Tensor = {
    val first = source(names.head)
    val shape =
      if (join)
        Shape(
          names.map(source(_).shape.dimensions.head).sum +:
            first.shape.dimensions.tail
        )
      else Shape(names.size.toLong +: first.shape.dimensions)
    upload(first.dtype, shape, names.flatMap(hostBytes).toArray)
  }

  def release(): Unit = made.foreach(ops.release)
}

/** Tensors named `g…` in a GGUF, `h…` in transformers' checkpoints. */
final class LinearAttentionWeights(
    ops: Ops,
    weights: HybridWeights,
    shape: HybridShape,
    g: String,
    h: String
) {
  private val rule = shape.deltaRule
  val qkv: Tensor = weights(g + "attn_qkv.weight", h + "in_proj_qkv.weight")
  val z: Tensor = weights(g + "attn_gate.weight", h + "in_proj_z.weight")
  val a: Tensor = weights(g + "ssm_alpha.weight", h + "in_proj_a.weight")
  val b: Tensor = weights(g + "ssm_beta.weight", h + "in_proj_b.weight")
  val dtBias: Tensor = weights(g + "ssm_dt.bias", h + "dt_bias")

  /** `−exp(A_log)`: stored so in a GGUF, computed from transformers'. */
  val decay: Tensor =
    if (weights.gguf) weights.source(g + "ssm_a")
    else {
      val logs = DType.F32.decode(
        MemorySegment.ofArray(weights.hostBytes(h + "A_log")),
        rule.valueHeads
      )
      weights.keep(
        ops.fromFloats(
          Shape.of(rule.valueHeads),
          logs.map(l => -math.exp(l).toFloat)
        )
      )
    }
  val conv: Tensor = weights(g + "ssm_conv1d.weight", h + "conv1d.weight")
    .view(shape.convChannels, shape.convTaps)
  val norm: Tensor = weights(g + "ssm_norm.weight", h + "norm.weight")
  val out: Tensor = weights(g + "ssm_out.weight", h + "out_proj.weight")
}

final class FullAttentionWeights(weights: HybridWeights, g: String, h: String) {

  /** Each head's query then its output gate. */
  val q: Tensor = weights(g + "attn_q.weight", h + "q_proj.weight")
  val k: Tensor = weights(g + "attn_k.weight", h + "k_proj.weight")
  val v: Tensor = weights(g + "attn_v.weight", h + "v_proj.weight")
  val qNorm: Tensor = weights(g + "attn_q_norm.weight", h + "q_norm.weight")
  val kNorm: Tensor = weights(g + "attn_k_norm.weight", h + "k_norm.weight")
  val o: Tensor = weights(g + "attn_output.weight", h + "o_proj.weight")
}

/** A layer's feed-forward weights. */
sealed trait FeedForwardWeights

object FeedForwardWeights {

  /** The weights `shape`'s feed-forward takes. */
  def apply(
      weights: HybridWeights,
      shape: HybridShape,
      g: String,
      h: String
  ): FeedForwardWeights = shape.feedForward match {
    case _: RoutedExperts    => new ExpertWeights(weights, shape, g, h)
    case DenseFeedForward(_) => new DenseWeights(weights, g, h)
  }
}

final class DenseWeights(weights: HybridWeights, g: String, h: String)
    extends FeedForwardWeights {
  val gate: Tensor = weights(g + "ffn_gate.weight", h + "gate_proj.weight")
  val up: Tensor = weights(g + "ffn_up.weight", h + "up_proj.weight")
  val down: Tensor = weights(g + "ffn_down.weight", h + "down_proj.weight")
}

final class ExpertWeights(
    weights: HybridWeights,
    shape: HybridShape,
    g: String,
    h: String
) extends FeedForwardWeights {
  private val routed = shape.routed
  val router: Tensor = weights(g + "ffn_gate_inp.weight", h + "gate.weight")
  private def stacked(ggufName: String, projection: String) =
    weights.stackedExperts(g, h, ggufName, projection, routed.experts)
  val gate: Tensor = stacked("ffn_gate_exps.weight", "gate_proj")
  val up: Tensor = stacked("ffn_up_exps.weight", "up_proj")
  val down: Tensor = stacked("ffn_down_exps.weight", "down_proj")
  val sharedDown: Tensor =
    weights(g + "ffn_down_shexp.weight", h + "shared_expert.down_proj.weight")

  /** The shared gate and up seen as expert 0 of `[1, N, K]`. */
  val sharedGatePair: (Tensor, Tensor) =
    (
      weights(g + "ffn_gate_shexp.weight", h + "shared_expert.gate_proj.weight")
        .view(1, routed.sharedIntermediate, shape.hidden),
      weights(g + "ffn_up_shexp.weight", h + "shared_expert.up_proj.weight")
        .view(1, routed.sharedIntermediate, shape.hidden)
    )

  /** The routed experts and the shared expert (as expert 0 of one). */
  /** The shared expert's gate vector, F32 (a GGUF may store it in F16). */
  val sharedRouter: Tensor = weights.floats(
    g + "ffn_gate_inp_shexp.weight",
    h + "shared_expert_gate.weight",
    Shape.of(shape.hidden)
  )

  /** The routed experts and the shared expert (as expert 0 of one). */
  val projections: ExpertProjections = ExpertProjections(
    router,
    gate,
    up,
    down,
    sharedGatePair._1,
    sharedGatePair._2,
    sharedDown,
    sharedRouter
  )
}

/** The blocks' activations for up to `capacity` tokens, allocated once. */
final class BlockBuffers(ops: Ops, shape: HybridShape, val capacity: Int) {
  private val buffers = mutable.ArrayBuffer.empty[Tensor]
  private def allocate(dtype: DType, dimensions: Long*): Tensor = {
    val tensor =
      ops.allocate(dtype, Shape(capacity.toLong +: dimensions.toVector))
    buffers += tensor
    tensor
  }
  private def f32(dimensions: Long*) = allocate(DType.F32, dimensions*)
  // the widths the other feed-forward kind's buffers take are 0
  private val (experts, k, expertWidth, sharedWidth, denseWidth) =
    shape.feedForward match {
      case RoutedExperts(e, used, intermediate, shared) =>
        (e.toLong, used.toLong, intermediate.toLong, shared.toLong, 0L)
      case DenseFeedForward(intermediate) =>
        (0L, 0L, 0L, 0L, intermediate.toLong)
    }
  private val (h, d) = (shape.hidden.toLong, shape.headDimension.toLong)
  private val rule = shape.deltaRule
  // linear attention
  val qkv: Tensor = f32(shape.convChannels)
  val convolved: Tensor = f32(shape.convChannels)
  val z: Tensor = f32(shape.valueWidth)
  val a: Tensor = f32(rule.valueHeads)
  val b: Tensor = f32(rule.valueHeads)
  val core: Tensor = f32(shape.valueWidth)
  // full attention
  val queryAndGate: Tensor = f32(shape.heads * 2 * d)
  val queries: Tensor = f32(shape.heads * d)
  val gate: Tensor = f32(shape.heads * d)
  val keys: Tensor = f32(shape.kvHeads * d)
  val values: Tensor = f32(shape.kvHeads * d)
  val rotatedQueries: Tensor = f32(shape.heads * d)
  val rotatedKeys: Tensor = f32(shape.kvHeads * d)
  val attended: Tensor = f32(shape.heads * d)
  // mixture of experts
  val routerLogits: Tensor = f32(experts)
  val ids: Tensor = allocate(DType.I32, k)
  val routeWeights: Tensor = f32(k)
  val expertGate: Tensor = f32(k * expertWidth)
  val expertOut: Tensor = f32(k * h)
  val sharedGate: Tensor = f32(sharedWidth)
  val sharedOut: Tensor = f32(h)
  // dense feed-forward
  val denseGate: Tensor = f32(denseWidth)
  val denseUp: Tensor = f32(denseWidth)

  /** Expert 0 for every token: the shared expert's id. */
  val zeros: Tensor = {
    val tensor = allocate(DType.I32)
    ops.zero(tensor)
    tensor
  }
  // inputs
  private val positionRows: Tensor = allocate(DType.I32, 3)

  /** The rotary positions of `t` tokens, axis-major (`[3, t]` for mRoPE). */
  def positions(t: Long): Tensor =
    shape.ropeSections.positionsPerToken match {
      case 1    => positionRows.view(3L * capacity).prefix(t)
      case axes => positionRows.view(3L * capacity).prefix(axes, t)
    }

  def release(): Unit = buffers.foreach(ops.release)
}

/** The blocks themselves, each from its normed input to its output. */
final class HybridBlocks(ops: Ops, shape: HybridShape) {

  private val rule = shape.deltaRule
  private val rope = Rope(
    shape.ropeTheta,
    shape.rotaryDimensions,
    RopeLayout.Neox,
    shape.ropeSections
  )
  private val attention = Attention(
    (1 / math.sqrt(shape.headDimension)).toFloat,
    causal = true,
    None,
    None,
    None
  )

  /** Gated DeltaNet on `normed` (`[t, hidden]`) into `out`, through the
    * convolution's and the delta rule's states (and histories).
    */
  def linearAttention(
      weights: LinearAttentionWeights,
      normed: Tensor,
      convState: Tensor,
      deltaState: Tensor,
      convHistory: Option[Tensor],
      deltaHistory: Option[Tensor],
      t: Long,
      w: BlockBuffers,
      out: Tensor
  ): Unit = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (qkv, z, a, b) = (rows(w.qkv), rows(w.z), rows(w.a), rows(w.b))
    ops.linears(
      normed,
      Seq(weights.qkv, weights.z, weights.a, weights.b),
      Seq(qkv, z, a, b)
    )
    val core = rows(w.core).view(t, rule.valueHeads, rule.dimension)
    ops.convolvedDeltaRule(
      qkv,
      weights.conv,
      convState,
      convHistory,
      rows(w.convolved),
      a,
      b,
      weights.decay,
      weights.dtBias,
      deltaState,
      rule,
      deltaHistory,
      core
    )
    // the gated norm: norm(o) · w · gate(z), per head
    val perHead = core.view(t * rule.valueHeads, rule.dimension)
    ops.gatedRmsNorm(
      perHead,
      weights.norm,
      z.view(t * rule.valueHeads, rule.dimension),
      shape.deltaGate,
      shape.rmsEpsilon,
      perHead
    )
    ops.linear(core.view(t, shape.valueWidth), weights.out, out)
  }

  /** Gated attention with partial RoPE on `normed` into `out`, its keys and
    * values written to `cache` at `start`; `w.positions` holds the tokens'
    * rotary positions.
    */
  def fullAttention(
      weights: FullAttentionWeights,
      normed: Tensor,
      cache: KvCache,
      sequence: Sequence,
      start: Int,
      t: Long,
      w: BlockBuffers,
      out: Tensor
  ): Unit = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (heads, kvHeads, d) =
      (shape.heads.toLong, shape.kvHeads.toLong, shape.headDimension.toLong)
    val (epsilon, offset) = (shape.rmsEpsilon, shape.normOffset)
    val (queryAndGate, keys, values) =
      (rows(w.queryAndGate), rows(w.keys), rows(w.values))
    ops.linears(
      normed,
      Seq(weights.q, weights.k, weights.v),
      Seq(queryAndGate, keys, values)
    )
    val (queries, gate) = (rows(w.queries), rows(w.gate))
    val rotatedQueries = rows(w.rotatedQueries).view(t, heads, d)
    ops.attentionInputs(
      queryAndGate.view(t, heads, 2 * d),
      keys.view(t, kvHeads, d),
      values.view(t, kvHeads, d),
      QueryKeyNorms(weights.qNorm, weights.kNorm, epsilon, offset, rope),
      w.positions(t),
      cache,
      sequence.pageTable,
      start,
      queries.view(t, heads, d),
      gate.view(t, heads, d),
      rows(w.rotatedKeys).view(t, kvHeads, d),
      rotatedQueries
    )
    val attended = rows(w.attended).view(t, heads, d)
    ops.attention(
      rotatedQueries,
      cache,
      sequence.pageTable,
      start,
      start + t.toInt,
      attention,
      attended
    )
    ops.gated(
      Activation.Sigmoid,
      gate,
      attended.view(t, heads * d),
      attended.view(t, heads * d)
    )
    ops.linear(attended.view(t, heads * d), weights.o, out)
  }

  /** The routed experts and the gated shared expert on `normed`, summed into
    * `out`, plus `residual` when given (`out` may be the residual).
    */
  def experts(
      weights: ExpertWeights,
      normed: Tensor,
      residual: Option[Tensor],
      t: Long,
      w: BlockBuffers,
      out: Tensor
  ): Unit = {
    val into = activations(t, w)
    ops.expertOutputs(normed, weights.projections, into)
    ops.moeCombine(
      into.expertOut,
      into.routeWeights,
      Some(SharedExpert(into.sharedOut, normed, weights.sharedRouter)),
      residual,
      out
    )
  }

  /** The experts' activations for `t` tokens. */
  private def activations(t: Long, w: BlockBuffers): ExpertActivations = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val k = shape.routed.used.toLong
    ExpertActivations(
      rows(w.routerLogits),
      rows(w.ids),
      rows(w.routeWeights),
      rows(w.expertGate).view(t * k, shape.routed.intermediate),
      rows(w.expertOut).view(t * k, shape.hidden),
      rows(w.zeros),
      rows(w.sharedGate),
      rows(w.sharedOut)
    )
  }

  /** The dense SwiGLU on `normed` into `out`. */
  def dense(
      weights: DenseWeights,
      normed: Tensor,
      t: Long,
      w: BlockBuffers,
      out: Tensor
  ): Unit = {
    val (gate, up) = (w.denseGate.rows(0, t), w.denseUp.rows(0, t))
    ops.linears(normed, Seq(weights.gate, weights.up), Seq(gate, up))
    ops.gated(Activation.Silu, gate, up, gate)
    ops.linear(gate, weights.down, out)
  }

  /** The feed-forward on `normed`, added to `residual` in place, then `next`'s
    * RMS norm of the residual (the next block's input, which may be `normed`).
    */
  def feedForward(
      weights: FeedForwardWeights,
      normed: Tensor,
      residual: Tensor,
      t: Long,
      w: BlockBuffers,
      scratch: Tensor,
      next: RowNorm
  ): Unit = weights match {
    case experts: ExpertWeights =>
      ops.mixtureNorm(
        normed,
        experts.projections,
        activations(t, w),
        residual,
        next
      )
    case dense: DenseWeights =>
      this.dense(dense, normed, t, w, scratch)
      ops.addRmsNorm(residual, scratch, next)
  }
}

/** A model's activations for up to `capacity` tokens and `logitRows` rows of
  * logits.
  */
trait SizedWorkspace {
  def capacity: Int
  def logitRows: Int
  def release(): Unit
}

/** A workspace kept for its next caller, grown when too small. */
final class WorkspaceSlot[W <: SizedWorkspace](create: (Int, Int) => W) {
  private var held = Option.empty[W]

  def apply(tokens: Int, allLogits: Boolean): W = {
    val logitRows = if (allLogits) tokens else 1
    held
      .filter(w => w.capacity >= tokens && w.logitRows >= logitRows)
      .getOrElse {
        held.foreach(_.release())
        val created = create(
          math.max(tokens, held.fold(1)(_.capacity)),
          math.max(logitRows, held.fold(1)(_.logitRows))
        )
        held = Some(created)
        created
      }
  }

  def release(): Unit = held.foreach(_.release())
}
