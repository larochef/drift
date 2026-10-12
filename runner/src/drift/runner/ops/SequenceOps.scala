package drift.runner.ops

import drift.runner.tensor.*

/** What language models keep and draw from token after token: the candidates
  * sampling draws from, the recurrent states, the residual streams and the
  * n-gram rows.
  */
trait SequenceOps {

  /** The index of an F32 tensor's largest value, the first among equals: greedy
    * decoding without bringing the logits back.
    */
  def argmax(values: Tensor): Int

  /** Each row's `count` largest values (F32 `[rows, width]`, at most
    * `Ops.MaximumCandidates`), largest first, the lower index first among
    * equals: what sampling draws from without bringing the logits back.
    */
  def candidates(values: Tensor, count: Int): Seq[Candidates]

  /** A causal depthwise convolution over time, its taps `dilation` tokens
    * apart, then SiLU: `x` and `out` `[tokens, C]`, `weight` `[C, taps]`,
    * `state` `[(taps − 1) × dilation, C]` holding the inputs before `x`'s first
    * token, and left holding its last ones. A `history` (`[≥ tokens, (taps − 1)
    * × dilation, C]`) receives the state after each token.
    */
  def causalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      history: Option[Tensor],
      out: Tensor
  ): Unit

  /** The gated delta rule over `tokens` in order (Gated DeltaNet): `qkv`
    * `[tokens, rule.qkvWidth]` (q, k, v heads), `a` and `b` `[tokens,
    * valueHeads]`, `decay` (`−exp(A_log)`) and `dtBias` `[valueHeads]`, `state`
    * `[valueHeads, D, D]` F32 carried between calls, `out` `[tokens,
    * valueHeads, D]`. A `history` (`[≥ tokens, valueHeads, D, D]`) receives the
    * state after each token.
    */
  def gatedDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit

  /** `causalConv` of `qkv` (dilation 1) into `convolved`, then `gatedDeltaRule`
    * of it: one launch where the backend fuses them (and leaves `convolved`
    * unwritten).
    */
  def convolvedDeltaRule(
      qkv: Tensor,
      convWeight: Tensor,
      convState: Tensor,
      convHistory: Option[Tensor],
      convolved: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    causalConv(qkv, convWeight, 1, convState, convHistory, convolved)
    gatedDeltaRule(convolved, a, b, decay, dtBias, state, rule, history, out)
  }

  /** A hyper-connection's read of the residual streams: `out[t] = mean_j
    * sigmoid(gates[t, j]) ⊙ normed[t, j]`, `gates` (logits) and `normed`
    * `[tokens, streams × width]`, `out` `[tokens, width]`.
    */
  def streamsMix(gates: Tensor, normed: Tensor, streams: Int, out: Tensor): Unit

  /** A hyper-connection's write: `x[t, j] += 2 sigmoid(logits[t, j] / streams)
    * × y[t]`, `x` `[tokens, streams × width]`, `y` `[tokens, width]`, `logits`
    * `[tokens, streams]` (zero: a plain residual add).
    */
  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit

  /** The n-gram embedding's gate per stream: `s = ⟨key[t, j], query[t, j]⟩ /
    * √width`, `out[t, j] = sigmoid(sign(s) √max(|s|, 1e−6)) × value[t]`; `key`,
    * `query` and `out` `[tokens, streams × width]`, `value` `[tokens, width]`.
    */
  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit

  /** The n-gram rows of each token (`NgramHash`): `tokens` I32 `[T]`, `rows`
    * I32 `[T, heads]`. `state` (F32 `[kept]`) holds the tokens before the
    * call's first as id + 1, oldest first (0: none, taken as EOS), and is left
    * holding its last ones. A `history` (`[≥ T, kept]`) receives it after each
    * token.
    */
  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit
}

/** The rules of `SequenceOps`.
  */
private[ops] trait SequenceChecks extends CommonChecks {

  /** The most candidates `candidates` returns per row: one workgroup sorts
    * them.
    */
  val MaximumCandidates = 1024

  def requireCandidates(values: Tensor, count: Int): Unit = {
    requireF32("candidates", values)
    require(
      count >= 1 && count <= MaximumCandidates,
      s"$count candidates, 1 to $MaximumCandidates"
    )
  }

  /** A state history holds one F32 state per token, room for `tokens`. */
  def checkHistory(history: Tensor, tokens: Int, state: Tensor): Unit =
    require(
      history.dtype == DType.F32 && history.shape.elementCount >= tokens.toLong * state.shape.elementCount,
      s"history ${history.shape} cannot keep $tokens states of ${state.shape}"
    )

  /** Returns `(tokens, channels, taps)`. */
  def checkCausalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    checkElementwise("causalConv", x, out)
    requireF32("causalConv", weight, state)
    require(
      x.shape.rank == 2,
      s"causalConv takes [tokens, channels], not ${x.shape}"
    )
    val Seq(tokens, channels) = x.shape.dimensions.map(_.toInt)
    val taps = weight.shape.last.toInt
    require(
      weight.shape == Shape.of(channels, taps) && taps <= 9,
      s"causalConv: weight ${weight.shape} for $channels channels"
    )
    val span = (taps - 1) * dilation
    require(
      dilation >= 1 && span <= 16 && state.shape == Shape.of(span, channels),
      s"causalConv: state ${state.shape}, expected [$span, $channels]"
    )
    (tokens, channels, taps)
  }

  /** Returns the token count. */
  def checkNgramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      rows: Tensor
  ): Int = {
    require(
      tokens.dtype == DType.I32 && rows.dtype == DType.I32,
      s"ngramRows: tokens ${tokens.dtype}, rows ${rows.dtype}"
    )
    requireF32("ngramRows", state)
    val count = tokens.shape.elementCount.toInt
    require(
      rows.shape == Shape.of(count, hash.heads) &&
        state.shape == Shape.of(hash.kept),
      s"ngramRows: rows ${rows.shape}, state ${state.shape} for $count tokens"
    )
    count
  }

  /** Returns `(tokens, streams, width)`. */
  def checkStreams(
      operation: String,
      streamsTensors: Seq[Tensor],
      single: Tensor
  ): (Int, Int, Int) = {
    requireF32(operation, (streamsTensors :+ single)*)
    requireSameShape(operation, streamsTensors*)
    val Seq(tokens, width) = single.shape.dimensions.map(_.toInt)
    val all = streamsTensors.head.shape
    require(
      all.rank == 2 && all.dimensions.head == tokens && all.last % width == 0,
      s"$operation: streams $all for rows of ${single.shape}"
    )
    (tokens, (all.last / width).toInt, width)
  }

  /** Returns the token count. */
  def checkDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      out: Tensor
  ): Int = {
    requireF32("gatedDeltaRule", qkv, a, b, decay, dtBias, state, out)
    val tokens = qkv.shape.dimensions.head.toInt
    val (v, d) = (rule.valueHeads.toLong, rule.dimension.toLong)
    require(
      qkv.shape == Shape.of(tokens, rule.qkvWidth),
      s"gatedDeltaRule: qkv ${qkv.shape}, expected [$tokens, ${rule.qkvWidth}]"
    )
    require(
      a.shape == Shape.of(tokens, v) && b.shape == a.shape,
      s"gatedDeltaRule: a ${a.shape}, b ${b.shape}"
    )
    require(
      decay.shape == Shape.of(v) && dtBias.shape == Shape.of(v),
      s"gatedDeltaRule: decay ${decay.shape}, dtBias ${dtBias.shape}"
    )
    require(
      state.shape == Shape.of(v, d, d),
      s"gatedDeltaRule: state ${state.shape}"
    )
    require(
      out.shape == Shape.of(tokens, v, d),
      s"gatedDeltaRule: out ${out.shape}"
    )
    tokens
  }
}
