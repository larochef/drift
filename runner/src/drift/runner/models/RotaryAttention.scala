package drift.runner.models

import drift.runner.ops.*
import drift.runner.tensor.*

/** Self-attention of a whole sequence without a causal order, its queries and
  * keys turned by their positions (`rope`), for the diffusion transformers that
  * run it the same way: queries of `heads`, keys and values of `kvHeads`, heads
  * of `headDimension`.
  *
  * The families differ in what stands around the attention — how they project
  * (fused or not, a bias, a LoRA), whether a head norm is learned, what gates
  * the result — so each form here takes exactly the part two of them share.
  */
final class RotaryAttention(
    ops: Ops,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    rope: Rope
) {

  private val (h, kv, d) = (heads.toLong, kvHeads.toLong, headDimension.toLong)

  /** `out` (`[tokens, heads × headDimension]`) from `q` (the same), `k` and `v`
    * (`[tokens, kvHeads × headDimension]`) at `positions`, the keys and values
    * through `cache`'s one page: `q` and `k` are turned into a prefix of the
    * flat `rotatedQueries` and `rotatedKeys`.
    */
  def paged(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      positions: Tensor,
      rotatedQueries: Tensor,
      rotatedKeys: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      attention: Attention,
      out: Tensor
  ): Unit = {
    val tokens = q.shape.dimensions.head
    val (queries, keys) =
      (rotatedQueries.prefix(tokens, h, d), rotatedKeys.prefix(tokens, kv, d))
    ops.rope(q.view(tokens, h, d), positions, rope, queries)
    ops.rope(k.view(tokens, kv, d), positions, rope, keys)
    ops.cacheWrite(keys, v.view(tokens, kv, d), cache, pageTable, 0)
    ops.attention(
      queries,
      cache,
      pageTable,
      0,
      tokens.toInt,
      attention,
      out.view(tokens, h, d)
    )
  }

  /** A block's whole attention, from its normed input `x` to `projected`: the
    * fused projections `query`, `key` and `value` into `q`, `k` and `v`, each
    * head of `q` and `k` RMS-normed (`queryNorm`, `keyNorm`), both turned at
    * `positions` into `rotatedQueries` and `rotatedKeys`, `attention` into
    * `attended`, and the `output` projection. Every buffer holds exactly the
    * sequence's rows.
    */
  def projected(
      x: Tensor,
      query: Tensor,
      key: Tensor,
      value: Tensor,
      queryNorm: Tensor,
      keyNorm: Tensor,
      epsilon: Float,
      positions: Tensor,
      attention: PlainAttention,
      output: Tensor,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      rotatedQueries: Tensor,
      rotatedKeys: Tensor,
      attended: Tensor,
      projected: Tensor
  ): Unit = {
    val tokens = q.shape.dimensions.head
    ops.linears(x, Seq(query, key, value), Seq(q, k, v))
    ops.rmsNorm(
      q.view(tokens * h, d),
      queryNorm,
      epsilon,
      0f,
      q.view(tokens * h, d)
    )
    ops.rmsNorm(
      k.view(tokens * kv, d),
      keyNorm,
      epsilon,
      0f,
      k.view(tokens * kv, d)
    )
    ops.rope(
      q.view(tokens, h, d),
      positions,
      rope,
      rotatedQueries.view(tokens, h, d)
    )
    ops.rope(
      k.view(tokens, kv, d),
      positions,
      rope,
      rotatedKeys.view(tokens, kv, d)
    )
    attention(rotatedQueries, rotatedKeys, v, attended)
    ops.linear(attended, output, projected)
  }
}
