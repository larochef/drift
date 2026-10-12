package drift.runner.ops

import drift.runner.tensor.*

/** A gated attention's query and key preparation: RMS norms with `queryNorm`
  * and `keyNorm` (`epsilon` and `weightOffset` as `Ops.rmsNorm` takes them),
  * then `rope`.
  */
final case class QueryKeyNorms(
    queryNorm: Tensor,
    keyNorm: Tensor,
    epsilon: Float,
    weightOffset: Float,
    rope: Rope
)

/** An RMS norm taken of a result as it is written: `weight`, `epsilon` and
  * `weightOffset` as `Ops.rmsNorm` takes them, into `out`.
  */
final case class RowNorm(
    weight: Tensor,
    epsilon: Float,
    weightOffset: Float,
    out: Tensor
)
