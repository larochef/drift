package drift.runner.ops

import drift.runner.tensor.*

/** Mixtures of experts: the routing, the experts' products and their sum.
  */
trait ExpertsOps { this: Ops =>

  /** Top-k routing: each token's `k` largest `logits` (`[tokens, experts]`)
    * into `ids` (I32 `[tokens, k]`), their softmax among themselves into
    * `weights` (`[tokens, k]`).
    */
  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit

  /** Experts' products: slot `s` multiplies expert `ids[s]` of `weight`
    * (`[experts, N, K]`) with row `s / (slots / rows of x)` of `x`, into row
    * `s` of `out` (`[slots, N]`).
    */
  def expertsLinear(x: Tensor, weight: Tensor, ids: Tensor, out: Tensor): Unit

  /** `out[t] = residual[t] + Σ_j weights[t, j] · expertOut[t·k + j]`, plus the
    * shared expert's output scaled by `sigmoid(input[t] · router)` when there
    * is one. `out` may be the residual (the layer's `x`).
    */
  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit

  /** `moeCombine` into `residual`, then `norm`'s RMS norm of it into its `out`
    * (the next block's input). `norm.out` may be the shared expert's input.
    */
  def moeCombineNorm(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    moeCombine(expertOut, weights, shared, Some(residual), residual)
    rmsNorm(residual, norm.weight, norm.epsilon, norm.weightOffset, norm.out)
  }

  /** `expertOutputs`, then their sum into `residual` and its norm
    * (`moeCombineNorm`, the shared expert weighted by its gate vector over
    * `x`): the whole mixture of experts in one launch where the backend fuses
    * it.
    */
  def mixtureNorm(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    expertOutputs(x, weights, into)
    moeCombineNorm(
      into.expertOut,
      into.routeWeights,
      Some(SharedExpert(into.sharedOut, x, weights.sharedRouter)),
      residual,
      norm
    )
  }

  /** A router's `logits = x · routerᵀ`, then `route` of them. */
  def routeLinear(
      x: Tensor,
      router: Tensor,
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): Unit = {
    linear(x, router, logits)
    route(logits, ids, weights)
  }

  /** SwiGLU of experts: slot `s` computes `silu(gate · x) × (up · x)` with
    * expert `ids[s]` of `gate` and `up` (`[experts, N, K]` each), `x` as in
    * `expertsLinear`.
    */
  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit

  /** `expertsGatedLinear` of the routed experts, and of a shared expert of the
    * same shapes (`sharedGate` and `sharedUp` `[1, N, K]`, `sharedIds` a zero
    * per token) into `sharedOut`: one launch where the backend fuses them.
    */
  def expertsGatedLinearWithShared(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      sharedGate: Tensor,
      sharedUp: Tensor,
      sharedIds: Tensor,
      sharedOut: Tensor
  ): Unit = {
    expertsGatedLinear(x, gate, up, ids, out)
    expertsGatedLinear(x, sharedGate, sharedUp, sharedIds, sharedOut)
  }

  /** A mixture of experts' outputs on `x` before they are summed (see
    * `ExpertActivations`): the routing, every slot's SwiGLU and down
    * projection, and the shared expert's. One launch where the backend fuses
    * them.
    */
  def expertOutputs(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Unit = {
    routeLinear(x, weights.router, into.logits, into.ids, into.routeWeights)
    val slots = into.ids.view(into.ids.shape.elementCount)
    expertsGatedLinearWithShared(
      x,
      weights.gate,
      weights.up,
      slots,
      into.gated,
      weights.sharedGate,
      weights.sharedUp,
      into.sharedIds,
      into.sharedGated
    )
    expertsLinear(into.gated, weights.down, slots, into.expertOut)
    linear(into.sharedGated, weights.sharedDown, into.sharedOut)
  }
}

/** The rules of `ExpertsOps`.
  */
private[ops] trait ExpertsChecks extends CommonChecks {

  /** Returns `(tokens, experts, k)`. */
  def checkRoute(
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): (Int, Int, Int) = {
    requireF32("route", logits, weights)
    require(
      ids.dtype == DType.I32 && ids.shape == weights.shape && logits.shape.rank == 2,
      s"route: ${logits.shape}, ${ids.shape}, ${weights.shape}"
    )
    val Seq(tokens, experts) = logits.shape.dimensions.map(_.toInt)
    val k = weights.shape.last.toInt
    require(
      weights.shape == Shape.of(tokens, k) && k <= 16 && experts <= 512,
      s"route: top-$k of $experts"
    )
    (tokens, experts, k)
  }

  /** Returns `(slots, N, K, divisor)`. */
  def checkExperts(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("expertsLinear", x, out)
    require(
      weight.shape.rank == 3 && ids.dtype == DType.I32,
      s"expertsLinear: weight ${weight.shape}, ids ${ids.dtype}"
    )
    val Seq(_, n, k) = weight.shape.dimensions.map(_.toInt)
    val slots = ids.shape.elementCount.toInt
    val rows = x.shape.dimensions.head.toInt
    require(
      x.shape == Shape.of(rows, k) && slots % rows == 0,
      s"expertsLinear: x ${x.shape} for $slots slots of $k"
    )
    require(
      out.shape == Shape.of(slots, n),
      s"expertsLinear: out ${out.shape}, expected [$slots, $n]"
    )
    (slots, n, k, slots / rows)
  }
}
