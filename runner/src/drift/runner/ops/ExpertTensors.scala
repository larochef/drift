package drift.runner.ops

import drift.runner.tensor.*

/** A mixture's shared expert in `Ops.moeCombine`: its `output` `[tokens, H]`,
  * weighted by the sigmoid of `input` (`[tokens, H]`) · `router` (F32 `[H]`).
  */
final case class SharedExpert(output: Tensor, input: Tensor, router: Tensor)

/** A mixture of experts' weights: an F32 router `[experts, hidden]`, the routed
  * experts' gate, up (`[experts, N, hidden]`) and down (`[experts, hidden,
  * N]`), and a shared expert's gate and up (`[1, N, hidden]`), down (`[hidden,
  * N]`) and F32 gate vector (`[hidden]`, see `SharedExpert`).
  */
final case class ExpertProjections(
    router: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    sharedGate: Tensor,
    sharedUp: Tensor,
    sharedDown: Tensor,
    sharedRouter: Tensor
)

/** A mixture of experts' activations for `t` tokens of `k` slots each: the
  * router's `logits` `[t, experts]`, the chosen `ids` and `routeWeights`
  * `[t, k]`, each slot's SwiGLU `gated` `[t·k, N]` and output `expertOut`
  * `[t·k, hidden]`; the shared expert's `sharedIds` (a zero per token),
  * `sharedGated` `[t, N]` and `sharedOut` `[t, hidden]`.
  */
final case class ExpertActivations(
    logits: Tensor,
    ids: Tensor,
    routeWeights: Tensor,
    gated: Tensor,
    expertOut: Tensor,
    sharedIds: Tensor,
    sharedGated: Tensor,
    sharedOut: Tensor
)
