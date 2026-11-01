package drift.runner.ops

/** The activations of `Ops.activation` and `Ops.gated`; `code` is the kernels'
  * `kind`.
  */
enum Activation(val code: Int) {
  case Silu extends Activation(0)

  /** GELU with the tanh approximation (`gelu_pytorch_tanh`). */
  case GeluTanh extends Activation(1)

  /** Exact GELU, through erf. */
  case GeluErf extends Activation(2)
  case Sigmoid extends Activation(3)

  /** MOSS-Image's SiTU, `4 tanh(v / 4) σ(v)`. In `gated` the up is soft-capped
    * too, `× 25 tanh(up / 25)`: its SiTU-GLU.
    */
  case Situ extends Activation(4)
}
