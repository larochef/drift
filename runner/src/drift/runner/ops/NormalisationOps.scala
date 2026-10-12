package drift.runner.ops

import drift.runner.tensor.*

/** Normalisations and activations: the RMS, group and layer norms, softmax, and
  * the activation functions, plain or gating.
  */
trait NormalisationOps { this: Ops =>

  def activation(kind: Activation, x: Tensor, out: Tensor): Unit =
    scaledActivation(kind, 1f, x, out)

  /** `out = kind(inputScale × x)`, elementwise. */
  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit

  /** `out = kind(gate) × up`: SwiGLU with `Silu`, GeGLU with a GELU. */
  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit

  /** RMS norm over the last dimension:
    * `out = x / sqrt(mean(x²) + epsilon) × (weight + weightOffset)`;
    * `weightOffset` is 1 for Gemma, whose norm weights are stored minus one.
    */
  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit

  /** The residual stream's update and the next block's input: `x += y`, then
    * `norm`'s RMS norm of `x` into its `out`. One launch where the backend
    * fuses them.
    */
  def addRmsNorm(x: Tensor, y: Tensor, norm: RowNorm): Unit = {
    add(x, y, x)
    rmsNorm(x, norm.weight, norm.epsilon, norm.weightOffset, norm.out)
  }

  /** RMS norm of each of a row's `groups` equal parts on its own, with `weight`
    * spanning the whole row (`[groups × width]`): the residual streams of Qwen
    * 3.8 Flash Next, each normed with its own weights.
    */
  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit

  /** Group norm over rows of channels (`[pixels, C]`): group `g` is the
    * channels `[g C / groups, (g + 1) C / groups)` of every row, normalized by
    * its mean and variance over all of them, then `× weight + bias`.
    */
  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit

  /** Layer norm over the last dimension, `weight` and `bias` optional. */
  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit

  /** `out = softmax(x × scale)` over the last dimension. */
  def softmax(x: Tensor, scale: Float, out: Tensor): Unit

  /** Gated DeltaNet's output norm: `out = rmsNorm(x) × weight ×
    * gateActivation(gate)` (SiLU in Qwen 3.5/3.6, sigmoid in 3.8), per row of
    * `x` and `gate` (`[rows, D]`).
    */
  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit
}

/** The rules of `NormalisationOps`.
  */
private[ops] trait NormalisationChecks extends CommonChecks {

  /** Returns `(rows, C)`. */
  def checkGroupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): (Long, Int) = {
    checkRowWise("groupNorm", x, out, weight, bias)
    require(x.shape.rank == 2, s"groupNorm takes [rows, C], not ${x.shape}")
    val Seq(rows, channels) = x.shape.dimensions
    require(
      groups > 0 && channels % groups == 0,
      s"groupNorm: $groups groups of $channels channels"
    )
    (rows, channels.toInt)
  }
}
