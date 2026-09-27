package drift.runner.ops

import drift.runner.tensor.Tensor

/** How queries see keys in `Ops.attention`: scores are `q·k × scale`, capped to
  * `softcap × tanh(score / softcap)` when set (Gemma 2), hidden past the
  * query's own position when `causal`, and before `position − window + 1` with
  * a sliding `window`. `sinks` (F32, one per query head) join each softmax as
  * one extra logit that takes weight but carries no value.
  */
final case class Attention(
    scale: Float,
    causal: Boolean,
    window: Option[Int],
    softcap: Option[Float],
    sinks: Option[Tensor]
)

object Attention {

  /** The head widths the GPU's attention kernels take. */
  val KernelHeadWidths: Seq[Int] = Seq(64, 80, 128, 256)

  /** The narrowest kernel width that holds heads of `width`: a model with other
    * heads pads them with zeros, which change neither scores nor values.
    */
  def paddedHeadWidth(width: Int): Int =
    KernelHeadWidths
      .find(_ >= width)
      .getOrElse(
        throw new IllegalArgumentException(
          s"no attention kernel for heads of $width"
        )
      )
}
