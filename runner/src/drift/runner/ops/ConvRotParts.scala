package drift.runner.ops

import drift.runner.tensor.*

/** What makes an int8 of a 4-bit code (`ComfyQuant`): the 16 levels, F32, and
  * each row's relative scales, fp8 E4M3, one a `groupSize` columns.
  */
final case class ConvRotLevels(
    codebook: Tensor,
    relative: Tensor,
    groupSize: Int
)

/** What a rotated linear's codes are decoded with: `scales` F32, one a row (or
  * one for all, without `levels`), and a 4-bit linear's `levels`.
  */
final case class ConvRotParts(scales: Tensor, levels: Option[ConvRotLevels])
