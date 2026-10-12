package drift.runner.ops

import drift.runner.tensor.*

/** Convolutions and resampling, channels-last: the VAEs' convolutions over
  * pictures, the vocoders' over time with their activations, and the changes of
  * resolution.
  */
trait ConvolutionOps {

  /** A 3×3 convolution, channels-last: `x` `[H, W, in]`, `weight` `[out, in ×
    * 9]` (a `[out][in][3][3]` kernel flattened; BF16 on the GPU, whose patches
    * and sums are BF16, or F16, whose patches are F16 and sums F32), `bias` F32
    * `[out]`, `out` `[H / stride, W / stride, out]`. Stride 1 pads by 1 on
    * every side; stride 2 (even H and W) by 1 at the bottom and right only, as
    * diffusers' downsamplers do. `replicate` pads with the nearest edge pixel
    * instead of zeros (PyTorch's `padding_mode="replicate"`; stride 1),
    * `reflect` with the reflection (`"reflect"`: −1 reads 1; either stride).
    */
  def conv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int = 1,
      replicate: Boolean = false,
      reflect: Boolean = false
  ): Unit

  /** A 1-D convolution over time, channels-last: `x` `[T, in]`, `weight` `[out,
    * in × taps]` (a `[out][in][taps]` kernel flattened; BF16 or F32 on the GPU,
    * its patches of the same type, the sums F32), taps `dilation` samples
    * apart, zeros `padLeft` before and `padRight` after `x`, `bias` F32 `[out]`
    * when given, `out` `[(T + padLeft + padRight − dilation × (taps − 1) − 1) /
    * stride + 1, out]`.
    */
  def conv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): Unit

  /** A transposed 1-D convolution's overlap-add: `columns` `[T, out × taps]`
    * (the input times the weight regrouped as `[out × taps, in]`), `out`
    * `[(T − 1) × stride − 2 × pad + taps, out]` with `out[t × stride + k − pad,
    * o] = Σ columns[t, o × taps + k] + bias[o]`.
    */
  def overlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): Unit

  /** BigVGAN's anti-aliased SnakeBeta over time, channels-last (`x` and `out`
    * `[T, C]`): each channel upsampled ×2 (a transposed convolution by
    * `upFilter`, times 2, replicate-padded by 5 then cropped by 15 at both
    * ends), then `u + inverseMagnitude × sin²(frequency × u)`, then filtered by
    * `downFilter` at stride 2 (replicate-padded by 5 before, 6 after). The two
    * filters F32 `[12]`, `frequency` and `inverseMagnitude` F32 `[C]`.
    */
  def antiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): Unit

  /** DAC's Snake over time, channels-last (`x` and `out` `[T, C]`): `x + sin²(α
    * x) / (α + 10⁻⁹)`, `alpha` F32 `[C]` as stored (not log-scale).
    */
  def snake(x: Tensor, alpha: Tensor, out: Tensor): Unit

  /** Nearest-neighbour upsampling ×2, channels-last: `x` `[H, W, C]`, `out`
    * `[2H, 2W, C]`.
    */
  def upsample2x(x: Tensor, out: Tensor): Unit

  /** Wan 2.2's up-shortcut on one frame (diffusers' `DupUp3D`, the last of its
    * `frames` kept), channels-last: `x` `[H, W, C]` into `out` `[2H, 2W, O]`,
    * output `(y, x, o)` reading input channel `j / (O × frames × 4 / C)` at
    * `(y / 2, x / 2)`, where `j = ((o × frames + frames − 1) × 2 + y % 2) × 2 +
    * x % 2`.
    */
  def duplicateUp(x: Tensor, frames: Int, out: Tensor): Unit

  /** Wan 2.2's down-shortcut on one frame (diffusers' `AvgDown3D`, zero frames
    * before it to make `frames`), channels-last: `x` `[H, W, C]` into `out`
    * `[H / 2, W / 2, O]`, output channel `o` the mean of the `G = C × frames ×
    * 4 / O` values `k = o × G + g`, where value `k = ((c × frames + t) × 2 +
    * dy) × 2 + dx` is input `(2y + dy, 2x + dx, c)` for the last frame `t` and
    * zero for the others.
    */
  def averageDown(x: Tensor, frames: Int, out: Tensor): Unit

  /** A depthwise 3×3 convolution, channels-last, zeros around the image: `x`
    * and `out` `[H, W, C]`, `weight` F32 `[C, 9]` (each channel's own kernel,
    * tap `ky × 3 + kx`), `bias` F32 `[C]`.
    */
  def depthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): Unit
}

/** The rules of `ConvolutionOps`.
  */
private[ops] trait ConvolutionChecks extends CommonChecks {

  /** Returns `(H, W, in, out)` of the input. */
  def checkConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int
  ): (Int, Int, Int, Int) = {
    requireF32("conv3x3", x, bias, out)
    require(
      x.shape.rank == 3 && out.shape.rank == 3,
      s"conv3x3 takes [H, W, C], not ${x.shape}"
    )
    val Seq(height, width, in) = x.shape.dimensions.map(_.toInt)
    val Seq(outChannels, columns) = weight.shape.dimensions.map(_.toInt)
    require(
      stride == 1 || stride == 2 && height % 2 == 0 && width % 2 == 0,
      s"conv3x3: stride $stride on ${x.shape}"
    )
    require(
      columns == 9 * in && bias.shape == Shape.of(outChannels) &&
        out.shape == Shape.of(height / stride, width / stride, outChannels),
      s"conv3x3: x ${x.shape}, weight ${weight.shape}, out ${out.shape}, stride $stride"
    )
    (height, width, in, outChannels)
  }

  /** `conv1d`'s `(T, in, out length, out channels)`. */
  def checkConv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("conv1d", (Seq(x, out) ++ bias)*)
    require(
      x.shape.rank == 2 && out.shape.rank == 2,
      s"conv1d takes [T, C], not ${x.shape}"
    )
    val Seq(length, in) = x.shape.dimensions.map(_.toInt)
    val Seq(outChannels, columns) = weight.shape.dimensions.map(_.toInt)
    require(
      taps > 0 && dilation > 0 && stride > 0 && padLeft >= 0 && padRight >= 0,
      s"conv1d: $taps taps, dilation $dilation, stride $stride, pads $padLeft/$padRight"
    )
    val span = length + padLeft + padRight - dilation * (taps - 1) - 1
    require(span >= 0, s"conv1d: $length samples under $taps taps")
    val outLength = span / stride + 1
    require(
      columns == in * taps && bias.forall(_.shape == Shape.of(outChannels)) &&
        out.shape == Shape.of(outLength, outChannels),
      s"conv1d: x ${x.shape}, weight ${weight.shape}, $taps taps, out ${out.shape}"
    )
    (length, in, outLength, outChannels)
  }

  /** `overlapAdd`'s `(T, out length, out channels)`. */
  def checkOverlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("overlapAdd", columns, bias, out)
    val Seq(length, width) = columns.shape.dimensions.map(_.toInt)
    val outChannels = width / taps
    val outLength = (length - 1) * stride - 2 * pad + taps
    require(
      width % taps == 0 && pad >= 0 && outLength > 0 &&
        bias.shape == Shape.of(outChannels) &&
        out.shape == Shape.of(outLength, outChannels),
      s"overlapAdd: columns ${columns.shape}, $taps taps, stride $stride, pad $pad, out ${out.shape}"
    )
    (length, outLength, outChannels)
  }

  /** `antiAliasedSnake`'s `(T, C)`. */
  def checkAntiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): (Int, Int) = {
    requireF32(
      "antiAliasedSnake",
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    requireSameShape("antiAliasedSnake", x, out)
    val Seq(length, channels) = x.shape.dimensions.map(_.toInt)
    require(
      frequency.shape == Shape.of(channels) &&
        inverseMagnitude.shape == Shape.of(channels) &&
        upFilter.shape == Shape.of(SnakeTaps) &&
        downFilter.shape == Shape.of(SnakeTaps),
      s"antiAliasedSnake: x ${x.shape}, filters ${upFilter.shape} and ${downFilter.shape}"
    )
    (length, channels)
  }

  /** `snake`'s `(T, C)`. */
  def checkSnake(x: Tensor, alpha: Tensor, out: Tensor): (Int, Int) = {
    requireF32("snake", x, alpha, out)
    requireSameShape("snake", x, out)
    val Seq(length, channels) = x.shape.dimensions.map(_.toInt)
    require(
      alpha.shape == Shape.of(channels),
      s"snake: x ${x.shape}, alpha ${alpha.shape}"
    )
    (length, channels)
  }

  /** The taps of BigVGAN's anti-aliasing filters. */
  val SnakeTaps = 12

  /** Returns `(H, W, C)` of the input. */
  def checkUpsample2x(x: Tensor, out: Tensor): (Int, Int, Int) = {
    requireF32("upsample2x", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == Shape.of(2L * height, 2L * width, channels),
      s"upsample2x: ${x.shape} into ${out.shape}"
    )
    (height, width, channels)
  }

  /** `duplicateUp`'s `(height, width, channels, out channels)`. */
  def checkDuplicateUp(
      x: Tensor,
      frames: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("duplicateUp", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    val outChannels = out.shape.last.toInt
    require(
      out.shape == Shape.of(2L * height, 2L * width, outChannels) &&
        outChannels * frames * 4 % channels == 0,
      s"duplicateUp: ${x.shape} into ${out.shape} over $frames frames"
    )
    (height, width, channels, outChannels)
  }

  /** `averageDown`'s `(height, width, channels, out channels)`, of `x`. */
  def checkAverageDown(
      x: Tensor,
      frames: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("averageDown", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    val outChannels = out.shape.last.toInt
    require(
      height % 2 == 0 && width % 2 == 0 &&
        out.shape == Shape.of(height / 2L, width / 2L, outChannels) &&
        channels * frames * 4 % outChannels == 0,
      s"averageDown: ${x.shape} into ${out.shape} over $frames frames"
    )
    (height, width, channels, outChannels)
  }

  /** `depthwiseConv3x3`'s `(height, width, channels)`. */
  def checkDepthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("depthwiseConv3x3", x, weight, bias, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == x.shape && weight.shape == Shape.of(channels, 9) &&
        bias.shape == Shape.of(channels),
      s"depthwiseConv3x3: ${x.shape} by ${weight.shape} and ${bias.shape} into ${out.shape}"
    )
    (height, width, channels)
  }
}
