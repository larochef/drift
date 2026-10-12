package drift.runner.ops

import drift.runner.native.*
import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

/** The GPU backend's convolutions and resampling.
  */
private[ops] trait HipConvolution extends HipBase {

  /** im2col's patches of this many bytes at most per GEMM. */
  private val PatchBudget = 256L << 20

  def conv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int,
      replicate: Boolean,
      reflect: Boolean
  ): Unit = {
    val (height, width, in, outChannels) =
      Ops.checkConv3x3(x, weight, bias, out, stride)
    require(
      weight.dtype == DType.BF16 || weight.dtype == DType.F16,
      s"conv3x3 takes BF16 or F16 weights, not ${weight.dtype}"
    )
    val half = weight.dtype == DType.F16
    val pixels = (height / stride).toLong * (width / stride)
    val columns = in * 9L
    val chunk = math.max(1L, math.min(pixels, PatchBudget / (2 * columns)))
    val patches = scratch(0, 2 * chunk * columns + 2 * chunk * outChannels)
    val result = offset(patches, 2 * chunk * columns)
    (0L until pixels by chunk).foreach { first =>
      val count = math.min(chunk, pixels - first)
      launch(
        kernel(imageKernels, if (half) "im2col_3x3_f16" else "im2col_3x3_bf16"),
        (count * columns + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(patches),
        I32(height),
        I32(width),
        I32(in),
        I32(width / stride),
        I32(stride),
        I32(if (replicate) 1 else if (reflect) 2 else 0),
        I64(first),
        I64(count)
      )
      val rows = out.view(pixels, outChannels.toLong).rows(first, count)
      if (half)
        // F16 operands, F32 sums: no rounding of the output
        blas.gemm(
          patches,
          pointer(weight),
          pointer(rows),
          count.toInt,
          outChannels,
          columns.toInt,
          HipBlas.RealF16,
          HipBlas.RealF32
        )
      else {
        blas.gemm(
          patches,
          pointer(weight),
          result,
          count.toInt,
          outChannels,
          columns.toInt,
          HipBlas.RealBF16
        )
        convertAt(result, pointer(rows), count * outChannels, 3)
      }
      addRow(rows, bias, rows)
    }
  }

  /** BF16 weights take BF16 patches, as `conv3x3`'s; F32 ones F32 patches. The
    * sums are F32 either way: a waveform does not take BF16's rounding at every
    * layer.
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
  ): Unit = {
    val (length, in, outLength, outChannels) = Ops.checkConv1d(
      x,
      weight,
      bias,
      taps,
      dilation,
      stride,
      padLeft,
      padRight,
      out
    )
    val (patchBytes, im2col, dataType) = weight.dtype match {
      case DType.BF16 => (2, "im2col_1d_bf16", HipBlas.RealBF16)
      case DType.F32  => (4, "im2col_1d_f32", HipBlas.RealF32)
      case other      =>
        throw new UnsupportedOperationException(
          s"conv1d takes BF16 or F32 weights, not $other"
        )
    }
    val columns = in.toLong * taps
    val chunk = math.max(
      1L,
      math.min(outLength.toLong, PatchBudget / (patchBytes * columns))
    )
    val patches = scratch(0, patchBytes * chunk * columns)
    (0L until outLength by chunk).foreach { first =>
      val count = math.min(chunk, outLength - first)
      launch(
        kernel(audioKernels, im2col),
        (count * columns + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(patches),
        I32(length),
        I32(in),
        I32(taps),
        I32(dilation),
        I32(stride),
        I32(padLeft),
        I64(first),
        I64(count)
      )
      val rows = out.rows(first, count)
      blas.gemm(
        patches,
        pointer(weight),
        pointer(rows),
        count.toInt,
        outChannels,
        columns.toInt,
        dataType,
        HipBlas.RealF32
      )
      bias.foreach(b => addRow(rows, b, rows))
    }
  }

  def overlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): Unit = {
    val (length, outLength, outChannels) =
      Ops.checkOverlapAdd(columns, taps, stride, pad, bias, out)
    launch(
      kernel(audioKernels, "overlap_add_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(columns)),
      Pointer(pointer(bias)),
      Pointer(pointer(out)),
      I32(length),
      I32(outLength),
      I32(outChannels),
      I32(taps),
      I32(stride),
      I32(pad)
    )
  }

  def snake(x: Tensor, alpha: Tensor, out: Tensor): Unit = {
    val (length, channels) = Ops.checkSnake(x, alpha, out)
    launch(
      kernel(audioKernels, "snake_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(alpha)),
      Pointer(pointer(out)),
      I64(length.toLong * channels),
      I32(channels)
    )
  }

  def antiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): Unit = {
    val (length, channels) = Ops.checkAntiAliasedSnake(
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    launch(
      kernel(audioKernels, "anti_aliased_snake_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(frequency)),
      Pointer(pointer(inverseMagnitude)),
      Pointer(pointer(upFilter)),
      Pointer(pointer(downFilter)),
      Pointer(pointer(out)),
      I32(length),
      I32(channels)
    )
  }

  def upsample2x(x: Tensor, out: Tensor): Unit = {
    val (height, width, channels) = Ops.checkUpsample2x(x, out)
    launch(
      kernel(imageKernels, "upsample_2x_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels)
    )
  }

  def duplicateUp(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (height, width, channels, outChannels) =
      Ops.checkDuplicateUp(x, frames, out)
    launch(
      kernel(imageKernels, "duplicate_up_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels),
      I32(outChannels),
      I32(frames)
    )
  }

  def averageDown(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (height, width, channels, outChannels) =
      Ops.checkAverageDown(x, frames, out)
    launch(
      kernel(imageKernels, "average_down_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels),
      I32(outChannels),
      I32(frames)
    )
  }

  def depthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): Unit = {
    val (height, width, channels) =
      Ops.checkDepthwiseConv3x3(x, weight, bias, out)
    require(
      pointer(x) != pointer(out),
      "depthwiseConv3x3 reads its neighbours: not in place"
    )
    launch(
      kernel(imageKernels, "depthwise_conv_3x3_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(weight)),
      Pointer(pointer(bias)),
      Pointer(pointer(out)),
      I32(height),
      I32(width),
      I32(channels)
    )
  }
}
