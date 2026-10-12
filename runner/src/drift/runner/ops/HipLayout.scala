package drift.runner.ops

import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

/** The GPU backend's rearrangements of values.
  */
private[ops] trait HipLayout extends HipBase {

  private def patchPixels(
      from: Tensor,
      to: Tensor,
      image: Tensor,
      patch: Int,
      back: Boolean
  ): Unit = {
    val (height, width, channels) =
      Ops.checkPatchPixels(image, if (back) from else to, patch)
    launch(
      kernel(imageKernels, "patch_pixels_f32"),
      (image.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(from)),
      Pointer(pointer(to)),
      I32(height),
      I32(width),
      I32(channels),
      I32(patch),
      I32(if (back) 1 else 0)
    )
  }

  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit =
    patchPixels(image, out, image, patch, back = false)

  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit =
    patchPixels(rows, out, out, patch, back = true)

  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (gridHeight, gridWidth, channels) =
      Ops.checkPackPatches(image, patch, out)
    launch(
      kernel(imageKernels, "pack_patches_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(image)),
      Pointer(pointer(out)),
      I32(gridHeight),
      I32(gridWidth),
      I32(channels),
      I32(patch)
    )
  }

  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit = {
    val (rows, columns) = Ops.checkConcatColumns(parts, out)
    parts.zip(columns).zip(columns.scanLeft(0)(_ + _)).foreach {
      case ((part, width), start) =>
        launch(
          kernel(imageKernels, "copy_columns_f32"),
          (rows * width + 255) / 256,
          256,
          Pointer(pointer(part)),
          Pointer(pointer(out)),
          I64(rows),
          I32(width),
          I32(columns.sum),
          I32(start)
        )
    }
  }

  def transpose(x: Tensor, out: Tensor): Unit = {
    val Seq(rows, cols) = Ops.checkTranspose(x, out)
    launch(
      kernel(
        imageKernels,
        if (x.dtype == DType.F32) "transpose_f32" else "transpose_16"
      ),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(rows),
      I32(cols)
    )
  }

  def scatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, columns) = Ops.checkScatterAddRows(rows, ids, weights, out)
    launch(
      kernel(imageKernels, "scatter_add_rows_f32"),
      (slots * columns + 255) / 256,
      256,
      Pointer(pointer(rows)),
      Pointer(pointer(ids)),
      Pointer(pointer(weights)),
      Pointer(pointer(out)),
      I64(slots),
      I32(columns)
    )
  }

  def columnMean(x: Tensor, out: Tensor): Unit = {
    val (rows, columns) = Ops.checkColumnMean(x, out)
    launch(
      kernel(imageKernels, "column_mean_f32"),
      columns,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I64(rows),
      I32(columns)
    )
  }

  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit = {
    val (gridWidth, channels) =
      Ops.checkUnpackPatches(packed, gridHeight, patch, out)
    launch(
      kernel(imageKernels, "unpack_patches_f32"),
      (out.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(packed)),
      Pointer(pointer(out)),
      I32(gridHeight),
      I32(gridWidth),
      I32(channels),
      I32(patch)
    )
  }

  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val count = first.shape.elementCount
    val d = first.shape.last
    launch(
      kernel(linearAttentionKernels, "split_halves"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(x)),
      Pointer(pointer(first)),
      Pointer(pointer(second)),
      I64(count / d),
      I32(d.toInt)
    )
  }

  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val count = first.shape.elementCount
    val d = first.shape.last
    launch(
      kernel(linearAttentionKernels, "join_halves"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(first)),
      Pointer(pointer(second)),
      Pointer(pointer(x)),
      I64(count / d),
      I32(d.toInt)
    )
  }
}
