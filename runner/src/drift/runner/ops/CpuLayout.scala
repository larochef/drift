package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The reference backend's rearrangements of values.
  */
private[ops] trait CpuLayout extends CpuBase {

  /** The image index of patch-ordered row value `index`. */
  private def patchPixel(
      index: Long,
      width: Int,
      channels: Int,
      patch: Int
  ): Long = {
    val c = index % channels
    val row = index / channels
    val area = patch * patch
    val (l, p) = (row / area, row % area)
    val gridWidth = width / patch
    val (y, x) =
      (l / gridWidth * patch + p / patch, l % gridWidth * patch + p % patch)
    (y * width + x) * channels + c
  }

  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, width, channels) = Ops.checkPatchPixels(image, out, patch)
    val input = reader(image)
    fill(out)(index => input(patchPixel(index, width, channels, patch)))
  }

  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, width, channels) = Ops.checkPatchPixels(out, rows, patch)
    val input = reader(rows)
    val target = segment(out)
    (0L until rows.shape.elementCount).foreach { index =>
      target.setAtIndex(
        JAVA_FLOAT,
        patchPixel(index, width, channels, patch),
        input(index).toFloat
      )
    }
  }

  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, gridWidth, channels) = Ops.checkPackPatches(image, patch, out)
    val input = reader(image)
    val features = channels.toLong * patch * patch
    fill(out) { index =>
      val (token, feature) = (index / features, index % features)
      val c = feature / (patch * patch)
      val (py, px) = (feature % (patch * patch) / patch, feature % patch)
      val (y, xx) =
        (token / gridWidth * patch + py, token % gridWidth * patch + px)
      input((y * gridWidth * patch + xx) * channels + c)
    }
  }

  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit = {
    val (_, columns) = Ops.checkConcatColumns(parts, out)
    val starts = columns.scanLeft(0)(_ + _)
    val inputs = parts.map(reader)
    val width = columns.sum
    fill(out) { index =>
      val (row, column) = (index / width, (index % width).toInt)
      val part = starts.lastIndexWhere(_ <= column)
      inputs(part)(row * columns(part) + column - starts(part))
    }
  }

  def transpose(x: Tensor, out: Tensor): Unit = {
    val Seq(rows, cols) = Ops.checkTranspose(x, out)
    if (x.dtype == DType.F32) {
      val input = reader(x)
      fill(out)(index => input((index % rows) * cols + index / rows))
    } else {
      val (from, to) = (segment(x), segment(out))
      (0L until rows.toLong * cols).foreach { index =>
        val source = (index % rows) * cols + index / rows
        to.setAtIndex(JAVA_SHORT, index, from.getAtIndex(JAVA_SHORT, source))
      }
    }
  }

  def scatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, columns) = Ops.checkScatterAddRows(rows, ids, weights, out)
    val (input, factors, index, target) =
      (reader(rows), reader(weights), segment(ids), segment(out))
    for {
      slot <- 0L until slots
      column <- 0 until columns
    } {
      val at = index.getAtIndex(JAVA_INT, slot).toLong * columns + column
      target.setAtIndex(
        JAVA_FLOAT,
        at,
        (target.getAtIndex(JAVA_FLOAT, at) +
          factors(slot) * input(slot * columns + column)).toFloat
      )
    }
  }

  def columnMean(x: Tensor, out: Tensor): Unit = {
    val (rows, columns) = Ops.checkColumnMean(x, out)
    val input = reader(x)
    fill(out)(c => (0L until rows).map(r => input(r * columns + c)).sum / rows)
  }

  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit = {
    val (gridWidth, channels) =
      Ops.checkUnpackPatches(packed, gridHeight, patch, out)
    val input = reader(packed)
    val width = gridWidth.toLong * patch
    fill(out) { index =>
      val c = index % channels
      val p = index / channels
      val (y, xx) = (p / width, p % width)
      val token = (y / patch) * gridWidth + xx / patch
      input(
        token * channels * patch * patch + c * patch * patch + (y % patch) * patch + xx % patch
      )
    }
  }

  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val input = reader(x)
    val d = first.shape.last
    fill(first)(i => input(i / d * 2 * d + i % d))
    fill(second)(i => input(i / d * 2 * d + d + i % d))
  }

  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val (left, right) = (reader(first), reader(second))
    val d = first.shape.last
    fill(x) { i =>
      val (row, column) = (i / (2 * d), i % (2 * d))
      if (column < d) left(row * d + column) else right(row * d + column - d)
    }
  }
}
