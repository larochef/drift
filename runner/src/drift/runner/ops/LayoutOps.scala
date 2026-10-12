package drift.runner.ops

import drift.runner.tensor.*

/** Rearrangements of values: patches and pixels, halves, columns, transposes,
  * scatters and means over rows.
  */
trait LayoutOps {

  /** A channels-last image `[H, W, C]` as rows of patch-ordered pixels
    * `[L × patch², C]`: row `l × patch² + py × patch + px` is pixel `(hs ×
    * patch + py, ws × patch + px)` of patch `l = hs × W / patch + ws`.
    * `patchesToPixels` is its inverse.
    */
  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit
  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit

  /** `unpackPatches`' inverse: a channels-last image `[gridH × patch, gridW ×
    * patch, C]` into packed patches `[gridH × gridW, C × patch²]`.
    */
  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit

  /** `out` (`[rows, Σ columns]`) as `parts` (each `[rows, columns]`) side by
    * side.
    */
  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit

  /** `out[c, r] = x[r, c]`: F32, or F16 or BF16 as they are stored. */
  def transpose(x: Tensor, out: Tensor): Unit

  /** A weighted scatter of rows: `out[ids[s]] += weights[s] × rows[s]` for
    * every slot `s`, `rows` `[slots, D]`, `ids` I32 `[slots]`, `weights` F32
    * `[slots]`, `out` `[tokens, D]`. A token may take any number of slots (a
    * mixture of experts where the experts choose their tokens).
    */
  def scatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): Unit

  /** Each column's mean over the rows: `x` `[rows, C]`, `out` `[C]` (a global
    * average pool of rows of pixels).
    */
  def columnMean(x: Tensor, out: Tensor): Unit

  /** Packed `patch × patch` patches (`[gridH × gridW, C × patch²]`, feature
    * `c × patch² + py × patch + px`) back to a channels-last image `[gridH ×
    * patch, gridW × patch, C]`.
    */
  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit

  /** Each row's last dimension, `2D`, split into its halves: `[.., 2D]` into
    * `first` and `second`, `[.., D]` each. Qwen 3.5's query projection holds a
    * head's query then its output gate.
    */
  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit

  /** The inverse of `splitHalves`: `first` and `second` side by side. */
  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit
}

/** The rules of `LayoutOps`.
  */
private[ops] trait LayoutChecks extends CommonChecks {

  def checkSplitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    requireF32("splitHalves", x, first, second)
    requireSameShape("splitHalves", first, second)
    require(
      x.shape.elementCount == 2 * first.shape.elementCount && x.shape.last == 2 * first.shape.last,
      s"splitHalves: ${x.shape} into ${first.shape}"
    )
  }

  /** Returns `(H, W, C)` of the image. */
  def checkPatchPixels(
      image: Tensor,
      rows: Tensor,
      patch: Int
  ): (Int, Int, Int) = {
    requireF32("patch pixels", image, rows)
    val Seq(height, width, channels) = image.shape.dimensions.map(_.toInt)
    require(
      height % patch == 0 && width % patch == 0 &&
        rows.shape == Shape.of(height.toLong * width, channels),
      s"patch pixels: ${image.shape} in patches of $patch as ${rows.shape}"
    )
    (height, width, channels)
  }

  /** Returns `(gridH, gridW, C)`. */
  def checkPackPatches(
      image: Tensor,
      patch: Int,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("packPatches", image, out)
    val Seq(height, width, channels) = image.shape.dimensions.map(_.toInt)
    require(
      height % patch == 0 && width % patch == 0 &&
        out.shape == Shape.of(
          (height / patch).toLong * (width / patch),
          channels.toLong * patch * patch
        ),
      s"packPatches: ${image.shape} in patches of $patch into ${out.shape}"
    )
    (height / patch, width / patch, channels)
  }

  /** Returns `(rows, columns of each part)`. */
  def checkConcatColumns(parts: Seq[Tensor], out: Tensor): (Long, Seq[Int]) = {
    requireF32("concatColumns", (parts :+ out)*)
    require(
      (parts :+ out).forall(_.shape.rank == 2),
      "concatColumns takes [rows, columns]"
    )
    val rows = out.shape.dimensions.head
    val columns = parts.map(_.shape.last.toInt)
    require(
      parts.forall(_.shape.dimensions.head == rows) &&
        columns.sum == out.shape.last,
      s"concatColumns: ${parts.map(_.shape).mkString(", ")} into ${out.shape}"
    )
    (rows, columns)
  }

  /** `columnMean`'s `(rows, columns)`. */
  def checkColumnMean(x: Tensor, out: Tensor): (Long, Int) = {
    requireF32("columnMean", x, out)
    val Seq(rows, columns) = x.shape.dimensions
    require(
      out.shape == Shape.of(columns),
      s"columnMean: ${x.shape} into ${out.shape}"
    )
    (rows, columns.toInt)
  }

  /** `scatterAddRows`' `(slots, columns)`. */
  def checkScatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): (Long, Int) = {
    requireF32("scatterAddRows", rows, weights, out)
    val Seq(slots, columns) = rows.shape.dimensions
    require(
      ids.dtype == DType.I32 && ids.shape == Shape.of(slots) &&
        weights.shape == Shape.of(slots) && out.shape.last == columns,
      s"scatterAddRows: ${rows.shape} by ${ids.shape} and ${weights.shape} into ${out.shape}"
    )
    (slots, columns.toInt)
  }

  def checkTranspose(x: Tensor, out: Tensor): Seq[Int] = {
    require(
      x.dtype == out.dtype &&
        Seq(DType.F32, DType.F16, DType.BF16).contains(x.dtype),
      s"transpose takes F32, F16 or BF16 into the same, not ${x.dtype} into ${out.dtype}"
    )
    val Seq(rows, cols) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == Shape.of(cols, rows),
      s"transpose: ${x.shape} into ${out.shape}"
    )
    Seq(rows, cols)
  }

  /** Returns `(gridW, C)`. */
  def checkUnpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): (Int, Int) = {
    requireF32("unpackPatches", packed, out)
    val Seq(tokens, features) = packed.shape.dimensions.map(_.toInt)
    val (gridWidth, channels) =
      (tokens / gridHeight, features / (patch * patch))
    require(
      tokens % gridHeight == 0 && features % (patch * patch) == 0 &&
        out.shape == Shape
          .of(gridHeight.toLong * patch, gridWidth.toLong * patch, channels),
      s"unpackPatches: ${packed.shape} on $gridHeight rows into ${out.shape}"
    )
    (gridWidth, channels)
  }
}
