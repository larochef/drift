package drift.frontend.pages.gallery

import drift.shared.{ImageRegion, Tiling}

/** The image the detail view is showing and the box dragged on it
  * (`specs/27-redraw.md`): its size in its own pixels, so a selection made on a
  * picture the browser scaled down still names real pixels, and the part of it
  * a redraw should repaint — none for the whole image.
  *
  * The viewer writes it, the redraw panel reads it: the size alone already
  * tells the panel how many tiles a job will cost.
  */
case class ViewedImage(
    width: Int,
    height: Int,
    selection: Option[ImageRegion] = None
)

/** How far the tile grid is shifted over the picture, in image px on each axis
  * (`specs/27-redraw.md`). The panel sends it with the job and the viewer moves
  * it when the grid is dragged, so both are looking at the one number.
  */
case class TileOffset(x: Int, y: Int) {
  def isCentred: Boolean = x == 0 && y == 0
}

/** What a redraw would make of a selection: the largest tile it cuts, the
  * margin it grows the box by, the window it will not go under, and where the
  * grid of tiles is cut (`specs/27-redraw.md`). The redraw panel owns these
  * numbers; the image needs them to draw the grid, to say how many tiles a box
  * costs and to stick to the sizes where that count changes.
  */
/** What has become of a tile drawn over the picture
  * (`specs/15-post-hoc-resize.md`): a job paints them in order, so the picture
  * fills up as it goes.
  */
enum TileState derives CanEqual {
  case Done, Running, Waiting

  def styleClass: String = this match {
    case Done    => "is-done"
    case Running => "is-running"
    case Waiting => "is-waiting"
  }
}

/** One tile drawn over the picture: where it is, and how far the job has got
  * with it (`specs/15-post-hoc-resize.md`). What the done ones hold is the
  * job's picture, laid under the grid whole (`DetailPicture.jobPicture`).
  */
case class TilePaint(area: ImageRegion, state: TileState)

case class RedrawGeometry(
    tileSize: Int = 1280,
    margin: Int = 64,
    minimumWindow: Int = 1024,
    /** The multiple the chosen configuration's model aligns every side up to
      * (`Architecture.sizeMultiple`) — the backend lays the tiles out on it.
      */
    sizeMultiple: Int = 16,
    /** How far the grid is shifted from the even spread, per axis. */
    offsetX: Int = 0,
    offsetY: Int = 0
) {

  /** The area a redraw of `region` would actually repaint. */
  def windowFor(region: ImageRegion, width: Int, height: Int): ImageRegion =
    Tiling.window(
      region,
      width,
      height,
      minimumWindow,
      margin,
      multiple = sizeMultiple
    )

  /** What a redraw would repaint: the window a box is drawn through, or the
    * whole picture when there is no box.
    */
  def areaFor(
      region: Option[ImageRegion],
      width: Int,
      height: Int
  ): ImageRegion =
    region.fold(ImageRegion(0, 0, width, height))(
      windowFor(_, width, height)
    )

  /** The tiles a redraw would run, in the picture's own pixels — the very
    * rectangles the backend lays out (`Tiling.layout`), so the grid drawn over
    * the image is the grid of the job. They overlap by `Tiling.Overlap`: those
    * bands are where two passes meet and are feathered together.
    *
    * The area is padded up to the model's multiple first, exactly as the
    * backend pads it, so the last row and column can reach a few px past the
    * picture.
    */
  def layoutFor(
      region: Option[ImageRegion],
      width: Int,
      height: Int
  ): List[ImageRegion] = {
    val area = areaFor(region, width, height)
    Tiling
      .layout(
        Tiling.roundUp(area.width, sizeMultiple),
        Tiling.roundUp(area.height, sizeMultiple),
        Tiling.roundUp(tileSize, sizeMultiple),
        Tiling.Overlap,
        multiple = sizeMultiple,
        align = 1,
        offsetX = offsetX,
        offsetY = offsetY
      )
      .flatten
      .map(tile =>
        ImageRegion(area.x + tile.x, area.y + tile.y, tile.width, tile.height)
      )
  }

  /** How many tiles that takes — one pass each, which is what a redraw costs.
    */
  def tilesFor(region: ImageRegion, width: Int, height: Int): Int =
    layoutFor(Some(region), width, height).size

  /** How far apart the tiles start on an axis of `length` px: the whole
    * distance a shift can move the grid through, since shifting it by a stride
    * gives back the grid it started from.
    */
  def strideFor(length: Int): Int =
    Tiling.strideFor(
      Tiling.roundUp(length, sizeMultiple),
      Tiling.roundUp(tileSize, sizeMultiple),
      Tiling.Overlap,
      multiple = sizeMultiple
    )

  /** The selection lengths where the tile count changes on one axis: the
    * longest box that still fits in one tile, in two, in three… A drag lands on
    * one of these rather than a pixel past it (`ViewedImage.snap`).
    */
  def stickyLengths(imageLength: Int): List[Int] =
    LazyList
      .from(1)
      .map(tiles =>
        Tiling.spanForTiles(tiles, tileSize, Tiling.Overlap) - 2 * margin
      )
      .takeWhile(_ < imageLength + tileSize)
      .toList
}

object ViewedImage {

  /** Under this many pixels a side, a drag is a click that clears the selection
    * rather than a region to repaint (`TiledJobs.MinimumRegionSide`).
    */
  val MinimumSide: Int = 16

  /** `length` pulled to the nearest sticky size within `tolerance` px, or left
    * alone. The box then stops at the last size that costs one tile, two tiles…
    * and only crosses when the drag insists.
    */
  def snap(length: Int, sticky: List[Int], tolerance: Int): Int =
    sticky
      .filter(candidate => math.abs(candidate - length) <= tolerance)
      .minByOption(candidate => math.abs(candidate - length))
      .getOrElse(length)
}
