package drift.shared

/** A rectangle of an image, in image pixels: a tile of a tiled job
  * (`specs/26-tiled-pid.md`), the part of a gallery image a redraw repaints, or
  * the window that part is repainted through (`specs/27-redraw.md`).
  */
case class ImageRegion(x: Int, y: Int, width: Int, height: Int)

/** Cutting a target into overlapping tiles, and the window a selected region is
  * redrawn through. Integer geometry only — no image class in sight — so the
  * browser lays out the very tiles the backend will run and can say what a job
  * costs before it starts; blending the decoded tiles back together is the
  * backend's `TileBlending`.
  */
object Tiling {

  /** The least overlap between neighbouring tiles, feather-blended — PiD's and
    * a redraw's.
    */
  val Overlap: Int = 256

  /** A tile of the target, in target pixels. */
  type Tile = ImageRegion
  val Tile: ImageRegion.type = ImageRegion

  /** One tile's place on an axis: where it starts and how long it is. Tiles of
    * a shifted grid are not all the same length — the two at the ends are cut
    * back to the axis — so each carries its own.
    */
  case class Span(start: Int, length: Int) {
    def end: Int = start + length
  }

  /** The tile length an axis of `length` px is cut into: the fewest tiles of at
    * most `maxTile` that cover it with neighbours sharing at least `overlap`
    * px, each tile then as short as that allows (a multiple of `multiple`, so
    * the overlaps stay near `overlap`) — a shorter tile is a faster pass. An
    * axis that fits is one tile of `length`.
    */
  def tileFor(length: Int, maxTile: Int, overlap: Int, multiple: Int): Int =
    if (length <= maxTile) length
    else {
      val count =
        math.ceil((length - overlap).toDouble / (maxTile - overlap)).toInt
      val covering =
        math.ceil((length + (count - 1) * overlap).toDouble / count).toInt
      (math.ceil(covering.toDouble / multiple).toInt * multiple) min maxTile
    }

  /** How far apart two neighbouring tiles start: what one pass of the model
    * advances by, and so the distance over which shifting the grid says
    * everything there is to say — a shift of a whole stride is the grid it
    * started from.
    */
  def strideFor(length: Int, maxTile: Int, overlap: Int, multiple: Int): Int =
    (tileFor(length, maxTile, overlap, multiple) - overlap) max 1

  /** Tiles an axis of `length` px: the tiles of `tileFor`, their starts spread
    * evenly as multiples of `align` and the last tile ending at `length`.
    * `length` and `maxTile` must be multiples of `multiple`, and `multiple` of
    * `align`.
    *
    * `offset` px shifts where the cuts fall (`specs/27-redraw.md`): the same
    * tiles on a grid moved by that much, so a face can be put inside one tile
    * instead of across the seam between two. It is taken modulo the stride — a
    * shift of one whole tile advance is no shift at all — and must be a
    * multiple of `multiple`, which is what the browser snaps a drag to. A
    * shifted grid usually costs one tile more than the even spread, and the
    * tiles at the two ends are shorter: `shifted` keeps them long enough to
    * paint.
    */
  def axis(
      length: Int,
      maxTile: Int,
      overlap: Int,
      multiple: Int,
      align: Int,
      offset: Int
  ): List[Span] =
    if (length <= maxTile) List(Span(0, length))
    else {
      val tile = tileFor(length, maxTile, overlap, multiple)
      val stride = tile - overlap
      val phase = ((offset % stride) + stride) % stride
      if (phase == 0) {
        val count =
          math.ceil((length - overlap).toDouble / (maxTile - overlap)).toInt
        List
          .tabulate(count)(index =>
            if (index == count - 1) length - tile
            else
              math
                .round(index.toDouble * (length - tile) / (count - 1) / align)
                .toInt * align
          )
          .map(start => Span(start, tile))
      } else shifted(length, tile, stride, phase, overlap, multiple)
    }

  /** The tiles of a grid shifted by `phase`: full-length tiles every `stride`
    * px from `phase` px before the axis starts, the two at the ends cut back to
    * the axis and rounded out to `multiple`.
    *
    * An end tile shorter than two overlaps is too little picture to paint well,
    * so it is given up and the tile beside it slides to that end instead —
    * never a gap, because a head that short means `phase > tile - 2 × overlap`,
    * which is exactly what puts the next tile within a tile's length of the
    * start; the same algebra holds at the other end. With fewer than three
    * tiles there is no neighbour to slide, and the short one is kept.
    */
  private def shifted(
      length: Int,
      tile: Int,
      stride: Int,
      phase: Int,
      overlap: Int,
      multiple: Int
  ): List[Span] = {
    val shortest = 2 * overlap
    val clipped = LazyList
      .iterate(-phase)(_ + stride)
      .takeWhile(_ < length)
      .toList
      .map { start =>
        if (start < 0) Span(0, roundUp(start + tile, multiple) min length)
        else if (start + tile > length) {
          val grown = roundUp(length - start, multiple) min length
          Span(length - grown, grown)
        } else Span(start, tile)
      }
    val fromHead =
      if (clipped.sizeIs < 3 || clipped.head.length >= shortest) clipped
      else
        clipped.tail match {
          case second :: rest => Span(0, second.length) :: rest
          case Nil            => clipped
        }
    fromHead.reverse match {
      case last :: previous :: rest
          if rest.nonEmpty && last.length < shortest =>
        (Span(length - previous.length, previous.length) :: rest).reverse
      case _ => fromHead
    }
  }

  /** The tiles covering a `width`×`height` target, row by row, on a grid
    * shifted by `offsetX`, `offsetY`.
    */
  def layout(
      width: Int,
      height: Int,
      maxTile: Int,
      overlap: Int,
      multiple: Int,
      align: Int,
      offsetX: Int,
      offsetY: Int
  ): List[List[Tile]] = {
    val columns = axis(width, maxTile, overlap, multiple, align, offsetX)
    val rows = axis(height, maxTile, overlap, multiple, align, offsetY)
    rows.map(row =>
      columns.map(column =>
        Tile(column.start, row.start, column.length, row.length)
      )
    )
  }

  /** The window a redraw of `region` runs through: the region grown by `margin`
    * px on every side, each side then at least `minimumSide` px — models return
    * mush well below the size they were trained at, and a selection is often
    * far smaller — rounded out to `multiple` and kept inside the image, shifted
    * rather than shrunk when it meets an edge so the minimum always holds. An
    * axis the window would cover anyway comes back whole, for the caller to pad
    * as it pads a full-image redraw.
    *
    * The margin is what the model sees around the selection *and* the room the
    * composite has to feather the repainted pixels into the untouched ones
    * (`TileBlending.paste`), so the window always contains the region.
    */
  def window(
      region: ImageRegion,
      imageWidth: Int,
      imageHeight: Int,
      minimumSide: Int,
      margin: Int,
      multiple: Int
  ): ImageRegion = {
    val (x, width) =
      windowAxis(
        region.x,
        region.width,
        imageWidth,
        minimumSide,
        margin,
        multiple
      )
    val (y, height) =
      windowAxis(
        region.y,
        region.height,
        imageHeight,
        minimumSide,
        margin,
        multiple
      )
    ImageRegion(x, y, width, height)
  }

  /** One axis of `window`: where the window starts and how long it is. */
  private def windowAxis(
      start: Int,
      size: Int,
      length: Int,
      minimumSide: Int,
      margin: Int,
      multiple: Int
  ): (Int, Int) = {
    val wanted = roundUp(math.max(size + 2 * margin, minimumSide), multiple)
    if (wanted >= length) (0, length)
    else {
      val centred = math.round(start + size / 2.0 - wanted / 2.0).toInt
      (centred.max(0).min(length - wanted), wanted)
    }
  }

  /** The longest an axis can be and still take exactly `tiles` tiles of at most
    * `maxTile`, overlapping by `overlap`. One px more and `axis` cuts another
    * tile — which is what makes these the sizes a selection should stick to: a
    * box just over the line costs a whole extra pass.
    */
  def spanForTiles(tiles: Int, maxTile: Int, overlap: Int): Int =
    tiles * maxTile - (tiles - 1) * overlap

  def roundUp(value: Int, multiple: Int): Int =
    (value + multiple - 1) / multiple * multiple
}
