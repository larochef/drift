package drift.frontend.pages.gallery

import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The pointer on the picture (`specs/27-redraw.md`): the box drawn, moved and
  * resized on it, and the tile grid dragged to another phase. All of it lives
  * in mouse events, which have no signal to observe, so the values they need —
  * the grid's numbers, whether it is even on screen — are kept here as they
  * come.
  *
  * The viewer draws; this decides what a press, a move and a release mean
  * (`specs/29-split-oversized-files.md`).
  */
class PicturePointer(
    /** The picture on screen and the box on it: what a drag writes. */
    viewed: Var[Option[ViewedImage]],
    /** The tiles the open task would run, which the box sticks to. */
    geometry: Signal[TileGeometry],
    /** Whether a box may be drawn: only a task that takes a selection has one,
      * and the grid is all there is to grab on the others.
      */
    selectable: Signal[Boolean],
    /** Whether the grid is drawn, since only a drawn line can be grabbed. */
    showTiles: Signal[Boolean],
    /** Where the grid is cut: a grid drag moves it. */
    gridOffset: Var[TileOffset]
) {

  /** The drag under way: where it started, where it is now, and what it
    * grabbed.
    */
  private val dragging = Var(Option.empty[PicturePointer.Drag])

  /** The row's numbers as they stand. A drag is handled in an event, which has
    * no signal to observe, and `Signal.now` is Airstream's own business — so
    * the latest value is kept here.
    */
  private val currentGeometry = Var[TileGeometry](RedrawGeometry())

  /** Whether a press may draw a box, kept for the same reason. */
  private val boxAllowed = Var(false)

  /** Whether the grid is on screen, for the same reason: a press has to know
    * whether there are lines to grab.
    */
  private val gridShown = Var(false)

  /** The grid drag under way: where it started and where the grid was then. */
  private val movingGrid = Var(Option.empty[PicturePointer.GridMove])

  /** Whether the pointer is over a grid line, which is what the move cursor
    * follows.
    */
  private val pointerOverGrid = Var(false)

  private def clearSelection(): Unit =
    viewed.update(_.map(_.copy(selection = None)))

  /** Where an event landed on the element handling it, in image pixels. */
  private def pointOf(event: dom.MouseEvent, image: ViewedImage): (Int, Int) = {
    val rect =
      event.currentTarget.asInstanceOf[dom.html.Element].getBoundingClientRect()
    def at(value: Double, start: Double, length: Double, total: Int) =
      if (length <= 0) 0
      else
        math.round(((value - start) / length) * total).toInt.max(0).min(total)
    (
      at(event.clientX, rect.left, rect.width, image.width),
      at(event.clientY, rect.top, rect.height, image.height)
    )
  }

  /** A handle's grab radius, and the drag's stickiness, in image pixels: both
    * are a fixed number of *screen* pixels, so they feel the same however the
    * browser has scaled the picture.
    */
  private def grabRadius(event: dom.MouseEvent, image: ViewedImage): Int = {
    val rect =
      event.currentTarget.asInstanceOf[dom.html.Element].getBoundingClientRect()
    if (rect.width <= 0) PicturePointer.GrabPixels
    else
      math
        .round(PicturePointer.GrabPixels * image.width / rect.width)
        .toInt
        .max(1)
  }

  /** The grid's own lines, in image px: the tile edges that are not the
    * picture's own, which are what a press grabs to move the grid.
    */
  private def gridLines(
      image: ViewedImage,
      sticky: TileGeometry
  ): (List[Int], List[Int]) = {
    val tiles = sticky.layoutFor(image.selection, image.width, image.height)
    def inner(edges: List[Int], limit: Int) =
      edges.distinct.filter(edge => edge > 0 && edge < limit).sorted
    (
      inner(
        tiles.flatMap(tile => List(tile.x, tile.x + tile.width)),
        image.width
      ),
      inner(
        tiles.flatMap(tile => List(tile.y, tile.y + tile.height)),
        image.height
      )
    )
  }

  /** Whether a point is on one of those lines. */
  private def onGrid(
      point: (Int, Int),
      image: ViewedImage,
      tolerance: Int
  ): Boolean =
    gridShown.now() && {
      val (vertical, horizontal) = gridLines(image, currentGeometry.now())
      vertical.exists(x => math.abs(x - point._1) <= tolerance) ||
      horizontal.exists(y => math.abs(y - point._2) <= tolerance)
    }

  /** What a press starts: moving the grid by the line it landed on, resizing
    * the box by the corner it landed on, moving the box it landed in, or
    * drawing a new one. The box wins over the grid where both are under the
    * pointer — it is the thing the user just drew. Where no box may be drawn, a
    * press off the grid starts nothing.
    */
  private def press(event: dom.MouseEvent): Unit =
    viewed.now().foreach { image =>
      event.preventDefault()
      val point = pointOf(event, image)
      val radius = grabRadius(event, image)
      val grip =
        image.selection
          .filter(_ => boxAllowed.now())
          .flatMap(PicturePointer.Grip.at(_, point, radius))
      if (grip.isEmpty && onGrid(point, image, radius))
        movingGrid.set(Some(PicturePointer.GridMove(point, gridOffset.now())))
      else if (boxAllowed.now())
        dragging.set(
          Some(PicturePointer.Drag(point, point, grip, image.selection))
        )
    }

  /** The grid follows the drag: snapped to the multiple the model aligns to,
    * and kept inside one stride, since shifting it by a whole stride gives back
    * the grid it started from. Dragging right moves the cuts right, which is a
    * *smaller* phase — the lattice is laid from `-phase`.
    */
  private def moveGrid(event: dom.MouseEvent): Unit =
    (movingGrid.now(), viewed.now()) match {
      case (Some(move), Some(image)) =>
        val (x, y) = pointOf(event, image)
        val sticky = currentGeometry.now()
        def shifted(from: Int, delta: Int, length: Int): Int = {
          val multiple = sticky.gridStep
          val stride = sticky.strideFor(length)
          val snapped =
            math.round((from - delta).toDouble / multiple).toInt * multiple
          ((snapped % stride) + stride) % stride
        }
        gridOffset.set(
          TileOffset(
            shifted(move.from.x, x - move.start._1, image.width),
            shifted(move.from.y, y - move.start._2, image.height)
          )
        )
      case _ => ()
    }

  /** Where the drag has got to: the box follows. */
  private def drag(event: dom.MouseEvent): Unit =
    (dragging.now(), viewed.now()) match {
      case (Some(started), Some(image)) =>
        val moved = started.copy(to = pointOf(event, image))
        dragging.set(Some(moved))
        boxOf(moved, image, grabRadius(event, image)).foreach(region =>
          viewed.set(Some(image.copy(selection = Some(region))))
        )
      case _ => ()
    }

  /** The drag ends. A press that never moved and grabbed nothing clears the
    * box, the way a click on the picture always has.
    */
  private def release(): Unit = {
    val started = dragging.now()
    dragging.set(None)
    movingGrid.set(None)
    started.foreach(movement =>
      if (movement.isClick && movement.grip.isEmpty) clearSelection()
    )
  }

  /** The box a drag makes: a new one, the old one moved, or the old one resized
    * by the corner grabbed — kept inside the picture, its sides pulled to the
    * sticky lengths.
    */
  private def boxOf(
      movement: PicturePointer.Drag,
      image: ViewedImage,
      tolerance: Int
  ): Option[ImageRegion] = {
    val sticky = currentGeometry.now()
    def stick(length: Int, imageLength: Int) =
      ViewedImage.snap(length, sticky.stickyLengths(imageLength), tolerance)
    def clamp(value: Int, limit: Int) = value.max(0).min(limit)

    (movement.grip, movement.from) match {
      case (Some(PicturePointer.Grip.Inside), Some(from)) =>
        // Moved: same size, still inside the picture.
        Some(
          ImageRegion(
            clamp(from.x + movement.dx, image.width - from.width),
            clamp(from.y + movement.dy, image.height - from.height),
            from.width,
            from.height
          )
        )
      case (Some(corner), Some(from)) =>
        // Resized: the corner opposite the one grabbed stays where it is.
        val (anchorX, anchorY) = corner.anchorOf(from)
        val width = stick(math.abs(movement.to._1 - anchorX), image.width)
        val height = stick(math.abs(movement.to._2 - anchorY), image.height)
        val x = if (movement.to._1 < anchorX) anchorX - width else anchorX
        val y = if (movement.to._2 < anchorY) anchorY - height else anchorY
        Some(
          ImageRegion(
            clamp(x, image.width),
            clamp(y, image.height),
            width.min(image.width),
            height.min(image.height)
          )
        )
      case _ =>
        // Drawn: from where the press landed to where the pointer is.
        val width = stick(math.abs(movement.dx), image.width)
        val height = stick(math.abs(movement.dy), image.height)
        Option.when(
          width >= ViewedImage.MinimumSide && height >= ViewedImage.MinimumSide
        ) {
          val x =
            if (movement.dx < 0) movement.start._1 - width
            else movement.start._1
          val y =
            if (movement.dy < 0) movement.start._2 - height
            else movement.start._2
          ImageRegion(
            clamp(x, image.width),
            clamp(y, image.height),
            width,
            height
          )
        }
    }
  }

  /** Whether the pointer is over a grid line, which is what the move cursor
    * follows.
    */
  val overGrid: Signal[Boolean] = pointerOverGrid.signal

  /** Everything the picture needs to be draggable: the values kept as they
    * come, and the four handlers.
    */
  val modifiers: Seq[Mod[HtmlElement]] = Seq(
    geometry --> currentGeometry,
    selectable --> boxAllowed,
    showTiles --> gridShown,
    onMouseDown --> press,
    onMouseMove --> { event =>
      if (movingGrid.now().isDefined) moveGrid(event)
      else if (dragging.now().isDefined) drag(event)
      else
        viewed
          .now()
          .foreach(image =>
            pointerOverGrid.set(
              onGrid(pointOf(event, image), image, grabRadius(event, image))
            )
          )
    },
    onMouseUp --> (_ => release()),
    onMouseLeave --> { _ =>
      pointerOverGrid.set(false)
      if (dragging.now().isDefined || movingGrid.now().isDefined) release()
    }
  )

  /** A picture shown in place of another is not a request to repaint the same
    * place on it.
    */
  def clearOnChange(of: EventStream[?]*): Seq[Mod[HtmlElement]] =
    of.map(_ --> (_ => clearSelection()))
}

object PicturePointer {

  /** How close to a corner a press counts as grabbing it, and how far a side
    * sticks to a tile boundary — in screen pixels, converted to image pixels
    * per picture.
    */
  val GrabPixels: Int = 14

  /** What a press grabbed: a corner of the box, or its middle. */
  enum Grip derives CanEqual {
    case TopLeft, TopRight, BottomLeft, BottomRight, Inside

    def cssClass: String = this match {
      case TopLeft     => "is-top-left"
      case TopRight    => "is-top-right"
      case BottomLeft  => "is-bottom-left"
      case BottomRight => "is-bottom-right"
      case Inside      => "is-inside"
    }

    /** The corner that stays put while this one is dragged. */
    def anchorOf(region: ImageRegion): (Int, Int) = this match {
      case TopLeft     => (region.x + region.width, region.y + region.height)
      case TopRight    => (region.x, region.y + region.height)
      case BottomLeft  => (region.x + region.width, region.y)
      case BottomRight => (region.x, region.y)
      case Inside      => (region.x, region.y)
    }
  }

  object Grip {
    val corners: List[Grip] =
      List(Grip.TopLeft, Grip.TopRight, Grip.BottomLeft, Grip.BottomRight)

    /** Which part of `region` a press at `point` grabbed, if any. */
    def at(
        region: ImageRegion,
        point: (Int, Int),
        radius: Int
    ): Option[Grip] = {
      val (x, y) = point
      def near(a: Int, b: Int) = math.abs(a - b) <= radius
      val right = region.x + region.width
      val bottom = region.y + region.height
      if (near(x, region.x) && near(y, region.y)) Some(TopLeft)
      else if (near(x, right) && near(y, region.y)) Some(TopRight)
      else if (near(x, region.x) && near(y, bottom)) Some(BottomLeft)
      else if (near(x, right) && near(y, bottom)) Some(BottomRight)
      else if (x >= region.x && x <= right && y >= region.y && y <= bottom)
        Some(Inside)
      else None
    }
  }

  /** A grid drag in progress: where it started, and where the grid stood then.
    */
  case class GridMove(start: (Int, Int), from: TileOffset)

  /** A drag in progress: where it started, where it is, what it grabbed, and
    * the box as it stood when it began.
    */
  case class Drag(
      start: (Int, Int),
      to: (Int, Int),
      grip: Option[Grip],
      from: Option[ImageRegion]
  ) {
    def dx: Int = to._1 - start._1
    def dy: Int = to._2 - start._2

    /** A press that never really moved: a click. */
    def isClick: Boolean =
      math.abs(dx) < ViewedImage.MinimumSide &&
        math.abs(dy) < ViewedImage.MinimumSide
  }
}
