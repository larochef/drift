package drift.frontend.pages.gallery

import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.TileAreaFields.{Cut, TilePlan}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Where a tiled task's tiles fall (`specs/27-redraw.md`,
  * `specs/39-seamless-edit.md`, `specs/26-tiled-pid.md`,
  * `specs/51-seedvr2-upscaling.md`): the tile field, the grid drawn over the
  * picture and moved by hand, and the plan — one piece every tiled task holds
  * its own of, since they all cut the picture the same way. A redraw and an
  * edit take a selection, and have the window and margin it is worked through;
  * an upscale works the whole picture only (`takesSelection`).
  *
  * The panels stay built while hidden, so only the one on screen publishes its
  * numbers to the picture: `active` says which, and becoming active publishes
  * them at once.
  */
class TileAreaFields(
    /** How the task cuts the picture as its other fields stand — none while
      * there is nothing to plan: no model chosen yet, a job that is refused, a
      * model that cuts nothing.
      */
    cut: Signal[Option[Cut]],
    /** Whether the task works a box drawn on the picture. */
    takesSelection: Boolean,
    /** The image on screen and the box drawn on it. */
    viewed: Var[Option[ViewedImage]],
    /** Where the numbers go for the picture to count and draw tiles with. */
    geometry: Var[TileGeometry],
    showTileGrid: Var[Boolean],
    gridOffset: Var[TileOffset],
    active: Signal[Boolean]
) {

  // A selection's tile is the model's to bear, 1280 by default; an upscale's
  // is the largest its runtime takes until a smaller one is asked for.
  private val tileVar = Var(if (takesSelection) "1280" else "")
  private val windowVar = Var("1024")
  private val marginVar = Var("64")

  private def number(state: Var[String], fallback: Int): Int =
    state.now().trim.toIntOption.getOrElse(fallback)

  def tileSize: Int = number(tileVar, 1280)

  /** The tile an upscale asks for, in target px — none for the largest. */
  def askedTileSize: Option[Int] = tileVar.now().trim.toIntOption
  def minimumWindowSide: Int = number(windowVar, 1024)
  def selectionMargin: Int = number(marginVar, 64)
  def region: Option[ImageRegion] = viewed.now().flatMap(_.selection)
  def gridOffsetX: Int = gridOffset.now().x
  def gridOffsetY: Int = gridOffset.now().y

  /** Whether a box is drawn on the picture — what the button is named after. */
  val hasSelection: Signal[Boolean] =
    viewed.signal.map(_.exists(_.selection.isDefined)).distinct

  /** The numbers the picture is judged by, as the fields stand. */
  private val currentGeometry: Signal[Option[TileGeometry]] =
    tileVar.signal
      .combineWith(marginVar.signal, windowVar.signal, cut, gridOffset.signal)
      .map((tile, margin, window, shape, offset) =>
        shape.map {
          case Cut.Selection(sizeMultiple) =>
            RedrawGeometry(
              tileSize = tile.trim.toIntOption.getOrElse(1280),
              margin = margin.trim.toIntOption.getOrElse(64),
              minimumWindow = window.trim.toIntOption.getOrElse(1024),
              sizeMultiple = sizeMultiple,
              offsetX = offset.x,
              offsetY = offset.y
            )
          case Cut.Whole(tiling, targetPerPixel) =>
            UpscaleGeometry(
              tiling.withTile(tile.trim.toIntOption),
              targetPerPixel,
              offset.x,
              offset.y
            )
        }
      )
      .distinct

  /** The geometry as it stands, for the request built on a click. */
  private val latestGeometry = Var(Option.empty[TileGeometry])

  /** How far an upscale's grid is shifted, in target px — what its request
    * carries, and what the grid on the picture is drawn from.
    */
  def targetGridOffset: (Int, Int) =
    latestGeometry.now() match {
      case Some(upscale: UpscaleGeometry) =>
        (upscale.shifted.offsetX, upscale.shifted.offsetY)
      case _ => (0, 0)
    }

  /** The picture gets them as they are typed, while this task is the one on
    * screen: the box counts its tiles, sticks to their boundaries and draws the
    * grid from the very numbers this panel plans with. With nothing to plan
    * there is no grid, rather than the last task's.
    */
  val publishGeometry: Modifier[HtmlElement] = Seq(
    currentGeometry --> latestGeometry,
    currentGeometry.combineWith(active).changes.collect {
      case (published, true) => published.getOrElse(NoTiles)
    } --> geometry
  )

  /** What the fields as they stand would run on the picture on screen. */
  val plan: Signal[Option[TilePlan]] =
    viewed.signal.combineWith(currentGeometry).map { (shown, current) =>
      for {
        picture <- shown
        sticky <- current
      } yield {
        val selection = picture.selection.filter(_ => takesSelection)
        TilePlan(
          selection,
          sticky.areaFor(selection, picture.width, picture.height),
          sticky.layoutFor(selection, picture.width, picture.height).size
        )
      }
    }

  /** The largest tile an upscale's runtime takes, as the field's hint. */
  private val largestTile: Signal[String] =
    cut.map {
      case Some(Cut.Whole(tiling, _)) => tiling.tile.toString
      case _                          => ""
    }.distinct

  /** What is worked on, and how it is cut up to fit through the model. */
  def areaGroup: HtmlElement =
    if (takesSelection) selectionGroup
    else
      group(
        "area",
        field(
          "tile",
          numberField(tileVar, "5rem").amend(
            minAttr := UpscaleTiling.MinimumTile.toString,
            stepAttr := "64",
            placeholder <-- largestTile,
            title := "largest tile, in px of the result — empty for the " +
              "largest the model takes in one pass, which is the fewest " +
              "tiles; smaller means more tiles and more seams to blend"
          )
        )
      )

  private def selectionGroup: HtmlElement = group(
    "area",
    field(
      "tile",
      numberField(tileVar, "5rem").amend(
        minAttr := "512",
        maxAttr := "2048",
        stepAttr := "16",
        title := "largest tile, in px — bigger means fewer tiles and more " +
          "picture in each; 2048 has crashed the VAE on ROCm before"
      )
    ),
    field(
      "window",
      numberField(windowVar, "5rem").amend(
        minAttr := "256",
        maxAttr := "4096",
        stepAttr := "16",
        title := "smallest side, in px, of the area a selection is worked " +
          "through — a model given far fewer pixels than it was trained on " +
          "paints mush"
      )
    ),
    field(
      "margin",
      numberField(marginVar, "4.5rem").amend(
        minAttr := "0",
        maxAttr := "512",
        stepAttr := "8",
        title := "px kept around the selection: what the model sees of its " +
          "surroundings, and the width of the ramp the new pixels are " +
          "blended back over"
      )
    )
  )

  /** What the picture shows, which changes nothing about the job: out of the
    * form altogether, under the line that says what the job would cost, since
    * that is the rest of what is read off the picture (François, 2026-09-21).
    *
    * The reset is always there, disabled while the grid is where it started: it
    * is also the one thing that says the grid can be moved at all, the drag
    * itself being invisible until a line is under the pointer.
    */
  def gridControls: HtmlElement = div(
    cls := "post-foot-controls",
    checkField(
      showTileGrid,
      "show the grid",
      "draw the tiles over the picture — where each pass of the model will " +
        "run, and the bands where neighbours are blended together. Drag one " +
        "of its lines to move the whole grid. While a job runs on this image " +
        "it is that job's grid that is drawn, filling in as its tiles are done"
    ),
    button(
      cls := "button is-small",
      title := "put the tiles back where they fall on their own, which is " +
        "the fewest of them",
      disabled <-- gridOffset.signal.map(_.isCentred),
      child.text <-- gridOffset.signal.map(offset =>
        if (offset.isCentred) "↺ reset the grid"
        else s"↺ reset the grid (${offset.x}, ${offset.y})"
      ),
      onClick --> (_ => gridOffset.set(TileOffset(0, 0)))
    )
  )

  /** The last line: what this job would work on and what it costs, as
    * `describe` words it, with the way out of a selection beside it.
    */
  def costLine(describe: Signal[TilePlan => String]): Signal[Node] =
    plan
      .combineWith(describe)
      .map((estimate, words) =>
        estimate.fold(emptyNode: Node)(cost =>
          span(
            words(cost),
            cost.selection.map(_ =>
              button(
                cls := "button is-small ml-2",
                "clear selection",
                onClick --> (_ =>
                  viewed.update(_.map(_.copy(selection = None)))
                )
              )
            )
          )
        )
      )
}

object TileAreaFields {

  /** How a task cuts the picture, the one thing that differs between them. */
  enum Cut {

    /** A redraw or an edit: tiles in the picture's own pixels, over the whole
      * of it or the window around a box, aligned to the model's multiple.
      */
    case Selection(sizeMultiple: Int)

    /** An upscale: the whole picture only, its tiles laid out in the target's
      * pixels, `targetPerPixel` of them to one of the picture's.
      */
    case Whole(tiling: UpscaleTiling, targetPerPixel: Double)
  }

  /** A redraw's or an edit's cut: the multiple the chosen configuration's model
    * aligns its sides up to — none until the list has arrived and one is
    * chosen, which is also when there is nothing to plan. No fallback: a tile
    * count on a guessed multiple would be a wrong number shown as a right one.
    */
  def selectionCut(
      configurations: Signal[List[ConfigurationOption]],
      chosenConfiguration: Signal[String]
  ): Signal[Option[Cut]] =
    configurations
      .combineWith(chosenConfiguration)
      .map((options, chosen) =>
        options
          .find(_.id == chosen)
          .map(option => Cut.Selection(option.sizeMultiple): Cut)
      )
      .distinct

  /** What the fields as they stand would run: the selection if there is one,
    * the window its tiles cover, and how many tiles that is — one inference
    * each.
    */
  case class TilePlan(
      selection: Option[ImageRegion],
      window: ImageRegion,
      tiles: Int
  )

  /** What is worked on and in how many tiles, the start of every cost line. */
  def extentOf(plan: TilePlan): String = {
    val what = plan.selection.fold(
      s"whole image ${plan.window.width}×${plan.window.height}"
    )(selection =>
      s"selection ${selection.width}×${selection.height} → window " +
        s"${plan.window.width}×${plan.window.height}"
    )
    val tiles =
      if (plan.tiles == 1) "1 tile" else s"${plan.tiles} tiles"
    s"$what · $tiles"
  }
}
