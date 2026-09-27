package drift.frontend.pages.gallery

import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.TileAreaFields.TilePlan
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Where a tiled task's tiles fall (`specs/27-redraw.md`,
  * `specs/39-seamless-edit.md`): the tile, window and margin fields, the grid
  * drawn over the picture, and the plan read off the box dragged on it — one
  * piece redraw and edit each hold their own of, since both cut the picture the
  * same way.
  *
  * Both panels stay built while hidden, so only the one on screen publishes its
  * numbers to the picture: `active` says which, and becoming active publishes
  * them at once.
  */
class TileAreaFields(
    /** The configurations the task offers, and the one chosen — its model's
      * size multiple is what the tiles are aligned to.
      */
    configurations: Signal[List[ConfigurationOption]],
    chosenConfiguration: Signal[String],
    /** The image on screen and the box drawn on it. */
    viewed: Var[Option[ViewedImage]],
    /** Where the numbers go for the picture to count and draw tiles with. */
    geometry: Var[RedrawGeometry],
    showTileGrid: Var[Boolean],
    gridOffset: Var[TileOffset],
    active: Signal[Boolean]
) {

  private val tileVar = Var("1280")
  private val windowVar = Var("1024")
  private val marginVar = Var("64")

  private def number(state: Var[String], fallback: Int): Int =
    state.now().trim.toIntOption.getOrElse(fallback)

  def tileSize: Int = number(tileVar, 1280)
  def minimumWindowSide: Int = number(windowVar, 1024)
  def selectionMargin: Int = number(marginVar, 64)
  def region: Option[ImageRegion] = viewed.now().flatMap(_.selection)
  def gridOffsetX: Int = gridOffset.now().x
  def gridOffsetY: Int = gridOffset.now().y

  /** Whether a box is drawn on the picture — what the button is named after. */
  val hasSelection: Signal[Boolean] =
    viewed.signal.map(_.exists(_.selection.isDefined)).distinct

  /** The multiple the chosen configuration's model aligns its sides up to —
    * none until the list has arrived and one is chosen, which is also when
    * there is nothing to plan. No fallback: a tile count on a guessed multiple
    * would be a wrong number shown as a right one.
    */
  private val architectureMultiple: Signal[Option[Int]] =
    configurations
      .combineWith(chosenConfiguration)
      .map((options, chosen) =>
        options.find(_.id == chosen).map(_.sizeMultiple)
      )
      .distinct

  /** The numbers the picture is judged by, as the fields stand. */
  private val currentGeometry: Signal[Option[RedrawGeometry]] =
    tileVar.signal
      .combineWith(
        marginVar.signal,
        windowVar.signal,
        architectureMultiple,
        gridOffset.signal
      )
      .map((tile, margin, window, multiple, offset) =>
        multiple.map(aligned =>
          RedrawGeometry(
            tileSize = tile.trim.toIntOption.getOrElse(1280),
            margin = margin.trim.toIntOption.getOrElse(64),
            minimumWindow = window.trim.toIntOption.getOrElse(1024),
            sizeMultiple = aligned,
            offsetX = offset.x,
            offsetY = offset.y
          )
        )
      )
      .distinct

  /** The picture gets them as they are typed, while this task is the one on
    * screen: the box counts its tiles, sticks to their boundaries and draws the
    * grid from the very numbers this panel plans with.
    */
  val publishGeometry: Modifier[HtmlElement] =
    currentGeometry.combineWith(active).changes.collect {
      case (Some(published), true) => published
    } --> geometry

  /** What the fields as they stand would run on the picture on screen. */
  val plan: Signal[Option[TilePlan]] =
    viewed.signal.combineWith(currentGeometry).map { (shown, current) =>
      for {
        picture <- shown
        sticky <- current
      } yield TilePlan(
        picture.selection,
        sticky.areaFor(picture.selection, picture.width, picture.height),
        sticky.layoutFor(picture.selection, picture.width, picture.height).size
      )
    }

  /** What is worked on, and how it is cut up to fit through the model. */
  def areaGroup: HtmlElement = group(
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
