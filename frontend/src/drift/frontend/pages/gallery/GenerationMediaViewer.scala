package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.pages.gallery.GenerationCard.*
import drift.frontend.pages.gallery.GenerationMediaViewer.*
import drift.frontend.services.ApiClient
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** A generation's output at full size — an image, or a player for video — with
  * a strip to pick through a batch and, on a derived entry, a toggle to compare
  * it with its original at matched zoom.
  *
  * The box drawn on an image is the part a redraw repaints
  * (`specs/27-redraw.md`). It can be drawn, moved by its middle, resized by any
  * corner, and it says as it moves how many tiles it would cost — a tile is a
  * whole pass of the model, so that number is the price. Its sides stick to the
  * lengths where the price changes and cross only when the drag insists
  * (François, 2026-09-19). A click clears it, and so does every change of
  * output, since the box belongs to the picture it was drawn on.
  */
class GenerationMediaViewer(
    generation: Generation,
    /** Which output of a batch is shown; the post-processing rows act on it. */
    selectedIndex: Var[Int],
    /** What to show: the chosen output, or the original where the detail's
      * compare toggle asks for it. The detail owns that choice, since its
      * header offers it (`specs/12-gallery.md`).
      */
    shown: Signal[Option[GenerationOutput]],
    /** Whether a box may be drawn on it: not on the original, which a redraw
      * does not act on.
      */
    selectable: Signal[Boolean],
    /** The image on screen and the box drawn on it, for the redraw panel. */
    viewed: Var[Option[ViewedImage]],
    /** What a redraw would make of that box, from the redraw panel's fields —
      * what the drag moves, and what the box sticks to.
      */
    geometry: Signal[RedrawGeometry],
    /** The tiles to draw over the picture, in its own pixels, each with what
      * has become of it: a running job's are done, running or still to come; a
      * task's are all still to come (`specs/15-post-hoc-resize.md`).
      */
    tiles: Signal[List[TilePaint]],
    /** What the job on the picture has made so far, scaled for the screen: laid
      * over the picture, so the result appears as its tiles finish.
      */
    jobPicture: Signal[Option[String]],
    /** Whether the tiles a redraw would run are drawn over the picture — the
      * redraw panel's own checkbox.
      */
    showTiles: Signal[Boolean],
    /** How far that grid is shifted. Dragging one of its lines moves it, so the
      * cuts can be placed away from a face — this is where that drag lands
      * (François, 2026-09-20).
      */
    gridOffset: Var[TileOffset]
) extends Component {

  /** The size of the file behind the preview, asked for when one is shown. The
    * page displays an image scaled to the screen — an 8192² upscale is a
    * quarter of a gigabyte decoded, and froze the view for seconds (François,
    * 2026-09-19) — but a selection has to be recorded in the pixels of the file
    * that will be repainted, not of the copy on screen.
    */
  private val sizeOf = ApiClient.stream(getOutputSize)

  /** What a press, a move and a release do to the box and the grid. */
  private val pointer = PicturePointer(viewed, geometry, showTiles, gridOffset)

  /** The box, its handles and what it would cost, over the picture. */
  private val overlay: Signal[Option[HtmlElement]] =
    viewed.signal.combineWith(geometry).map { (shown, sticky) =>
      for {
        image <- shown
        region <- image.selection
      } yield {
        val tiles = sticky.tilesFor(region, image.width, image.height)
        val window = sticky.windowFor(region, image.width, image.height)
        div(
          cls := "gallery-selection",
          styleAttr :=
            s"left: ${percent(region.x, image.width)}; " +
              s"top: ${percent(region.y, image.height)}; " +
              s"width: ${percent(region.width, image.width)}; " +
              s"height: ${percent(region.height, image.height)};",
          span(
            cls := "gallery-selection-label",
            s"${region.width}×${region.height} · " +
              (if (tiles == 1) "1 tile" else s"$tiles tiles") +
              s" · repaints ${window.width}×${window.height}"
          ),
          PicturePointer.Grip.corners
            .map(corner => div(cls := s"gallery-grip ${corner.cssClass}"))
        )
      }
    }

  /** The tiles the job would run, drawn over the picture: the whole image's
    * tiling, or the window's once a box narrows the pass to it. A tile is one
    * pass of the model, and the brighter bands are the overlaps two
    * neighbouring passes are feathered together over, so the grid says where
    * the seams will fall before anything runs (François, 2026-09-20).
    */
  private val tileGrid: Signal[Option[HtmlElement]] =
    viewed.signal.combineWith(tiles, showTiles).map { (shown, drawn, on) =>
      Option
        .when(on)(shown)
        .flatten
        .filter(image => image.width > 0 && image.height > 0)
        .map { image =>
          div(
            cls := "gallery-tiles",
            drawn
              .map(painted =>
                div(
                  cls := "gallery-tile",
                  cls := painted.state.styleClass,
                  styleAttr :=
                    s"left: ${percent(painted.area.x, image.width)}; " +
                      s"top: ${percent(painted.area.y, image.height)}; " +
                      s"width: ${percent(painted.area.width, image.width)}; " +
                      s"height: ${percent(painted.area.height, image.height)};"
                )
              )
          )
        }
    }

  /** The job's picture over the source, one element for as long as there is
    * one: a new tile changes its address, and the browser keeps showing the
    * last one until the next has arrived.
    */
  private def jobPictureLayer: Modifier[HtmlElement] =
    child <-- jobPicture.splitOption[Node](
      (_, url) =>
        img(
          cls := "gallery-job-picture",
          src <-- url,
          alt := "",
          onDragStart --> (_.preventDefault())
        ),
      emptyNode
    )

  private def media(
      output: GenerationOutput,
      selectable: Boolean
  ): HtmlElement =
    if (GenerationMediaViewer.isVideo(output))
      videoTag(
        src := output.url,
        controlsAttr := true,
        loopAttr := true,
        autoPlayAttr := true,
        playsInlineAttr := true,
        onMountCallback(_ => viewed.set(None))
      )
    else {
      val picture = img(
        src := GenerationMediaViewer.previewUrl(output),
        alt := RecordedParameters.titleOf(generation),
        // The browser's own image drag would take over the selection drag.
        onDragStart --> (_.preventDefault()),
        // The file's own size, not the preview's: the selection is in its
        // pixels. The preview's size stands in until the answer arrives.
        sizeOf((output.date, output.fileName)).recoverToTry --> Observer[
          scala.util.Try[OutputSize]
        ] { answer =>
          if (selectable)
            answer.toOption.foreach(size =>
              viewed.update(
                _.map(_.copy(width = size.width, height = size.height))
                  .orElse(Some(ViewedImage(size.width, size.height)))
              )
            )
        },
        onLoad --> { event =>
          val loaded = event.target.asInstanceOf[dom.html.Image]
          if (selectable && viewed.now().isEmpty)
            viewed.set(
              Some(ViewedImage(loaded.naturalWidth, loaded.naturalHeight))
            )
        }
      )
      // The wrapper is given the picture's own ratio, so it fits the box the
      // layout leaves *and* hugs the image exactly: the box drawn on it is
      // positioned in percentages of this element, and a letterboxed wrapper
      // would put the selection somewhere the image is not.
      val ratio = styleAttr <-- viewed.signal.map(
        _.filter(image => image.width > 0 && image.height > 0)
          .map(image => s"aspect-ratio: ${image.width} / ${image.height};")
          .getOrElse("")
      )
      if (!selectable)
        div(cls := "gallery-selectable", ratio, picture, jobPictureLayer)
      else
        div(
          cls := "gallery-selectable is-selecting",
          ratio,
          picture,
          jobPictureLayer,
          cls("is-on-grid") <-- pointer.overGrid,
          pointer.modifiers,
          child.maybe <-- tileGrid,
          child.maybe <-- overlay
        )
    }

  private def stripEntry(output: GenerationOutput, index: Int): HtmlElement = {
    val selectedClass =
      selectedIndex.signal.map(i => if (i == index) "is-selected" else "")
    if (GenerationMediaViewer.isVideo(output))
      videoTag(
        src := output.url,
        mutedAttr := true,
        cls <-- selectedClass,
        onClick --> (_ => selectedIndex.set(index))
      )
    else
      img(
        src := output.url,
        cls <-- selectedClass,
        onClick --> (_ => selectedIndex.set(index))
      )
  }

  lazy val element: HtmlElement = div(
    cls := "gallery-media-column",
    // The box was drawn on one picture; showing another is not a request to
    // repaint the same place on it.
    pointer.clearOnChange(selectedIndex.signal.changes, shown.changes),
    div(
      cls := "gallery-media-row",
      // The batch down the side, where it costs no height and cannot be
      // scrolled out of sight: not knowing a generation was a batch is how a
      // result gets missed (François, 2026-09-19).
      if (generation.outputs.size > 1)
        div(
          cls := "gallery-strip is-vertical",
          // How many there are, so each thumbnail can take its share of the
          // height: the whole batch has to be visible at once, whatever the
          // window (François, 2026-09-19).
          styleAttr := s"--strip-count: ${generation.outputs.size};",
          generation.outputs.zipWithIndex.map(stripEntry)
        )
      else emptyNode,
      div(
        cls := "gallery-detail-media",
        child <-- shown
          .combineWith(selectable)
          .distinct
          .map { (output, canSelect) =>
            output
              .map(media(_, selectable = canSelect))
              .getOrElse(p(cls := "text-secondary", "No output file recorded."))
          }
      )
    )
  )
}

object GenerationMediaViewer {

  def isVideo(output: GenerationOutput): Boolean =
    output.mimeType.startsWith("video/")

  /** What the page actually loads for an image: a copy scaled to the screen.
    * Videos are served as they are — there is nothing to scale.
    */
  def previewUrl(output: GenerationOutput): String =
    if (isVideo(output)) output.url else s"${output.url}/preview"

  /** A length as a percentage of the picture: the browser scales the image, so
    * the box is placed in the picture's own terms.
    */
  def percent(value: Int, total: Int): String =
    if (total <= 0) "0%" else f"${value * 100.0 / total}%.3f%%"

}
