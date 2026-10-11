package drift.frontend.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The picture a generation is sampling, as the model's last step saw it
  * (`specs/54-step-preview.md`), under its progress bar
  * ([[drift.frontend.components.LogProgressView]]): a small picture enlarged,
  * captioned with the step it is the estimate of. One `<img>` whose source
  * changes, so a new step redraws it in place. A video's is a GIF, which the
  * `<img>` plays in a loop.
  *
  * A video's preview can be paused — the first steps' are a flicker more than a
  * clip. An `<img>` cannot stop a GIF, so a paused preview is the frame it was
  * on, drawn on a canvas in its place; each later step then shows as its first
  * frame, still. `paused` outlives the generation: paused once, the next
  * videos' previews come paused too.
  *
  * Nothing before the first step, and nothing for an engine or a model that
  * draws none.
  */
class StepPreviewView(
    preview: Signal[Option[StepPreview]],
    paused: Var[Boolean]
) extends Component {

  private val shown = preview.distinct

  /** Whether the preview is a clip, the only kind there is anything to pause
    * in.
    */
  private val moving = shown.map(_.exists(_.mimeType == "image/gif")).distinct

  private val still = moving.combineWith(paused.signal).map(_ && _).distinct

  private val picture = img(
    src <-- shown.map(_.fold("")(_.url)),
    alt := "Preview of what is being generated",
    // a step that arrives while paused is held on its first frame
    onLoad --> (_ => if (paused.now()) hold()),
    display <-- still.map(held => if (held) "none" else "")
  )

  private val held = canvasTag(
    display <-- still.map(held => if (held) "" else "none")
  )

  /** Draws the frame the picture is on onto the canvas. */
  private def hold(): Unit = {
    val image = picture.ref
    if (image.naturalWidth > 0) {
      val canvas = held.ref
      canvas.width = image.naturalWidth
      canvas.height = image.naturalHeight
      canvas
        .getContext("2d")
        .asInstanceOf[dom.CanvasRenderingContext2D]
        .drawImage(image, 0, 0)
    }
  }

  lazy val element: HtmlElement = div(
    cls := "step-preview mt-2",
    // a clip is coarser than a picture's preview, and shown smaller
    cls("is-clip") <-- moving,
    display <-- shown.map(found => if (found.isDefined) "" else "none"),
    picture,
    held,
    div(
      cls := "is-flex is-align-items-center mt-1",
      p(
        cls := "is-size-7 text-secondary is-flex-grow-1",
        child.text <-- shown.map(
          _.fold("")(picture =>
            s"Preview after step ${picture.step} of ${picture.steps}"
          )
        )
      ),
      button(
        cls := "button is-small is-light",
        typ := "button",
        display <-- moving.map(clip => if (clip) "" else "none"),
        title <-- paused.signal.map(stopped =>
          if (stopped) "Play the preview" else "Hold the preview on one frame"
        ),
        child.text <-- paused.signal.map(stopped =>
          if (stopped) "▶ Play" else "⏸ Pause"
        ),
        onClick --> { _ =>
          if (!paused.now()) hold()
          paused.update(!_)
        }
      )
    )
  )
}
