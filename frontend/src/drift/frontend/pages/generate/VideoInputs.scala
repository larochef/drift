package drift.frontend.pages.generate

import drift.frontend.components.{Component, FoldedSection}

import com.raquo.laminar.api.L.*

/** A video model's guides (MiniMax H3): media the video holds at a frame, a
  * negative index counting from the end. Rows are keyed so typing a frame index
  * updates the row in place instead of rebuilding the list, which would take
  * the input's focus away mid-number.
  */
class GuidesPicker(state: GenerationFormState) extends Component {
  import state.guidesVar

  /** One guide: its medium never changes under its key, only the frame. */
  private def row(
      key: Int,
      initial: GuideInput,
      guide: Signal[GuideInput]
  ): HtmlElement =
    thumbnail(
      initial.media,
      () => guidesVar.update(_.filterNot(_.key == key)),
      span(cls := "is-size-7 text-secondary ml-1", "frame"),
      input(
        cls := "input is-small is-digits",
        styleAttr := "--digits: 4;",
        typ := "number",
        title := "Frame index (negative counts from the end)",
        value <-- guide.map(_.frameIndex),
        onInput.mapToValue --> (typed =>
          guidesVar.update(
            _.map(g => if (g.key == key) g.copy(frameIndex = typed) else g)
          )
        )
      )
    )

  lazy val element: HtmlElement =
    div(
      cls := "field",
      label(
        cls := "label text-primary is-small",
        "Guides (held at a frame; -1 is the last)"
      ),
      div(
        cls := "control",
        children <-- guidesVar.signal.split(_.key)((key, initial, guide) =>
          row(key, initial, guide)
        ),
        // A new guide lands on the last frame: the first is what a start
        // image is for.
        fileInput(
          "Add a guide",
          MediaAccept.AnyMedia,
          media => state.addGuide(media, "-1")
        )
      )
    )
}

/** The control inputs of a video model with a ControlNet (MiniMax H3's Fun
  * ControlNet): the control video, its strength and — folded, as they rarely
  * move — the fraction of the steps it applies over, then an optional mask
  * whose white is regenerated over an optional source video.
  */
class ControlVideoInputs(state: GenerationFormState) extends Component {
  import state.*
  import FormFields.numberField

  private val advancedOpen = Var(false)

  lazy val element: HtmlElement =
    div(
      singleMediaPicker(
        "Control video (pose, depth, edges…)",
        controlVideoVar,
        MediaAccept.Videos
      ),
      child <-- controlVideoVar.signal.map(_.isDefined).distinct.map {
        case false => emptyNode
        case true  =>
          div(
            numberField("Control strength (default 1)", controlStrengthVar),
            FoldedSection(
              Val("Control range and mask"),
              advancedOpen,
              keepBody = true,
              body = div(
                numberField(
                  "Control start (fraction of the steps, default 0)",
                  controlStartVar
                ),
                numberField(
                  "Control end (fraction of the steps, default 1)",
                  controlEndVar
                ),
                singleMediaPicker(
                  "Mask (white is regenerated)",
                  controlMaskVar,
                  MediaAccept.ImagesAndVideos
                ),
                singleMediaPicker(
                  "Source video (under the mask)",
                  sourceVideoVar,
                  MediaAccept.Videos
                )
              )
            ).element
          )
      }
    )
}
