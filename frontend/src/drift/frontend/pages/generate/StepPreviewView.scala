package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The picture a generation is sampling, as the model's last step saw it
  * (`specs/54-step-preview.md`), under its progress bar
  * ([[drift.frontend.components.LogProgressView]]): a small picture enlarged,
  * captioned with the step it is the estimate of. One `<img>` whose source
  * changes, so a new step redraws it in place.
  *
  * Nothing before the first step, and nothing for an engine or a model that
  * draws none.
  */
class StepPreviewView(preview: Signal[Option[StepPreview]]) extends Component {

  private val shown = preview.distinct

  lazy val element: HtmlElement = div(
    cls := "step-preview mt-2",
    display <-- shown.map(found => if (found.isDefined) "" else "none"),
    img(
      src <-- shown.map(_.fold("")(_.url)),
      alt := "Preview of the picture being generated"
    ),
    p(
      cls := "is-size-7 text-secondary",
      child.text <-- shown.map(
        _.fold("")(picture =>
          s"Preview after step ${picture.step} of ${picture.steps}"
        )
      )
    )
  )
}
