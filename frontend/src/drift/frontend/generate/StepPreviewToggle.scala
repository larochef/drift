package drift.frontend.generate

import drift.frontend.components.Component
import drift.frontend.services.GenerationService

import com.raquo.laminar.api.L.*

/** Whether a generation shows what the model has so far
  * (`specs/54-step-preview.md`): a checkbox over the results, ticked unless the
  * user said otherwise, for images and videos alike. Unticked, the next
  * generations are submitted without asking for a preview — the server draws
  * none — and the one under way hides its own.
  */
class StepPreviewToggle(service: GenerationService) extends Component {

  lazy val element: HtmlElement = label(
    cls := "checkbox is-size-7 mb-2 is-block",
    title := "Show the model's estimate of the picture or the video after " +
      "each step, under the progress bar",
    input(
      typ := "checkbox",
      checked <-- service.preferences.map(_.stepPreview),
      onChange.mapToChecked --> (shown => service.showStepPreview(shown))
    ),
    " Preview while generating"
  )
}
