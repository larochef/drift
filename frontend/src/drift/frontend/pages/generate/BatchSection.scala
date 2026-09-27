package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** How many images one run makes (`specs/14`): the same recipe on that many
  * seeds, arriving as one generation with that many outputs. img_gen only — the
  * native video request has no batch count.
  */
class BatchSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.batchCountVar

  private val summary: Signal[Option[String]] =
    batchCountVar.signal.map(typed =>
      typed.trim.toIntOption.filter(_ > 1).map(n => s"$n images per run")
    )

  lazy val element: HtmlElement =
    if (currentMode != "img_gen") div()
    else
      CollapsibleSection(
        "Batch",
        summary,
        div(
          FormFields.numberField(
            s"Images per run (1–${capabilities.limits.maxBatchCount})",
            batchCountVar,
            Some(1),
            Some(capabilities.limits.maxBatchCount)
          ),
          p(
            cls := "is-size-7 text-secondary",
            "One run, several seeds: the images arrive together as one " +
              "generation, and a batch stays the same version of the recipe."
          )
        )
      ).element
}
