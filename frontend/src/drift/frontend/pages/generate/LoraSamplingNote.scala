package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Under the LoRA picker: what the selected LoRAs set in the form
  * (`specs/49-lora-sampling-settings.md`), a way to put those values back over
  * what the fields hold, and a word when the steps are fewer than a LoRA was
  * made for. Nothing is refused: the values are defaults.
  */
class LoraSamplingNote(
    state: GenerationFormState,
    /** The selected LoRAs that carry settings, in selection order. */
    loras: Signal[List[Lora]],
    onApply: () => Unit
) extends Component {

  private val tooFewSteps: Signal[Option[Int]] =
    loras
      .map(LoraSampling.of(_).steps)
      .combineWith(state.stepsVar.signal, state.sigmasVar.signal)
      .map { (wanted, typed, sigmas) =>
        val steps = GenerationFormState
          .sigmasOf(sigmas)
          .filter(_.nonEmpty)
          .map(GenerationFormState.sigmaSteps)
          .orElse(typed.trim.toIntOption)
        wanted.filter(minimum => steps.exists(_ < minimum))
      }

  lazy val element: HtmlElement =
    div(
      child <-- loras.map {
        case Nil      => emptyNode
        case carrying =>
          div(
            cls := "mb-2",
            carrying.map(lora =>
              p(
                cls := "help text-secondary",
                s"${lora.label} sets ${lora.sampling.summary.mkString(" · ")}"
              )
            ),
            button(
              cls := "button is-small mt-1",
              "Apply the LoRA settings",
              title := "Puts the values these LoRAs set in the form again, " +
                "over what those fields hold. They are defaults: change " +
                "them as you like.",
              onClick --> (_ => onApply())
            )
          )
      },
      child <-- tooFewSteps.map {
        case Some(minimum) =>
          p(
            cls := "help is-danger",
            s"Fewer steps than a selected LoRA was made for ($minimum): " +
              "expect a broken image."
          )
        case None => emptyNode
      }
    )
}
