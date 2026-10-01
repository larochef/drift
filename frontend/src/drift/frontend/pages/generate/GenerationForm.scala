package drift.frontend.pages.generate

import drift.frontend.components.{Component, LoraPicker}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The form for one session (`specs/08-inference-ui.md`), built from its
  * capabilities, never from hardcoded lists: the core fields, the LoRA picker
  * where the mode takes LoRAs, then the folded sections (`specs/14`).
  */
class GenerationForm(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    /** Built once by the panel, so a mode switch keeps its search. */
    loraPicker: LoraPicker,
    /** The selected LoRAs that carry sampling settings (`specs/49`). */
    samplingLoras: Signal[List[Lora]],
    onApplyLoraSampling: () => Unit,
    onSwitchMode: String => Unit,
    onGenerate: () => Unit
) extends Component {

  private def takesLoras(currentMode: String): Boolean =
    capabilities.featuresByMode
      .getOrElse(currentMode, Map.empty)
      .getOrElse("lora", false)

  lazy val element: HtmlElement = div(
    CoreFields(state, capabilities, onSwitchMode, onGenerate).element,
    child <-- state.mode.signal.map(m =>
      if (takesLoras(m))
        div(
          loraPicker.element,
          LoraSamplingNote(state, samplingLoras, onApplyLoraSampling).element
        )
      else div()
    ),
    // Everything below this point folds away (`specs/14`): one click to
    // reach, out of the way until it is wanted.
    child <-- state.mode.signal.map(m =>
      SamplingSection(state, capabilities, m).element
    ),
    child <-- state.mode.signal.map(m =>
      InputsSection(state, capabilities, m).element
    ),
    child <-- state.mode.signal.map(m =>
      HiresSection(state, capabilities, m).element
    ),
    child <-- state.mode.signal.map(m =>
      VaeTilingSection(state, capabilities, m).element
    ),
    child <-- state.mode.signal.map(m =>
      GuidanceSection(state, capabilities, m).element
    ),
    child <-- state.mode.signal.map(m =>
      BatchSection(state, capabilities, m).element
    )
  )
}
