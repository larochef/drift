package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The fields a run is shaped by day to day: the mode tabs, both prompts with
  * Generate right under them, size, steps and CFG — with a two-expert model's
  * high-noise pair beneath —, frames and FPS for a video, and the seed.
  */
class CoreFields(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    onSwitchMode: String => Unit,
    onGenerate: () => Unit
) extends Component {
  import state.*
  import FormFields.{field, numberField}

  /** The seed, with the tick that has drift roll a fresh one each run. */
  private def seedField: HtmlElement =
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Seed"),
      div(
        cls := "field has-addons mb-0",
        div(
          cls := "control",
          input(
            // A random seed runs to ten digits.
            cls := "input is-small is-digits",
            styleAttr := "--digits: 11;",
            typ := "number",
            disabled <-- randomSeedVar.signal,
            placeholder := "random",
            value <-- seedVar.signal,
            onInput.mapToValue --> seedVar
          )
        ),
        div(
          cls := "control",
          label(
            cls := "button is-small checkbox text-primary",
            input(
              typ := "checkbox",
              checked <-- randomSeedVar.signal,
              onChange.mapToChecked --> randomSeedVar
            ),
            " Random",
            title := "drift picks a fresh seed for each generation and " +
              "shows it here; untick to keep the last one"
          )
        )
      )
    )

  lazy val element: HtmlElement = {
    val limits = capabilities.limits
    div(
      if (capabilities.supportedModes.size > 1)
        div(
          cls := "tabs is-boxed is-small mb-3",
          ul(
            capabilities.supportedModes.map(m =>
              li(
                cls("is-active") <-- mode.signal.map(_ == m),
                a(
                  GenerationFormState.modeLabel(m),
                  onClick --> (_ => onSwitchMode(m))
                )
              )
            )
          )
        )
      else emptyNode,
      field(
        "Prompt",
        textArea(
          cls := "textarea",
          rows := 4,
          placeholder := "What to generate",
          value <-- promptVar.signal,
          onInput.mapToValue --> promptVar
        )
      ),
      field(
        "Negative prompt",
        textArea(
          cls := "textarea",
          rows := 2,
          value <-- negativePromptVar.signal,
          onInput.mapToValue --> negativePromptVar
        )
      ),
      // Right under the prompts, where the loop happens (François,
      // 2026-09-08); the parameters below are for the occasional adjustment.
      div(
        cls := "field",
        button(
          cls := "button is-primary",
          disabled <-- promptVar.signal.map(_.trim.isEmpty),
          "✨ Generate",
          onClick --> (_ => onGenerate())
        )
      ),
      // Size, steps, CFG and seed each as wide as what they hold, side by
      // side and wrapping, instead of two per row across the whole form
      // (François, 2026-09-29).
      div(
        cls := "form-compact-row",
        numberField(
          "Width",
          widthVar,
          Some(limits.minWidth),
          Some(limits.maxWidth),
          digits = Some(4)
        ),
        numberField(
          "Height",
          heightVar,
          Some(limits.minHeight),
          Some(limits.maxHeight),
          digits = Some(4)
        ),
        numberField("Steps", stepsVar, Some(1), digits = Some(3))
          .amend(onInput --> (_ => touch("steps"))),
        // Two digits and a decimal: 7.5, 3.5.
        numberField("CFG", cfgVar, digits = Some(4))
          .amend(onInput --> (_ => touch("cfg"))),
        seedField
      ),
      // The high-noise expert's own steps and CFG sit directly under the
      // low-noise pair they mirror (François, 2026-09-11): on wan 2.2 both
      // are tuned together, so separating them hides half the answer to "how
      // many steps is this run".
      //
      // A two-expert model announces itself by reporting high-noise defaults;
      // nothing else has a second pass to configure. Video only -
      // `ImageGenerationParameters` carries no high-noise block, so the
      // mode's defaults never report one for an image model.
      child <-- mode.signal.map { m =>
        capabilities.defaultsByMode
          .get(m)
          .flatMap(_.highNoiseSampleParams) match {
          case None    => emptyNode
          case Some(_) =>
            div(
              cls := "form-compact-row",
              numberField(
                "High-noise steps",
                highNoiseStepsVar,
                Some(1),
                digits = Some(3)
              ).amend(onInput --> (_ => touch("highNoiseSteps"))),
              numberField("High-noise CFG", highNoiseCfgVar, digits = Some(4))
                .amend(onInput --> (_ => touch("highNoiseCfg")))
            )
        }
      },
      child <-- mode.signal.map {
        case "vid_gen" =>
          div(
            cls := "form-compact-row",
            numberField("Frames", videoFramesVar, Some(1), digits = Some(4)),
            numberField("FPS", fpsVar, Some(1), digits = Some(3))
          )
        case _ => emptyNode
      }
    )
  }
}
