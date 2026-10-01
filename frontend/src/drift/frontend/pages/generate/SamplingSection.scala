package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Sampler, scheduler, flow shift and custom sigmas. Folded: the model's own
  * defaults are usually right, and what does get changed daily is steps and
  * CFG, which stay above.
  */
class SamplingSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.{flowShiftVar, samplerVar, schedulerVar, sigmasVar}
  import FormFields.{field, numberField, selectField}

  /** Only a *deviation* is worth announcing here: a sampler seeded straight
    * from the session's own defaults is what a closed section already implies,
    * and repeating it on every panel is noise rather than a warning.
    */
  private val summary: Signal[Option[String]] = {
    val defaults = GenerationFormState.cleanse(
      capabilities.defaultsByMode
        .getOrElse(currentMode, GenerationDefaults())
        .sampleParams
    )
    samplerVar.signal
      .combineWith(schedulerVar.signal, sigmasVar.signal, flowShiftVar.signal)
      .map { (sampler, scheduler, sigmas, flowShift) =>
        val changed = List(
          Option.when(sampler != defaults.sampleMethod.getOrElse(""))(sampler),
          Option.when(scheduler != defaults.scheduler.getOrElse(""))(scheduler)
        ).flatten.map(value => if (value.isEmpty) "model default" else value) ++
          Option.when(
            flowShift.trim.toDoubleOption != defaults.flowShift
          )(
            if (flowShift.trim.isEmpty) "model's flow shift"
            else s"flow shift ${flowShift.trim}"
          ) ++
          Option.when(sigmas.trim.nonEmpty)("custom sigmas")
        Option.when(changed.nonEmpty)(changed.mkString(" · "))
      }
  }

  lazy val element: HtmlElement =
    CollapsibleSection(
      "Sampling",
      summary,
      div(
        div(
          cls := "columns is-mobile",
          div(
            cls := "column",
            selectField("Sampler", samplerVar, capabilities.samplers)
          ),
          div(
            cls := "column",
            selectField("Scheduler", schedulerVar, capabilities.schedulers)
          )
        ),
        // A turbo LoRA wants the steps nearer the noisy end than the model's
        // own schedule puts them: a higher shift does that at any step count.
        div(
          cls := "form-compact-row",
          numberField("Flow shift", flowShiftVar, digits = Some(4))
        ),
        p(
          cls := "help text-secondary",
          "Empty: the model's own. Higher spends more of the steps at high noise; a turbo LoRA usually wants about 3."
        ),
        // Or its exact noise levels, when its page lists them: they go here,
        // and replace the schedule, shift included.
        field(
          "Sigmas",
          input(
            cls := "input is-small",
            typ := "text",
            placeholder := "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25",
            value <-- sigmasVar.signal,
            onInput.mapToValue --> sigmasVar
          )
        ),
        p(
          cls := "help",
          cls("is-danger") <-- sigmasVar.signal.map(
            GenerationFormState.sigmasOf(_).isEmpty
          ),
          cls("text-secondary") <-- sigmasVar.signal.map(
            GenerationFormState.sigmasOf(_).isDefined
          ),
          child.text <-- sigmasVar.signal.map { text =>
            GenerationFormState.sigmasOf(text) match {
              case None      => "Numbers separated by commas."
              case Some(Nil) =>
                "Noise levels to step through in place of the scheduler's, as a turbo LoRA lists them. Empty: the model's schedule."
              case Some(sigmas) =>
                s"${GenerationFormState.sigmaSteps(sigmas)} steps, whatever Steps says; the scheduler is not used."
            }
          }
        )
      )
    ).element
}
