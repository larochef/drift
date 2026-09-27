package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Sampler and scheduler. Folded: the model's own defaults are usually right,
  * and what does get changed daily is steps and CFG, which stay above.
  */
class SamplingSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.{samplerVar, schedulerVar}
  import FormFields.selectField

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
    samplerVar.signal.combineWith(schedulerVar.signal).map {
      (sampler, scheduler) =>
        val changed = List(
          Option.when(sampler != defaults.sampleMethod.getOrElse(""))(sampler),
          Option.when(scheduler != defaults.scheduler.getOrElse(""))(scheduler)
        ).flatten.map(value => if (value.isEmpty) "model default" else value)
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
        )
      )
    ).element
}
