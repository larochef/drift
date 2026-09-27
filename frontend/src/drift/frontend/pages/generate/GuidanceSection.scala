package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Image CFG, distilled guidance and skip-layer guidance (`specs/14`): seeded
  * from the session's defaults and untouched unless edited, so a run that never
  * opens this sends exactly what it always did.
  */
class GuidanceSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.{
    distilledGuidanceVar,
    imgCfgVar,
    slgLayerEndVar,
    slgLayersVar,
    slgLayerStartVar,
    slgScaleVar
  }
  import FormFields.{field, numberField}

  /** A deviation from the session's own guidance, as for sampling; active SLG
    * is always worth announcing — it costs a pass per listed layer.
    */
  private val summary: Signal[Option[String]] = {
    val defaults = capabilities.defaultsByMode
      .getOrElse(currentMode, GenerationDefaults())
      .sampleParams
      .guidance
    imgCfgVar.signal
      .combineWith(
        distilledGuidanceVar.signal,
        slgLayersVar.signal,
        slgScaleVar.signal
      )
      .map { (imgCfg, distilled, layers, scale) =>
        val changed =
          Option
            .when(imgCfg.trim != defaults.imgCfg.map(_.toString).getOrElse(""))(
              if (imgCfg.trim.isEmpty) "image CFG model's own"
              else s"image CFG ${imgCfg.trim}"
            )
            .toList :::
            Option
              .when(
                distilled.trim.toDoubleOption
                  .exists(_ != defaults.distilledGuidance)
              )(s"distilled ${distilled.trim}")
              .toList :::
            Option
              .when(
                layers.trim.nonEmpty &&
                  scale.trim.toDoubleOption.exists(_ != 0.0)
              )(s"SLG ${layers.trim} ×${scale.trim}")
              .toList
        Option.when(changed.nonEmpty)(changed.mkString(" · "))
      }
  }

  lazy val element: HtmlElement =
    CollapsibleSection(
      "Guidance & SLG",
      summary,
      div(
        div(
          cls := "columns is-mobile",
          div(
            cls := "column",
            numberField("Image CFG (blank = model's own)", imgCfgVar)
          ),
          div(
            cls := "column",
            numberField("Distilled guidance", distilledGuidanceVar)
          )
        ),
        field(
          "SLG layers (comma-separated)",
          input(
            cls := "input is-small",
            placeholder := "e.g. 7,8,9",
            value <-- slgLayersVar.signal,
            onInput.mapToValue --> slgLayersVar
          )
        ),
        div(
          cls := "columns is-mobile",
          div(cls := "column", numberField("SLG scale", slgScaleVar)),
          div(cls := "column", numberField("SLG start", slgLayerStartVar)),
          div(cls := "column", numberField("SLG end", slgLayerEndVar))
        )
      )
    ).element
}
