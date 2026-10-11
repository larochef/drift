package drift.frontend.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Highres fix (`specs/10-generation-time-upscaling.md`), shown only in image
  * mode where the capabilities report the feature: generate at the form's size,
  * upscale by the chosen mode, refine with a second pass. The modes are the
  * live server's own — latent modes and plain filters; upscaling with a model
  * is the gallery's Upscale task (`specs/15-post-hoc-resize.md`).
  */
class HiresSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.*
  import FormFields.{field, numberField}
  import HiresParameters.{isBuiltIn, isLatentMode}

  /** What a closed section is doing, or None when it changes nothing about the
    * run — an off block costs nothing, so it has nothing to announce.
    */
  private val summary: Signal[Option[String]] =
    hiresEnabledVar.signal
      .combineWith(hiresUpscalerVar.signal, hiresScaleVar.signal)
      .map { (on, upscaler, scale) =>
        Option.when(on)(
          s"on · ×$scale" +
            (if (upscaler.isEmpty) "" else s" · $upscaler")
        )
      }

  /** Picking an upscaler fills the field that follows from its kind — the
    * second-pass denoising: a latent upscale is blurry and needs a heavy
    * refine, a filtered picture a lighter one. The user can still override
    * afterwards; this only moves the value the moment the kind changes.
    */
  private def selectUpscaler(name: String): Unit = {
    hiresUpscalerVar.set(name)
    if (name.nonEmpty)
      hiresDenoisingStrengthVar.set(
        (if (isLatentMode(name)) 0.65 else 0.5).toString
      )
  }

  /** The upscaler dropdown, grouped so the two kinds read apart: the
    * latent-space modes and the plain pixel filters, as the live server names
    * them.
    */
  private def upscalerSelectField(names: List[String]): HtmlElement = {
    val (latentModes, filters) = names.partition(isLatentMode)
    def optionsOf(values: List[String]) = values.map(name =>
      option(
        value := name,
        selected <-- hiresUpscalerVar.signal.map(_ == name),
        name
      )
    )
    def group(labelText: String, values: List[String]) =
      if (values.isEmpty) emptyNode
      else optGroup(labelAttr := labelText, optionsOf(values))
    field(
      "Upscaler",
      div(
        cls := "select is-small is-fullwidth",
        select(
          onChange.mapToValue --> (name => selectUpscaler(name)),
          option(
            value := "",
            selected <-- hiresUpscalerVar.signal.map(_.isEmpty),
            "(server default)"
          ),
          group("Latent modes (upscale before decode)", latentModes),
          group("Plain filters (fast, soft)", filters)
        )
      )
    )
  }

  lazy val element: HtmlElement = {
    val features =
      capabilities.featuresByMode.getOrElse(currentMode, Map.empty)
    if (currentMode != "img_gen" || !features.getOrElse("hires", false)) div()
    else {
      // No "None": "no upscaling" is what the checkbox is for.
      val upscalerNames = capabilities.upscalers.map(_.name).filter(isBuiltIn)
      CollapsibleSection(
        "Hires",
        summary,
        div(
          label(
            cls := "checkbox label text-primary is-small mb-1",
            input(
              typ := "checkbox",
              cls := "mr-1",
              checked <-- hiresEnabledVar.signal,
              onChange.mapToChecked --> hiresEnabledVar
            ),
            "Highres fix (upscale + refine)"
          ),
          child <-- hiresEnabledVar.signal.map {
            case false => emptyNode
            case true  =>
              div(
                upscalerSelectField(upscalerNames),
                div(
                  cls := "columns is-mobile",
                  div(cls := "column", numberField("Scale", hiresScaleVar)),
                  div(
                    cls := "column",
                    numberField(
                      "2nd-pass steps (0 = auto)",
                      hiresStepsVar,
                      Some(0)
                    )
                  )
                ),
                div(
                  cls := "columns is-mobile",
                  div(
                    cls := "column",
                    numberField(
                      "Target width (overrides scale)",
                      hiresTargetWidthVar,
                      Some(0)
                    )
                  ),
                  div(
                    cls := "column",
                    numberField("Target height", hiresTargetHeightVar, Some(0))
                  )
                ),
                numberField(
                  "Denoising strength (0-1)",
                  hiresDenoisingStrengthVar
                ),
                p(
                  cls := "is-size-7 text-secondary",
                  "Denoising is set from the upscaler you pick — heavy for a " +
                    "latent mode, lighter for a filter — tune from there. To " +
                    "upscale with a model, use the Upscale task on the result " +
                    "in the gallery."
                )
              )
          }
        )
      ).element
    }
  }
}
