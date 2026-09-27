package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Highres fix (`specs/10-generation-time-upscaling.md`), shown only in image
  * mode where the capabilities report the feature: generate at the form's size,
  * upscale by the chosen upscaler, refine with a second pass. The upscaler list
  * is the live server's scan — built-ins plus the models found in
  * `--hires-upscalers-dir` at launch.
  */
class HiresSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.*
  import FormFields.{field, numberField}

  private def isLatentMode(name: String): Boolean = name.startsWith("Latent")
  private def isPlainFilter(name: String): Boolean =
    name == "Lanczos" || name == "Nearest"

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

  /** Picking an upscaler fills the fields that sensibly follow from its kind —
    * chiefly the second-pass denoising, which differs sharply: a model upscaler
    * already adds detail so it wants a light refine, a latent upscale is blurry
    * and needs a heavy one. The user can still override afterwards; this only
    * moves the value the moment the kind changes.
    */
  private def selectUpscaler(name: String): Unit = {
    hiresUpscalerVar.set(name)
    if (name.nonEmpty)
      hiresDenoisingStrengthVar.set(
        (if (isLatentMode(name)) 0.65
         else if (isPlainFilter(name)) 0.5
         else 0.35).toString
      )
  }

  /** The upscaler dropdown, grouped so the three kinds read apart: the
    * installed ESRGAN models, the latent-space modes, and the plain pixel
    * filters. The names come from the live server's scan — whatever is not a
    * known built-in is a model file from `--hires-upscalers-dir`.
    */
  private def upscalerSelectField(names: List[String]): HtmlElement = {
    val latentModes = names.filter(isLatentMode)
    val filters = names.filter(isPlainFilter)
    val models = names.filterNot(latentModes.toSet ++ filters)
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
          group("Upscaler models (sharpest pixels)", models),
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
      // "None" means "no upscaling" — that is what the checkbox is for.
      val upscalerNames = capabilities.upscalers.map(_.name).filter(_ != "None")
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
                div(
                  cls := "columns is-mobile",
                  div(
                    cls := "column",
                    numberField(
                      "Denoising strength (0-1)",
                      hiresDenoisingStrengthVar
                    )
                  ),
                  div(
                    cls := "column",
                    numberField(
                      "Upscale tile size (model upscalers)",
                      hiresUpscaleTileSizeVar,
                      Some(0)
                    )
                  )
                ),
                p(
                  cls := "is-size-7 text-secondary",
                  "Denoising is set from the upscaler you pick — light for a " +
                    "model, heavy for a latent mode — tune from there. New " +
                    "models install from the Model Cache page and appear on the " +
                    "next session launch."
                )
              )
          }
        )
      ).element
    }
  }
}
