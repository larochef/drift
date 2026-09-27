package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The attached images, folded — a txt2img run has none, and the summary names
  * them when it does. Shown only where the capabilities report the features: an
  * init image makes it img2img / img2vid, reference images drive edit-style
  * models, an end image bounds a video, and a mask paints into the init image.
  */
class InputsSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.{endImageVar, initImageVar, maskImageVar, refImagesVar}
  import FormFields.numberField

  private val features =
    capabilities.featuresByMode.getOrElse(currentMode, Map.empty)

  private def enabled(name: String) = features.getOrElse(name, false)

  private val summary: Signal[Option[String]] =
    initImageVar.signal
      .combineWith(maskImageVar.signal, refImagesVar.signal, endImageVar.signal)
      .map { (init, mask, refs, end) =>
        val attached =
          init.map(_ => "init image").toList :::
            mask.filter(_ => init.isDefined).map(_ => "mask").toList :::
            Option
              .when(refs.nonEmpty)(
                s"${refs.size} reference image${
                    if (refs.size == 1) "" else "s"
                  }"
              )
              .toList :::
            end.map(_ => "end image").toList
        Option.when(attached.nonEmpty)(attached.mkString(" · "))
      }

  /** The pickers for the mode. Strength appears once an init image is attached
    * — it means nothing without one.
    */
  private def pickers: HtmlElement = {
    val strengthField: Modifier[HtmlElement] =
      if (!enabled("init_image")) emptyMod
      else
        child <-- initImageVar.signal.map {
          case Some(_) =>
            numberField(
              "Strength (0-1, lower keeps more of the image)",
              state.strengthVar
            )
          case None => emptyNode
        }
    if (currentMode == "vid_gen")
      div(
        if (enabled("init_image"))
          singleImagePicker("Start image (img2vid)", initImageVar)
        else emptyNode,
        // No strength input for video: wan-style I2V conditions on the clean
        // first frame, and where strength does apply (LTX) its semantics are
        // conditioning strength, not img2img denoise. The server default
        // passes through untouched.
        if (enabled("end_image"))
          singleImagePicker("End image", endImageVar)
        else emptyNode
      )
    else
      div(
        if (enabled("init_image"))
          singleImagePicker("Init image (img2img)", initImageVar)
        else emptyNode,
        strengthField,
        // Offered only once there is an image to paint into.
        if (enabled("mask_image"))
          child <-- initImageVar.signal.map {
            case Some(_) => singleImagePicker("Mask (inpaint)", maskImageVar)
            case None    => emptyNode
          }
        else emptyMod,
        if (enabled("ref_images"))
          multiImagePicker("Reference images (edit)", refImagesVar)
        else emptyNode
      )
  }

  lazy val element: HtmlElement =
    if (!List("init_image", "ref_images", "end_image").exists(enabled)) div()
    else CollapsibleSection("Inputs", summary, pickers).element
}
