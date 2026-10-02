package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The attached inputs, folded — a txt2img run has none, and the summary names
  * them when it does. Shown only where the capabilities report the features: an
  * init image makes it img2img / img2vid, reference images drive edit-style
  * models, an end image bounds a video, and a mask paints into the init image.
  * A video model may also take references (images, clips, sounds), guides held
  * at a frame and a control video (`specs/42`, step 14).
  */
class InputsSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String,
    /** The gallery and the assistant, where the page has them. */
    tools: InputTools
) extends Component {
  import state.{
    controlVideoVar,
    endImageVar,
    guidesVar,
    initImageVar,
    maskImageVar,
    refImagesVar,
    referencesVar
  }
  import FormFields.numberField

  private val features =
    capabilities.featuresByMode.getOrElse(currentMode, Map.empty)

  private def enabled(name: String) = features.getOrElse(name, false)

  private def counted(count: Int, noun: String): Option[String] =
    Option.when(count > 0)(s"$count $noun${if (count == 1) "" else "s"}")

  private val imageInputs: Signal[List[String]] =
    initImageVar.signal
      .combineWith(maskImageVar.signal, refImagesVar.signal)
      .map { (init, mask, refs) =>
        init.map(_ => "init image").toList :::
          mask.filter(_ => init.isDefined).map(_ => "mask").toList :::
          counted(refs.size, "reference image").toList
      }

  private val videoInputs: Signal[List[String]] =
    initImageVar.signal
      .combineWith(
        endImageVar.signal,
        referencesVar.signal,
        guidesVar.signal,
        controlVideoVar.signal
      )
      .map { (init, end, references, guides, control) =>
        init
          .filter(_ => enabled("init_image"))
          .map(_ => "start image")
          .toList :::
          end.filter(_ => enabled("end_image")).map(_ => "end image").toList :::
          counted(references.size, "reference")
            .filter(_ => enabled("references"))
            .toList :::
          counted(guides.size, "guide")
            .filter(_ => enabled("guides"))
            .toList :::
          control
            .filter(_ => enabled("control_video"))
            .map(_ => "control video")
            .toList
      }

  private val summary: Signal[Option[String]] =
    (if (currentMode == "vid_gen") videoInputs else imageInputs).map(attached =>
      Option.when(attached.nonEmpty)(attached.mkString(" · "))
    )

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
          singleMediaPicker(
            "Start image (img2vid)",
            initImageVar,
            MediaAccept.Images,
            tools
          )
        else emptyNode,
        // No strength input for video: wan-style I2V conditions on the clean
        // first frame, and where strength does apply (LTX) its semantics are
        // conditioning strength, not img2img denoise. The server default
        // passes through untouched.
        if (enabled("end_image"))
          singleMediaPicker(
            "End image",
            endImageVar,
            MediaAccept.Images,
            tools
          )
        else emptyNode,
        // Order is meaning here: the prompt names references by position.
        if (enabled("references"))
          multiMediaPicker(
            "References (in the order the prompt names them)",
            referencesVar,
            MediaAccept.AnyMedia,
            ordered = true,
            tools
          )
        else emptyNode,
        if (enabled("guides")) GuidesPicker(state, tools).element
        else emptyNode,
        if (enabled("control_video")) ControlVideoInputs(state, tools).element
        else emptyNode
      )
    else
      div(
        if (enabled("init_image"))
          singleMediaPicker(
            "Init image (img2img)",
            initImageVar,
            MediaAccept.Images,
            tools
          )
        else emptyNode,
        strengthField,
        // Offered only once there is an image to paint into.
        if (enabled("mask_image"))
          child <-- initImageVar.signal.map {
            case Some(_) =>
              singleMediaPicker(
                "Mask (inpaint)",
                maskImageVar,
                MediaAccept.Images,
                tools,
                askable = false
              )
            case None => emptyNode
          }
        else emptyMod,
        if (enabled("ref_images"))
          multiMediaPicker(
            "Reference images (edit)",
            refImagesVar,
            MediaAccept.Images,
            ordered = false,
            tools
          )
        else emptyNode
      )
  }

  lazy val element: HtmlElement =
    if (
      !List(
        "init_image",
        "ref_images",
        "end_image",
        "references",
        "guides",
        "control_video"
      ).exists(enabled)
    ) div()
    else CollapsibleSection("Inputs", summary, pickers).element
}
