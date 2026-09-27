package drift.frontend.pages.architectures

import drift.frontend.components.*
import drift.shared.*

import com.raquo.laminar.api.L.*

class ArchitectureForm(
    showId: Boolean = true,
    initialId: String = "",
    initialLabel: String = "",
    initialCheckpoints: List[CheckpointRef] = Nil,
    initialParams: List[(String, String)] = Nil,
    initialCivitaiBaseModels: List[String] = Nil,
    initialTags: List[String] = Nil,
    initialPixelDiffusionDecoder: Boolean = false,
    /** The architecture's `sizeMultiple`. 16 is where the form starts a new
      * one, which is what nearly every family wants; the field is there to be
      * changed for the ones that do not (`specs/27-redraw.md`).
      */
    initialSizeMultiple: Int = 16,
    initialTool: RuntimeTool = RuntimeTool.SdCpp,
    /** The assistant system templates to offer as the architecture's default
      * (`specs/32-prompt-library.md`).
      */
    assistantTemplates: Signal[List[PromptTemplate]] = Val(Nil),
    initialAssistantTemplateId: Option[String] = None,
    /** The installed runtimes: the drift runner's say which model kinds it runs
      * (`specs/43`).
      */
    runtimes: Signal[List[Runtime]] = Val(Nil)
) extends Component {
  private val idVar = Var(initialId)
  private val labelVar = Var(initialLabel)
  private val toolVar = Var(initialTool)
  private val pixelDiffusionDecoderVar = Var(initialPixelDiffusionDecoder)
  private val sizeMultipleVar = Var(initialSizeMultiple.toString)
  private val assistantTemplateVar = Var(initialAssistantTemplateId)
  private val modelKindVar = Var("")
  private val driftRunnerVar = Var(false)

  /** What the model is, as the engines name it; none when left empty. */
  def modelKind: Option[String] =
    Option(modelKindVar.now().trim).filter(_.nonEmpty)

  /** The engines that run it: its tool's upstream one, and the drift runner
    * when ticked (`specs/43`).
    */
  def runners: List[RuntimeEngine] =
    RuntimeEngine.upstream(toolVar.now()) ::
      Option.when(driftRunnerVar.now())(RuntimeEngine.DriftRunner).toList

  /** The kinds the installed drift runner of the form's tool runs, when one is
    * installed and valid.
    */
  private val runnerKinds: Signal[Option[List[String]]] =
    runtimes
      .combineWith(toolVar.signal)
      .map((all, tool) =>
        all
          .find(r =>
            r.engine == RuntimeEngine.DriftRunner && r.tool == tool && r.valid
          )
          .flatMap(_.modelKinds)
      )

  /** The default assistant template chosen, none meaning drift's helper. */
  def assistantTemplateId: Option[String] = assistantTemplateVar.now()
  private val checkpointEditor = CheckpointEditor(initialCheckpoints)
  // The bottom layer: a default that is "not passed" is a default that is
  // simply not written here (`specs/16-parameter-resolution.md`).
  private val paramsEditor = ParamEditor(initialParams, allowsRemoval = false)
  private val baseModelManager = BaseModelManager(initialCivitaiBaseModels)

  /** What this architecture is for. The same little list manager the base
    * models use - a set of words with an add box and a remove on each.
    */
  private val tagManager = BaseModelManager(initialTags)

  def snapshot(): (
      String,
      String,
      List[CheckpointRef],
      Map[String, String],
      List[String],
      Boolean,
      RuntimeTool,
      List[String],
      Int
  ) =
    (
      idVar.now(),
      labelVar.now(),
      checkpointEditor.snapshot(),
      paramsEditor.snapshot(),
      // Civitai hosts no chat models: a llama.cpp architecture keeps none.
      if (toolVar.now() == RuntimeTool.SdCpp) baseModelManager.snapshot()
      else Nil,
      pixelDiffusionDecoderVar.now(),
      toolVar.now(),
      tagManager.snapshot(),
      // An unreadable or absurd number is the one every family but a handful
      // runs on, never a silent zero: `Tiling` divides by it.
      sizeMultipleVar.now().trim.toIntOption.filter(_ >= 1).getOrElse(16)
    )

  def reset(
      id: String = "",
      label: String = "",
      checkpoints: List[CheckpointRef] = Nil,
      params: List[(String, String)] = Nil,
      civitaiBaseModels: List[String] = Nil,
      pixelDiffusionDecoder: Boolean = false,
      tool: RuntimeTool = RuntimeTool.SdCpp,
      tags: List[String] = Nil,
      assistantTemplateId: Option[String] = None,
      sizeMultiple: Int = 16,
      modelKind: Option[String] = None,
      runners: List[RuntimeEngine] = Nil
  ): Unit = {
    idVar.set(id)
    modelKindVar.set(modelKind.getOrElse(""))
    driftRunnerVar.set(runners.contains(RuntimeEngine.DriftRunner))
    assistantTemplateVar.set(assistantTemplateId)
    labelVar.set(label)
    toolVar.set(tool)
    pixelDiffusionDecoderVar.set(pixelDiffusionDecoder)
    sizeMultipleVar.set(sizeMultiple.toString)
    checkpointEditor.reset(checkpoints)
    paramsEditor.reset(params)
    baseModelManager.reset(civitaiBaseModels)
    tagManager.reset(tags)
  }

  /** What the model is, as the engines name it (`specs/43`). */
  private def modelKindField: HtmlElement = div(
    cls := "field",
    label(cls := "label text-primary", "Model kind"),
    div(
      cls := "control",
      input(
        cls := "input",
        placeholder := "qwen35moe, krea2…",
        value <-- modelKindVar.signal,
        onInput.mapToValue --> modelKindVar
      )
    ),
    p(
      cls := "help text-secondary",
      "What the model is, as the engines name it: the GGUF's " +
        "general.architecture for a chat model, the family for an image model. " +
        "It decides whether the drift runner can run it."
    )
  )

  /** Whether the drift runner runs it: offered when the installed runner runs
    * the kind, or when none is installed yet.
    */
  private def driftRunnerField: HtmlElement = div(
    cls := "field",
    child <-- runnerKinds
      .combineWith(modelKindVar.signal, driftRunnerVar.signal)
      .map { (kinds, kind, ticked) =>
        val runs = kinds.forall(_.contains(kind.trim))
        div(
          label(
            cls := "checkbox text-primary",
            input(
              typ := "checkbox",
              checked := ticked,
              disabled := !runs && !ticked,
              onChange.mapToChecked --> driftRunnerVar
            ),
            " Runs on the drift runner too"
          ),
          p(
            cls := "help text-secondary",
            kinds match {
              case None =>
                "No drift runner is installed; it can be ticked now and used once it is."
              case Some(list) if runs =>
                s"The installed runner runs ${kind.trim}."
              case Some(list) =>
                s"The installed runner does not run this kind; it runs ${list.mkString(", ")}."
            }
          )
        )
      }
  )

  /** The tag editor: the words already on the architecture, each removable, the
    * suggested vocabulary as one-click adds, and a box for anything else.
    */
  private def tagsField: HtmlElement = div(
    cls := "field",
    label(cls := "label text-primary", "Tags"),
    p(
      cls := "text-secondary is-size-7 mb-1",
      "What this architecture is for - it is how the lists filter."
    ),
    div(
      cls := "tags-editor",
      children <-- tagManager.models.map(
        _.map(tag =>
          span(
            cls := "tag is-primary",
            tag,
            button(
              cls := "delete is-small ml-1",
              onClick --> (_ => tagManager.removeBaseModel(tag))
            )
          )
        )
      )
    ),
    div(
      cls := "buttons are-small mt-1",
      children <-- tagManager.models.map { current =>
        drift.shared.ArchitectureTags.suggested
          .filterNot(current.contains)
          .map(tag =>
            button(
              cls := "button is-small",
              s"+ $tag",
              onClick --> { _ =>
                tagManager.newInput.set(tag)
                tagManager.addBaseModel()
              }
            )
          )
      }
    ),
    div(
      cls := "field has-addons",
      div(
        cls := "control is-expanded",
        input(
          cls := "input is-small",
          placeholder := "another tag",
          controlled(
            value <-- tagManager.newInput.signal,
            onInput.mapToValue --> tagManager.newInput
          ),
          onKeyDown.filter(_.key == "Enter") --> (_ =>
            tagManager.addBaseModel()
          )
        )
      ),
      div(
        cls := "control",
        button(
          cls := "button is-small",
          "Add",
          onClick --> (_ => tagManager.addBaseModel())
        )
      )
    )
  )

  private def removeBaseModel(name: String): Unit = {
    baseModelManager.removeBaseModel(name)
  }

  lazy val element: HtmlElement = div(
    if (showId) {
      div(
        cls := "field",
        label(cls := "label text-primary", "ID"),
        div(
          cls := "control",
          input(
            cls := "input",
            placeholder := "e.g. my-custom-model",
            value <-- idVar.signal,
            onInput.mapToValue --> idVar
          )
        )
      )
    } else emptyNode,
    div(
      cls := "field",
      label(cls := "label text-primary", "Label"),
      div(
        cls := "control",
        input(
          cls := "input",
          placeholder := "My Custom Model",
          value <-- labelVar.signal,
          onInput.mapToValue --> labelVar
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Runner"),
      div(
        cls := "control",
        select(
          cls := "select",
          onChange.mapToValue --> Observer[String] { name =>
            RuntimeTool.values.find(_.toString == name).foreach(toolVar.set)
          },
          RuntimeTool.values.toList.map(tool =>
            option(
              value := tool.toString,
              selected <-- toolVar.signal.map(_ == tool),
              tool match {
                case RuntimeTool.SdCpp    => "sd-cpp — image and video"
                case RuntimeTool.LlamaCpp => "llama.cpp — chat assistant"
              }
            )
          )
        )
      ),
      p(
        cls := "help text-secondary",
        "Checkpoint slots and parameters mean the same for both; the runner " +
          "decides the executable, its listen flags and how readiness is probed."
      )
    ),
    modelKindField,
    driftRunnerField,
    div(
      cls := "field",
      label(
        cls := "checkbox text-primary",
        input(
          typ := "checkbox",
          checked <-- pixelDiffusionDecoderVar.signal,
          onChange.mapToChecked --> pixelDiffusionDecoderVar
        ),
        " Pixel diffusion decoder (PiD) — its run configurations upscale " +
          "gallery images instead of generating from a prompt"
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Size multiple"),
      div(
        cls := "control",
        input(
          cls := "input",
          typ := "number",
          minAttr := "1",
          stepAttr := "8",
          styleAttr := "width: 8rem;",
          value <-- sizeMultipleVar.signal,
          onInput.mapToValue --> sizeMultipleVar
        )
      ),
      p(
        cls := "help text-secondary",
        "What sd-cpp rounds every side up to for this family — its VAE scale " +
          "factor times the diffusion model's down factor. 16 for nearly " +
          "everything, 32 for Qwen Image 2.1 and a few others. A redraw asks " +
          "for tiles on this multiple; get it wrong and the model answers " +
          "with a bigger image than the job asked for."
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Default assistant prompt"),
      PromptTemplatePicker(
        assistantTemplates,
        assistantTemplateVar,
        none = Some("drift's helper (default)"),
        small = false
      ).element,
      p(
        cls := "help text-secondary",
        "The system prompt a project's assistant starts with when it targets " +
          "this architecture; a project can still pick another."
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Checkpoints"),
      checkpointEditor.element
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Default parameters"),
      paramsEditor.element
    ),
    tagsField,
    // Civitai hosts image and video models only.
    child <-- toolVar.signal.map(_ == RuntimeTool.SdCpp).distinct.map {
      case false => emptyNode
      case true  =>
        div(
          // The form's last field: Bulma drops the margin of a last child,
          // and the page's buttons follow right after.
          cls := "field mb-4",
          label(cls := "label text-primary", "Civitai base models"),
          CivitaiBaseModelsEditor(
            baseModelManager,
            removeBaseModel,
            heading = false
          ).element
        )
    }
  )
}
