package drift.frontend.pages.architectures

import drift.frontend.components.*
import drift.frontend.services.BrowserServices
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Registers one model in a checkpoint slot (`specs/02-model-registry.md`,
  * `specs/03-model-discovery.md`), or edits one already registered. The browser
  * opens first and what is picked there *is* the source — none of it is typed
  * (François, 2026-09-17) — then this panel takes the id and the label to save
  * it under, shows the source as it stands, and offers the way back to the
  * browser.
  *
  * Editing starts from the stored model instead, with the browser closed and
  * the id fixed: the id is the file name and every run configuration assigns by
  * it, so a rename would be a different model.
  */
class ModelForm(
    browsers: BrowserServices,
    architecture: Architecture,
    familyId: String,
    /** Every registered model, so the browser marks what drift already has
      * (`specs/03-model-discovery.md`).
      */
    allModels: Signal[List[Model]],
    /** Nothing was picked and the browser was dismissed: there is no model to
      * add, so the panel goes away with it.
      */
    onCancel: () => Unit,
    /** The model being edited, none meaning a new one. */
    editing: Option[Model] = None
) extends Component {

  private val source = Var(editing.map(_.source))
  private val browsing = Var(editing.isEmpty)
  private val modelId = Var(editing.map(_.id).getOrElse(""))
  private val modelLabel = Var(editing.map(_.label).getOrElse(""))
  // The architecture's defaults are what this model overrides or removes
  // (`specs/16-parameter-resolution.md`), so the editor can offer them.
  private val paramsEditor = ParamEditor(
    editing.map(_.parameters.toList.sortBy(_._1)).getOrElse(Nil),
    editing.map(_.removedParameters).getOrElse(Nil),
    inherited = Val(
      architecture.defaultParameters.toList.map((flag, value) =>
        ResolvedParameter(flag, value, ParameterLayer.Architecture)
      )
    )
  )

  /** What the last pick suggested. A field still holding it was never touched,
    * so another pick may fill it again; one the user wrote is left alone.
    */
  private var suggestedId = ""
  private var suggestedLabel = ""

  /** Enough to save: a file, an id and a label. */
  val ready: Signal[Boolean] = source.signal
    .combineWith(modelId.signal, modelLabel.signal)
    .map((picked, id, label) =>
      picked.isDefined && id.nonEmpty && label.nonEmpty
    )

  def snapshot(): Option[Model] = source.now().map { picked =>
    Model(
      // Entity ids become file names (~/.config/drift/models/<id>.json), so
      // slashes would create nested paths that `list` never sees.
      editing.map(_.id).getOrElse(modelId.now().replace('/', '-')),
      familyId,
      modelLabel.now(),
      picked,
      ModelForm.formatOf(picked),
      paramsEditor.snapshot(),
      paramsEditor.removedSnapshot(),
      builtIn = editing.exists(_.builtIn)
    )
  }

  private def pick(picked: ModelSource, label: String): Unit = {
    val id = ModelForm.slug(label)
    if (modelLabel.now().isEmpty || modelLabel.now() == suggestedLabel)
      modelLabel.set(label)
    if (modelId.now().isEmpty || modelId.now() == suggestedId)
      modelId.set(id)
    suggestedLabel = label
    suggestedId = id
    source.set(Some(picked))
    browsing.set(false)
  }

  lazy val element: HtmlElement = div(
    child <-- source.signal.map {
      case Some(picked) => panel(picked)
      // The browser is up and nothing is picked yet: there is nothing to say
      // behind it.
      case None => emptyNode
    },
    child <-- browsing.signal.map {
      case false => emptyNode
      case true  =>
        SourceBrowser(
          browsers = browsers,
          architecture = architecture,
          forLoras = false,
          installed = allModels.map(Installed.models),
          onFile = (picked, label) => pick(picked, label),
          onCancel = () => {
            browsing.set(false)
            if (source.now().isEmpty) onCancel()
          }
        ).element
    }
  )

  private def panel(picked: ModelSource): HtmlElement = div(
    cls := "card bg-table-header card-content",
    div(
      cls := "columns",
      div(
        cls := "column",
        label(cls := "label text-primary", "Model ID"),
        editing match {
          // Configurations assign by id: changing it would leave them
          // pointing at a model that no longer exists.
          case Some(stored) => p(cls := "text-secondary", stored.id)
          case None         =>
            input(
              cls := "input",
              placeholder := "e.g. my-vae-model",
              value <-- modelId,
              onInput.mapToValue --> modelId
            )
        }
      ),
      div(
        cls := "column",
        label(cls := "label text-primary", "Label"),
        input(
          cls := "input",
          placeholder := "My VAE Model",
          value <-- modelLabel,
          onInput.mapToValue --> modelLabel
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Source"),
      div(
        cls := "is-flex is-align-items-center",
        styleAttr := "gap: 0.75rem;",
        div(
          styleAttr := "flex: 1 1 auto; min-width: 0;",
          picked.lines.map(line =>
            p(cls := "is-size-7 text-secondary text-break", line)
          ),
          p(
            cls := "is-size-7 text-secondary",
            s"weights: ${ModelForm.formatOf(picked)}"
          )
        ),
        button(
          cls := "button is-small",
          styleAttr := "flex: 0 0 auto;",
          "Change model",
          title := "Pick another file; the source follows what you pick",
          onClick --> (_ => browsing.set(true))
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary", "Parameters"),
      p(
        cls := "help text-secondary mb-2",
        "What this checkpoint wants on the command line, over its " +
          "architecture's defaults. 🚫 keeps one of those defaults " +
          "off entirely — a run configuration can still set it again."
      ),
      paramsEditor.element
    )
  )
}

object ModelForm {

  /** The weight format, read off the chosen file rather than assumed: most of
    * the built-in architectures are served as `.gguf`, which the previous
    * hardcoded "safetensors" mislabelled.
    */
  def formatOf(source: ModelSource): String = {
    val name = source.fileName
    // A split model is registered by its index; its weights are safetensors.
    if (ShardedSafetensors.isIndex(name)) "safetensors"
    else
      name.lastIndexOf('.') match {
        case i if i >= 0 && i < name.length - 1 =>
          name.substring(i + 1).toLowerCase
        case _ => "safetensors"
      }
  }

  /** An id suggested from a label: what the user would have typed anyway, and
    * theirs to change before saving.
    */
  def slug(label: String): String =
    label.toLowerCase
      .replaceAll("""[^a-z0-9]+""", "-")
      .replaceAll("""^-+|-+$""", "")
}
