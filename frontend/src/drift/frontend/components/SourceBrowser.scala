package drift.frontend.components

import drift.frontend.services.BrowserServices
import drift.shared.*

import com.raquo.laminar.api.L.*

/** One browser for every place a file can come from (François, 2026-09-17: "I'm
  * getting sick of these + civitai, + huggingface and so on, this makes too
  * many buttons"): the source is a row of buttons at the top of the modal, and
  * what comes back carries the source it was picked from, so nothing about it
  * is ever typed.
  *
  * Both callers open it: the LoRA section of an architecture
  * (`specs/33-lora-sources.md`), where a Civitai *version* installs as one
  * LoRA, and the model form (`specs/03-model-discovery.md`), where one file
  * becomes one model.
  */
class SourceBrowser(
    browsers: BrowserServices,
    architecture: Architecture,
    /** Browsing for a LoRA: LoRAs only, the architecture's base models on each
      * site, and a Civitai version installs whole.
      */
    forLoras: Boolean,
    /** Registering a model: one file, with the label to suggest for it. */
    onFile: (ModelSource, String) => Unit,
    onCancel: () => Unit,
    /** Installing LoRAs (`specs/33-lora-sources.md`): the files ticked in one
      * model or repository, and where they land. The browser stays open —
      * finding a wan 2.2 pair often means several installs from one search.
      */
    onInstall: Option[(LoraInstallSource, LoraGrouping) => Unit] = None,
    installed: Signal[Installed] = Val(Installed.none),
    /** Registering a model: the slot it is for. A component — a text encoder, a
      * VAE, a vision projector — is looked for by its family, not by the
      * architecture (`specs/03-model-discovery.md`).
      */
    slot: Option[CheckpointRef] = None
) extends Component {

  /** Where the search starts. A site that filters by base model needs no name —
    * the filter is sharper, and many files trained on a model do not repeat its
    * name. A parenthesis ("(chat)") is no part of the name.
    */
  private def searchFor(baseModels: List[String]): String =
    if (baseModels.nonEmpty) ""
    else architecture.label.replaceAll("""\s*\([^)]*\)""", "").trim

  /** A slot for a model the architecture shares with others: its files are not
    * the architecture's, and no base-model filter can find them.
    */
  private val component: Option[CheckpointRef] =
    slot.filter(ref => !forLoras && !ref.ownModel)

  /** What a component is searched as: its family, less the suffix naming the
    * part of a release it is — an mmproj or an MTP head is in the same
    * repositories as the model itself.
    */
  private def familyQuery(ref: CheckpointRef): String =
    ref.familyId.replaceAll("""-(mmproj|mtp|tokenizer)$""", "")

  /** A component's two searches: its family finds the standalone releases, the
    * architecture's name the repositories that repackage it. A VAE's family
    * rarely names a repository, so the architecture comes first there.
    */
  private val componentSearches: List[String] = component.toList.flatMap {
    ref =>
      val family = familyQuery(ref)
      val architectureName = searchFor(Nil)
      (if (ref.familyId.contains("vae")) List(architectureName, family)
       else List(family, architectureName)).distinct
  }

  /** Civitai hosts no chat model LoRAs (`specs/35-assistant-loras.md`), and it
    * lists LoRAs by base model: without one there is nothing it could show.
    */
  private def civitaiUnavailable: Option[String] =
    if (!forLoras) None
    else
      Option.when(architecture.civitaiBaseModels.isEmpty)(
        s"${architecture.label} declares no Civitai base model, and the " +
          "Civitai browser lists LoRAs by base model" +
          (if (architecture.builtIn) "."
           else " — add one in the architecture's form.")
      )

  private val offered: List[ModelSourceType] =
    List(
      // Civitai has no type for a text encoder or a projector, and its
      // base-model filter only finds the architecture's own checkpoints.
      Option.when(
        !(forLoras && architecture.tool == RuntimeTool.LlamaCpp) &&
          component.isEmpty
      )(ModelSourceType.Civitai),
      Some(ModelSourceType.HuggingFace),
      Some(ModelSourceType.ModelScope),
      Some(ModelSourceType.Local)
    ).flatten

  /** Where the browser opens: most LoRAs are on Civitai, most models on
    * HuggingFace — and whichever it is must be one of the sources offered.
    */
  private val source = Var(
    offered
      .find(
        _ == (
          if (forLoras && civitaiUnavailable.isEmpty) ModelSourceType.Civitai
          else ModelSourceType.HuggingFace
        )
      )
      .getOrElse(offered.head)
  )

  /** The row in the modal's head, in every source's browser: where this browser
    * is looking is the first thing it says.
    */
  private def switch: HtmlElement =
    div(
      cls := "buttons has-addons mb-0",
      offered.map { kind =>
        val unavailable =
          if (kind == ModelSourceType.Civitai) civitaiUnavailable else None
        button(
          cls := "button",
          cls("is-info") <-- source.signal.map(_ == kind),
          ProviderIcon.of(kind, cls := "is-large mr-2"),
          SourceBrowser.name(kind),
          title := unavailable.getOrElse(
            s"Look for it on ${SourceBrowser.name(kind)}"
          ),
          aria.disabled := unavailable.isDefined,
          unavailable.map(_ =>
            styleAttr := "opacity: 0.45; cursor: not-allowed;"
          ),
          onClick --> (_ => if (unavailable.isEmpty) source.set(kind))
        )
      }
    )

  /** What to call a file picked with no label of its own: its name without the
    * extension, which is what the LoRA installer names one.
    */
  private def labelOf(name: String): String =
    name.split('/').last.replaceAll("""\.[^.]+$""", "")

  lazy val element: HtmlElement = div(
    child <-- source.signal.map {
      case ModelSourceType.Civitai =>
        CivitaiBrowser(
          service = browsers.civitai,
          authTokens = browsers.authTokens,
          initialQuery =
            if (forLoras) "" else searchFor(architecture.civitaiBaseModels),
          initialModelType = Some(if (forLoras) "LORA" else "CHECKPOINT"),
          civitaiBaseModels = architecture.civitaiBaseModels,
          onSelect = (modelId, versionId, fileId, fileName, label) =>
            onFile(Civitai(modelId, versionId, fileId, fileName), label),
          onCancel = onCancel,
          onInstall = onInstall.map(install =>
            (modelId, files, grouping) =>
              install(LoraInstallSource.CivitaiFiles(modelId, files), grouping)
          ),
          installed = installed,
          sourceSwitch = switch
        ).element
      case ModelSourceType.HuggingFace =>
        val baseModels =
          if (forLoras) architecture.huggingFaceBaseModels else Nil
        HuggingFaceBrowser(
          huggingFace = browsers.huggingFace,
          initialQuery =
            componentSearches.headOption.getOrElse(searchFor(baseModels)),
          suggestions = componentSearches,
          onSelect =
            (repo, file) => onFile(HuggingFace(repo, file), labelOf(file)),
          onCancel = onCancel,
          baseModels = baseModels,
          forLoras = forLoras,
          installed = installed,
          onInstall = onInstall.map(install =>
            (repo, paths, grouping) =>
              install(
                LoraInstallSource.HuggingFaceFiles(repo, paths),
                grouping
              )
          ),
          sourceSwitch = switch
        ).element
      case ModelSourceType.ModelScope =>
        val baseModels =
          if (forLoras) architecture.modelScopeBaseModels else Nil
        ModelScopeBrowser(
          modelScope = browsers.modelScope,
          initialQuery =
            componentSearches.headOption.getOrElse(searchFor(baseModels)),
          suggestions = componentSearches,
          onSelect =
            (repo, path) => onFile(ModelScope(repo, path), labelOf(path)),
          onCancel = onCancel,
          baseModels = baseModels,
          forLoras = forLoras,
          installed = installed,
          onInstall = onInstall.map(install =>
            (repo, paths, grouping) =>
              install(LoraInstallSource.ModelScopeFiles(repo, paths), grouping)
          ),
          sourceSwitch = switch
        ).element
      case ModelSourceType.Local =>
        FileBrowser(
          service = browsers.files,
          onSelect = path => onFile(Local(path), labelOf(path)),
          onCancel = onCancel,
          onInstall = onInstall.map(install =>
            (paths, grouping) =>
              install(LoraInstallSource.LocalFiles(paths), grouping)
          ),
          // A file on this machine says nothing about where it belongs, so
          // every LoRA of the architecture is a candidate to join.
          candidates = installed.map(_.all),
          sourceSwitch = switch
        ).element
    }
  )
}

object SourceBrowser {

  /** What each source is called on its button and wherever a source is named.
    */
  def name(kind: ModelSourceType): String = kind match {
    case ModelSourceType.HuggingFace => "HuggingFace"
    case ModelSourceType.ModelScope  => "ModelScope"
    case ModelSourceType.Civitai     => "Civitai"
    case ModelSourceType.Local       => "This machine"
  }
}
