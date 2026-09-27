package drift.frontend.pages.architectures

import drift.frontend.components.*
import drift.frontend.services.*
import drift.frontend.services.ArchitectureService.{Command, Event}
import drift.shared.*

import com.raquo.laminar.api.L.*

class ArchitecturesPage(
    service: ArchitectureService,
    modelService: ModelService,
    cacheService: CacheService,
    downloadService: DownloadService,
    loraService: LoraService,
    browsers: BrowserServices,
    /** The assistant system templates, for an architecture's default. */
    assistantTemplates: Signal[List[PromptTemplate]],
    /** The installed runtimes, for the drift runner's model kinds. */
    runtimeService: RuntimeService
) extends Component {
  private val runtimes = runtimeService.runtimes
  private val showForm = Var(false)
  private val createForm =
    ArchitectureForm(
      assistantTemplates = assistantTemplates,
      runtimes = runtimes
    )
  private val editingArchId = Var(Option.empty[String])
  private val editForm =
    ArchitectureForm(
      showId = false,
      assistantTemplates = assistantTemplates,
      runtimes = runtimes
    )
  private val searchQuery = Var("")
  private val tag = Var(Option.empty[String])
  private val filteredArchs: Signal[List[Architecture]] =
    service.architectures
      .combineWith(searchQuery.signal, tag.signal)
      .map { (archs, q, tag) =>
        val needle = q.trim.toLowerCase
        archs
          .filter(a => tag.forall(a.tags.contains))
          .filter(a =>
            needle.isEmpty ||
              a.label.toLowerCase.contains(needle) ||
              a.id.toLowerCase.contains(needle)
          )
      }

  private def handleCreate(): Unit = {
    val (
      id,
      label,
      checkpoints,
      params,
      civitaiBaseModels,
      decoder,
      tool,
      tags,
      sizeMultiple
    ) =
      createForm.snapshot()
    if (id.nonEmpty && label.nonEmpty)
      service.push(
        Command.Create(
          Architecture(
            id,
            label,
            tool,
            checkpoints,
            params,
            civitaiBaseModels,
            tags,
            builtIn = false,
            pixelDiffusionDecoder = decoder,
            assistantTemplateId = createForm.assistantTemplateId,
            sizeMultiple = sizeMultiple,
            modelKind = createForm.modelKind,
            runners = createForm.runners
          )
        )
      )
  }

  private def handleDelete(id: String): Unit =
    service.push(Command.Delete(id))

  private def startEdit(a: Architecture): Unit = {
    editingArchId.set(Some(a.id))
    editForm.reset(
      label = a.label,
      checkpoints = a.checkpoints,
      params = a.defaultParameters.toList,
      civitaiBaseModels = a.civitaiBaseModels,
      pixelDiffusionDecoder = a.pixelDiffusionDecoder,
      tool = a.tool,
      tags = a.tags,
      assistantTemplateId = a.assistantTemplateId,
      sizeMultiple = a.sizeMultiple,
      modelKind = a.modelKind,
      runners = a.runners
    )
  }

  private def cancelEdit(): Unit =
    editingArchId.set(None)

  private def handleSave(a: Architecture): Unit = {
    val (
      _,
      label,
      checkpoints,
      params,
      civitaiBaseModels,
      decoder,
      tool,
      tags,
      sizeMultiple
    ) =
      editForm.snapshot()
    if (label.nonEmpty)
      service.push(
        Command.Update(
          a.id,
          Architecture(
            a.id,
            label,
            tool,
            checkpoints,
            params,
            civitaiBaseModels,
            tags,
            a.builtIn,
            pixelDiffusionDecoder = decoder,
            promptingNotes = a.promptingNotes,
            assistantTemplateId = editForm.assistantTemplateId,
            sizeMultiple = sizeMultiple,
            // Not in the form yet: kept as stored.
            initImage = a.initImage,
            referenceImages = a.referenceImages,
            huggingFaceBaseModels = a.huggingFaceBaseModels,
            modelScopeBaseModels = a.modelScopeBaseModels,
            modelKind = editForm.modelKind,
            runners = editForm.runners
          )
        )
      )
  }

  private def resetCreateForm(): Unit = {
    createForm.reset()
    showForm.set(false)
  }

  private def handleAddModel(m: Model): Unit =
    modelService.push(ModelService.Command.Create(m))

  private def handleDeleteModel(modelId: String): Unit =
    modelService.push(ModelService.Command.Delete(modelId))

  private def handleSaveModel(m: Model): Unit =
    modelService.push(ModelService.Command.Update(m.id, m))

  /** A model just registered or re-pointed: its cache status is unknown until
    * it is re-read, and a remote source starts downloading right away, like a
    * LoRA install — registering a model from a browser means wanting its
    * weights. An already-cached source answers with an immediately-completed
    * job.
    */
  private def registered(model: Model): Unit = {
    cacheService.push(CacheService.Command.Load)
    model.source match {
      case Local(_) => ()
      case _        =>
        downloadService.push(DownloadService.Command.Start(model.id))
    }
  }

  private def handleSaveArch(a: Architecture): Unit =
    service.push(Command.Update(a.id, a))

  lazy val element: HtmlElement = div(
    cls := "content",
    service.effects,
    // Without this, ModelService's command subscriptions are never active and
    // every `modelService.push(...)` below is silently dropped by its EventBus.
    modelService.effects,
    cacheService.effects,
    downloadService.effects,
    loraService.effects,
    runtimeService.effects,
    onMountCallback(_ => runtimeService.push(RuntimeService.Command.Load)),
    downloadService.events --> Observer {
      case DownloadService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
        downloadService.push(DownloadService.Command.Load)
    },
    service.events --> Observer {
      case Event.Created(_) => createForm.reset(); showForm.set(false)
      case Event.Updated(_) => editingArchId.set(None)
      case _                => ()
    },
    modelService.events --> Observer {
      // An edit can point the model at another file, which wants fetching
      // exactly like a fresh registration does.
      case ModelService.Event.Created(model) =>
        registered(model)
      case ModelService.Event.Updated(model) =>
        registered(model)
      case _ => ()
    },
    onMountCallback { _ =>
      service.push(Command.Load)
      modelService.push(ModelService.Command.Load)
      cacheService.push(CacheService.Command.Load)
      loraService.push(LoraService.Command.Load)
      loraService.push(LoraService.Command.LoadCatalog)
      loraService.push(LoraService.Command.LoadJobs)
    },
    div(
      cls := "level",
      div(
        cls := "level-left",
        h1(cls := "title text-primary", "Architectures")
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-primary",
          span(cls := "plus-icon", "+"),
          " New Architecture",
          onClick --> (_ => showForm.update(!_))
        )
      )
    ),
    hr(),
    ErrorBanner(service),
    ErrorBanner(modelService),
    ErrorBanner(cacheService),
    ErrorBanner(downloadService),
    ErrorBanner(loraService),
    div(
      cls := "list-filters mb-4",
      div(
        cls := "field is-expanded mb-0",
        div(
          cls := "control",
          input(
            cls := "input",
            placeholder := "Search architectures\u2026",
            onInput.mapToValue --> searchQuery
          )
        )
      ),
      TagFilter(tag, service.architectures.map(TagFilter.inUse))
    ),
    child <-- showForm.signal.map { open =>
      if (open) {
        div(
          cls := "card bg-card mb-4",
          div(
            cls := "card-content",
            createForm,
            div(
              cls := "buttons",
              button(
                cls := "button is-success",
                span(cls := "plus-icon", "+"),
                " Create",
                onClick --> (_ => handleCreate())
              ),
              button(
                cls := "button",
                "\u274C Cancel",
                onClick --> (_ => resetCreateForm())
              )
            )
          )
        )
      } else emptyNode
    },
    child <-- editingArchId.signal.combineWith(filteredArchs).map {
      (editing, archs) =>
        val items = archs.flatMap { arch =>
          if (editing.contains(arch.id)) {
            List(
              div(
                cls := "column is-full",
                div(
                  cls := "card bg-card mb-4",
                  div(
                    cls := "card-header",
                    p(
                      cls := "card-header-title text-primary",
                      s"Edit: ${arch.id}"
                    )
                  ),
                  div(
                    cls := "card-content",
                    editForm,
                    div(
                      cls := "buttons",
                      button(
                        cls := "button is-success",
                        "\uD83D\uDCBE Save",
                        onClick --> (_ => handleSave(arch))
                      ),
                      button(
                        cls := "button",
                        "\u274C Cancel",
                        onClick --> (_ => cancelEdit())
                      )
                    )
                  )
                )
              )
            )
          } else {
            List(
              div(
                cls := "column is-half-medium",
                ArchitectureCard(
                  browsers,
                  arch,
                  handleDelete,
                  startEdit,
                  modelService.allModels,
                  cacheService.statuses,
                  downloadService.jobs,
                  id => downloadService.push(DownloadService.Command.Start(id)),
                  id =>
                    downloadService.push(DownloadService.Command.Cancel(id)),
                  handleAddModel,
                  handleDeleteModel,
                  handleSaveModel,
                  handleSaveArch,
                  loraService
                ).element
              )
            )
          }
        }
        if (items.isEmpty) emptyNode
        else div(cls := "columns is-multiline", items)
    }
  )
}
