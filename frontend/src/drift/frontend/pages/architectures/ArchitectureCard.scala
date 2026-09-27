package drift.frontend.pages.architectures

import drift.frontend.components.*
import drift.frontend.services.{BrowserServices, LoraService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

class ArchitectureCard(
    browsers: BrowserServices,
    a: Architecture,
    onDelete: String => Unit,
    onEdit: Architecture => Unit,
    allModels: Signal[List[Model]],
    cacheStatuses: Signal[Map[String, ModelCacheStatus]],
    downloadJobs: Signal[Map[String, DownloadJob]],
    onDownload: String => Unit,
    onCancelDownload: String => Unit,
    onAddModel: Model => Unit,
    onDeleteModel: String => Unit,
    /** A registered model edited in place — its label, its file, and the
      * parameters it puts on the command line
      * (`specs/16-parameter-resolution.md`).
      */
    onSaveModel: Model => Unit,
    onSave: Architecture => Unit,
    loraService: LoraService
) extends Component {
  private val expandedCp = Var(Option.empty[Int])
  private val addForm = Var[Option[(Int, ModelForm)]](None)
  private val editForm = Var[Option[(String, ModelForm)]](None)
  private val baseModelManager = BaseModelManager(a.civitaiBaseModels)
  private val loraSection = LoraSection(browsers, a, loraService)

  lazy val element: HtmlElement = div(
    cls := "card bg-card mb-4",
    div(
      cls := "card-header",
      paddingRight := "0.75rem",
      p(
        cls := "card-header-title text-primary",
        a.label,
        if (a.pixelDiffusionDecoder)
          span(
            cls := "tag is-info is-small ml-2",
            title := "Pixel diffusion decoder: its run configurations " +
              "upscale gallery images",
            "PiD upscaler"
          )
        else emptyNode
      ),
      if (!a.builtIn) {
        List(
          button(
            cls := "button is-info is-small card-header-action",
            "\u270F\uFE0F Edit",
            onClick --> (_ => onEdit(a))
          ),
          button(
            cls := "button is-danger is-small card-header-action",
            "\uD83D\uDDD1\uFE0F Delete",
            onClick --> (_ =>
              if (window.confirm("Delete this architecture?")) onDelete(a.id)
            )
          )
        )
      } else emptyNode
    ),
    div(
      cls := "card-content",
      p(cls := "text-secondary mb-2", s"ID: ${a.id}"),
      if (a.tags.isEmpty) emptyNode
      else
        div(
          cls := "tags-editor mb-2",
          a.tags.map(tag => span(cls := "tag is-primary is-light", tag))
        ),
      div(
        cls := "field",
        label(cls := "label text-primary", "Checkpoints"),
        table(
          cls := "table is-fullwidth is-narrow bg-table-header text-primary mb-3",
          thead(
            tr(
              th(cls := "text-secondary", "Name"),
              th(cls := "text-secondary", "Flag"),
              th(cls := "text-secondary", "Family"),
              th("")
            )
          ),
          tbody(
            children <-- expandedCp.signal.combineWith(allModels).map {
              (expanded, models) =>
                a.checkpoints.zipWithIndex.flatMap { (cp, idx) =>
                  val cnt = models.count(_.familyId == cp.familyId)
                  val isExpanded = expanded.contains(idx)
                  val mainRow = tr(
                    td(cls := "text-primary table-cell-break", cp.name),
                    td(span(cls := "tag is-small is-dark", cp.flag)),
                    td(span(cls := "text-primary is-size-7", cp.familyId)),
                    td(
                      cls := "has-text-right",
                      span(cls := "tag is-info is-small", s"$cnt"),
                      " ",
                      // A tag like the count beside it, so the two match in
                      // size.
                      span(
                        cls := "tag is-small cursor-pointer",
                        title := (if (isExpanded) "Hide the models"
                                  else "Show the models"),
                        onClick --> (_ => {
                          expandedCp.update {
                            case Some(`idx`) => None; case _ => Some(idx)
                          };
                          addForm.set(None)
                          editForm.set(None)
                        }),
                        if (isExpanded) "\u25BC" else "\u25B6"
                      )
                    )
                  )
                  val expandedRow = if (isExpanded) {
                    List(
                      tr(
                        td(
                          colSpan := 4,
                          cls := "p-0",
                          renderExpandedContent(cp, idx)
                        )
                      )
                    )
                  } else Nil
                  mainRow +: expandedRow
                }
            }
          )
        )
      ),
      loraSection.element,
      div(cls := "columns"),
      div(
        cls := "columns",
        div(
          cls := "column is-half",
          optionRow("Default parameters", a.defaultParameters)
        ),
        // Civitai hosts image and video models: a chat architecture has no
        // base model there (`specs/35-assistant-loras.md`).
        if (a.tool == RuntimeTool.SdCpp)
          div(cls := "column is-half", baseModelsColumn)
        else emptyNode
      )
    )
  )

  private def renderExpandedContent(
      cp: CheckpointRef,
      idx: Int
  ): HtmlElement = {
    val familyModels = allModels.map(_.filter(_.familyId == cp.familyId))
    div(
      cls := "p-2 bg-surface rounded",
      div(
        cls := "field",
        label(
          cls := "label text-primary is-size-6",
          s"Models in family '${cp.familyId}'"
        ),
        children <-- familyModels.map { ms =>
          if (ms.isEmpty) {
            List(p(cls := "text-secondary is-size-7", "No models defined yet."))
          } else {
            ms.map { m =>
              div(
                cls := "level is-mobile mb-1 is-marginless",
                div(
                  cls := "level-left",
                  div(
                    p(
                      cls := "text-primary text-break",
                      ProviderIcon.of(m.source).amend(cls := "mr-2"),
                      m.label
                    ),
                    m.source.lines.map(line =>
                      p(cls := "text-secondary is-size-7 text-break", line)
                    )
                  )
                ),
                div(
                  cls := "level-item",
                  child <-- cacheStatuses
                    .combineWith(downloadJobs)
                    .map { (statuses, jobs) =>
                      CacheIndicator(
                        m.id,
                        statuses.get(m.id),
                        jobs.get(m.id),
                        onDownload,
                        onCancelDownload
                      ).element
                    }
                ),
                div(
                  cls := "level-right",
                  // A built-in model is re-seeded on every start and the
                  // server refuses to delete it: an edit would be overwritten
                  // on the next start, so a run configuration's own overrides
                  // are where its parameters are changed. Its downloaded file
                  // is what can go, from the Model Cache.
                  if (m.builtIn)
                    span(
                      cls := "tag is-small",
                      title := "Built-in models cannot be edited or deleted: " +
                        "they are rewritten from drift's reference on every " +
                        "start. Override their parameters on a run " +
                        "configuration, and delete the file itself in Model " +
                        "Cache → On disk.",
                      "built-in"
                    )
                  else
                    List(
                      button(
                        cls := "button is-info is-small mr-1",
                        title := "Edit this model: its file, its label and " +
                          "the parameters it asks for",
                        "\u270F\uFE0F",
                        onClick --> (_ => startEdit(m))
                      ),
                      button(
                        cls := "button is-danger is-small",
                        "\uD83D\uDDD1\uFE0F",
                        onClick --> (_ => onDeleteModel(m.id))
                      )
                    )
                )
              )
            }
          }
        }
      ),
      // One panel at a time, under the models it belongs to: only one
      // checkpoint is expanded, and opening either form closes the other.
      child <-- editForm.signal.map {
        case None            => emptyNode
        case Some((_, form)) =>
          div(
            cls := "mt-2 mb-3",
            form.element,
            div(
              cls := "buttons mt-2",
              button(
                cls := "button is-small is-success",
                "Save",
                disabled <-- form.ready.map(!_),
                onClick --> (_ => handleEdit())
              ),
              button(
                cls := "button is-small",
                "Cancel",
                onClick --> (_ => editForm.set(None))
              )
            )
          )
      },
      child <-- addForm.signal.map {
        case Some((`idx`, form)) =>
          div(
            cls := "mt-2",
            form.element,
            div(
              cls := "buttons mt-2",
              button(
                cls := "button is-small is-success",
                span(cls := "plus-icon", "+"),
                " Add",
                disabled <-- form.ready.map(!_),
                onClick --> (_ => handleAdd())
              ),
              button(
                cls := "button is-small",
                "Cancel",
                onClick --> (_ => addForm.set(None))
              )
            )
          )
        case _ =>
          button(
            cls := "button is-small is-info",
            span(cls := "plus-icon", "+"),
            " Add Model",
            onClick --> (_ => {
              editForm.set(None)
              addForm.set(
                Some(
                  (
                    idx,
                    ModelForm(
                      browsers,
                      a,
                      cp.familyId,
                      allModels,
                      onCancel = () => addForm.set(None)
                    )
                  )
                )
              )
            })
          )
      }
    )
  }

  private def handleAdd(): Unit =
    addForm.now().foreach { case (_, form) =>
      form.snapshot().foreach { model =>
        onAddModel(model)
        addForm.set(None)
      }
    }

  private def startEdit(model: Model): Unit = {
    addForm.set(None)
    editForm.set(
      Some(
        (
          model.id,
          ModelForm(
            browsers,
            a,
            model.familyId,
            allModels,
            onCancel = () => editForm.set(None),
            editing = Some(model)
          )
        )
      )
    )
  }

  private def handleEdit(): Unit =
    editForm.now().foreach { case (_, form) =>
      form.snapshot().foreach { model =>
        onSaveModel(model)
        editForm.set(None)
      }
    }

  private def optionRow(
      label: String,
      params: Map[String, String]
  ): Mod[HtmlElement] = {
    if (params.isEmpty) emptyNode
    else {
      div(
        cls := "mt-3",
        p(cls := "has-text-weight-bold text-secondary is-size-7", label),
        ul(
          params.toList.map { case (k, v) =>
            li(code(cls := "is-size-7 text-break", s"$k $v"))
          }
        )
      )
    }
  }

  private def baseModelsColumn: HtmlElement = {
    div(
      cls := "mt-3",
      div(
        cls := "mt-3",
        CivitaiBaseModelsEditor(
          baseModelManager,
          onRemove = bm => {
            removeBaseModel(bm)
          },
          onSave = () => {
            val updated = baseModelManager.snapshot()
            val arch = a.copy(civitaiBaseModels = updated)
            onSave(arch)
          }
        ).element
      )
    )
  }

  private def removeBaseModel(name: String): Unit = {
    baseModelManager.removeBaseModel(name)
    val updated = baseModelManager.snapshot()
    val arch = a.copy(civitaiBaseModels = updated)
    onSave(arch)
  }

}
