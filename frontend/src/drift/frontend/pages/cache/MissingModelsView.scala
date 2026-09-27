package drift.frontend.pages.cache

import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The answer to "what did I declare but never fetch"
  * (`specs/05-model-cache-and-downloads.md`): each registered model whose
  * weights are missing, with its download control, and one button to start them
  * all.
  */
class MissingModelsView(
    /** Registered models whose weights are not on disk. */
    missingModels: Signal[List[Model]],
    cacheService: CacheService,
    downloadService: DownloadService
) extends Component {

  private def row(
      model: Model,
      statuses: Map[String, ModelCacheStatus],
      jobs: Map[String, DownloadJob]
  ): HtmlElement = {
    val sourceText = model.source match {
      case HuggingFace(repo, file)        => s"HuggingFace $repo — $file"
      case ModelScope(repo, file)         => s"ModelScope $repo — $file"
      case Civitai(modelId, _, fileId, f) => s"Civitai $modelId/$fileId — $f"
      case Local(path)                    => s"Local $path"
    }
    div(
      cls := "level is-mobile mb-1 is-marginless",
      div(
        cls := "level-left",
        div(
          p(
            cls := "text-primary is-size-7",
            ProviderIcon.of(model.source).amend(cls := "mr-2"),
            model.label
          ),
          p(
            cls := "text-secondary is-size-7 text-break",
            s"${model.familyId} · $sourceText"
          )
        )
      ),
      div(
        cls := "level-right",
        CacheIndicator(
          model.id,
          statuses.get(model.id),
          jobs.get(model.id),
          id => downloadService.push(DownloadService.Command.Start(id)),
          id => downloadService.push(DownloadService.Command.Cancel(id))
        )
      )
    )
  }

  lazy val element: HtmlElement = div(
    children <-- Signal
      .combine(missingModels, cacheService.statuses, downloadService.jobs)
      .map { (missing, statuses, jobs) =>
        if (missing.isEmpty)
          List(
            p(
              cls := "text-secondary",
              "Every registered model is on disk."
            )
          )
        else {
          val idle = missing
            .filterNot(model => jobs.get(model.id).exists(_.state.isActive))
          val header = div(
            cls := "level is-mobile mb-3",
            div(
              cls := "level-left",
              p(
                cls := "text-secondary",
                s"${missing.size} model(s) declared but not downloaded."
              )
            ),
            div(
              cls := "level-right",
              if (idle.isEmpty) emptyNode
              else
                button(
                  cls := "button is-info is-small",
                  s"⬇ Download all (${idle.size})",
                  onClick --> (_ =>
                    idle.foreach(model =>
                      downloadService
                        .push(DownloadService.Command.Start(model.id))
                    )
                  )
                )
            )
          )
          header :: missing.map(row(_, statuses, jobs))
        }
      }
  )
}
