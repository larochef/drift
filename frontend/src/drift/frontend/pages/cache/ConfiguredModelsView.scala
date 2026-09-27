package drift.frontend.pages.cache

import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Every registered model grouped by the architecture whose slots take its
  * family, dimmed with a download control when absent
  * (`specs/05-model-cache-and-downloads.md`) — "what do I still need before I
  * can run this?".
  */
class ConfiguredModelsView(
    architectureService: ArchitectureService,
    modelService: ModelService,
    cacheService: CacheService,
    downloadService: DownloadService
) extends Component {

  private def modelRow(
      model: Model,
      slot: String,
      shared: Boolean,
      statuses: Map[String, ModelCacheStatus],
      jobs: Map[String, DownloadJob]
  ): HtmlElement = {
    val missing =
      statuses.get(model.id).exists(_.state != CacheState.Cached) &&
        !jobs.get(model.id).exists(_.state.isActive)
    div(
      cls := "level is-mobile mb-1 is-marginless",
      // Not-downloaded entries read as disabled, per the spec.
      styleAttr := (if (missing) "opacity: 0.55;" else ""),
      div(
        cls := "level-left",
        div(
          p(
            cls := "text-primary is-size-7",
            model.label,
            if (shared) span(cls := "tag is-info is-small ml-2", "shared")
            else emptyNode
          ),
          p(cls := "text-secondary is-size-7", s"$slot · ${model.familyId}")
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
      .combine(
        architectureService.architectures,
        modelService.allModels,
        cacheService.statuses,
        downloadService.jobs
      )
      .map { (architectures, models, statuses, jobs) =>
        val familyUseCount = architectures
          .flatMap(_.checkpoints.map(_.familyId).distinct)
          .groupBy(identity)
          .view
          .mapValues(_.size)
          .toMap
        val usedFamilies = familyUseCount.keySet

        val architectureSections = architectures.map { architecture =>
          val rows = architecture.checkpoints.flatMap { checkpoint =>
            models
              .filter(_.familyId == checkpoint.familyId)
              .map(model =>
                modelRow(
                  model,
                  checkpoint.name,
                  shared = familyUseCount.getOrElse(checkpoint.familyId, 0) > 1,
                  statuses,
                  jobs
                )
              )
          }
          div(
            cls := "mb-4",
            h2(
              cls := "is-size-6 has-text-weight-bold text-primary mb-1",
              architecture.label
            ),
            if (rows.isEmpty)
              p(cls := "text-secondary is-size-7", "No models registered.")
            else rows
          )
        }

        val unassigned =
          models.filterNot(m => usedFamilies.contains(m.familyId))
        val unassignedSection =
          if (unassigned.isEmpty) Nil
          else
            List(
              div(
                cls := "mb-4",
                h2(
                  cls := "is-size-6 has-text-weight-bold text-secondary mb-1",
                  "Not used by any architecture"
                ),
                unassigned.map(
                  modelRow(_, "—", shared = false, statuses, jobs)
                )
              )
            )
        architectureSections ++ unassignedSection
      }
  )
}
