package drift.frontend.pages.gallery

import drift.frontend.gallery.GalleryFilterState
import drift.frontend.services.{HistoryService, ProjectService}

import com.raquo.laminar.api.L.*

/** What the gallery's toolbar is set to (`GalleryFilterState`), and the
  * configurations it offers.
  */
final class GalleryFilters(
    historyService: HistoryService,
    projectService: ProjectService,
    labels: GalleryLabels
) extends GalleryFilterState(projectService) {

  /** The configurations seen in the loaded days — what the filter offers. */
  val configurationOptions: Signal[List[(String, String)]] =
    historyService.generationsByDay.combineWith(labels.labels).map {
      (byDay, known) =>
        byDay.values.flatten
          .map(_.runConfigurationId)
          // Imports have none, and "" is already "All configurations".
          .filter(_.nonEmpty)
          .toList
          .distinct
          .map(id => id -> labels.labelOf(known, id))
          .sortBy(_._2)
    }
}
