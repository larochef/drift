package drift.frontend.pages.gallery

import drift.frontend.services.RunConfigurationService

import com.raquo.laminar.api.L.*

/** The names the gallery shows for what a generation only carries the id of:
  * its run configuration and its project.
  */
final class GalleryLabels(runConfigurationService: RunConfigurationService) {

  val labels: Signal[Map[String, String]] =
    runConfigurationService.runConfigurations
      .map(_.map(rm => rm.id -> rm.label).toMap)

  /** An imported image, and whatever is made from it, has no configuration
    * (`specs/30`).
    */
  def labelOf(labels: Map[String, String], id: String): String =
    if (id.isEmpty) "Imported" else labels.getOrElse(id, id)

  /** Project labels by id, mirrored for the cards' badges. */
  val projectLabels = Var(Map.empty[String, String])
}
