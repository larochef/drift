package drift.frontend.gallery

import drift.frontend.services.ProjectService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What a view of the gallery is filtered by — the page's toolbar, the picker's
  * — and what follows from it: which generations are kept, and whether anything
  * may be hidden.
  */
class GalleryFilterState(projectService: ProjectService) {

  val configurationFilter = Var("")
  val kindFilter = Var("")
  val searchVar = Var("")

  // Declared before `filter`, which reads it: a val referenced before its
  // own initialisation is null at bind.
  val projectFilter = Var("")

  /** NSFW projects' images stay out of the grid unless asked for, as NSFW
    * projects stay out of the projects list; the choice lasts for the view's
    * life.
    */
  val showNsfw = Var(false)

  /** The projects flagged NSFW. A generation carries its project, and so does
    * anything derived from it, so the flag reaches upscales and redraws too.
    */
  val nsfwProjectIds: Signal[Set[String]] =
    projectService.projects.map(_.filter(_.nsfw).map(_.id).toSet).distinct

  val filter: Signal[Generation => Boolean] = Signal
    .combine(
      configurationFilter.signal,
      kindFilter.signal,
      searchVar.signal,
      projectFilter.signal,
      showNsfw.signal,
      nsfwProjectIds
    )
    .map((configuration, kind, search, project, nsfwShown, nsfwProjects) =>
      GalleryFilter.matches(
        configuration,
        kind,
        search,
        project,
        nsfwShown,
        nsfwProjects
      )
    )

  /** Whether a filter may be hiding something: then a day cannot be judged by
    * its count, so every day is loaded and shown only for what it has that
    * matches — a day left with nothing does not appear at all. NSFW projects
    * kept out count as a filter.
    */
  val filtering: Signal[Boolean] = Signal
    .combine(
      configurationFilter.signal,
      kindFilter.signal,
      searchVar.signal,
      projectFilter.signal,
      showNsfw.signal,
      nsfwProjectIds
    )
    .map((configuration, kind, search, project, nsfwShown, nsfwProjects) =>
      configuration.nonEmpty || kind.nonEmpty || search.trim.nonEmpty ||
        project.nonEmpty || (!nsfwShown && nsfwProjects.nonEmpty)
    )
    .distinct

  /** Every filter back to showing all, on `project` when one is given — with
    * its entries shown when it is an NSFW one.
    */
  def reset(project: String, nsfw: Boolean): Unit = {
    projectFilter.set(project)
    configurationFilter.set("")
    kindFilter.set("")
    searchVar.set("")
    showNsfw.set(nsfw)
  }
}
