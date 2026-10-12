package drift.frontend.pages.gallery

import drift.frontend.Page
import drift.frontend.components.{Component, ErrorBanner}
import drift.frontend.gallery.GenerationDetailHost
import drift.frontend.services.*
import drift.frontend.services.HistoryService.{Command, Event}
import drift.frontend.workspace.NewProjectModal
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The gallery (`specs/12-gallery.md`): every generation drift ever recorded,
  * grouped by day, newest first, rebuilt from the sidecars on disk — so it is
  * complete after a restart and needs no session. Its own sidebar entry since
  * 2026-09-10: it is somewhere you go, not a detour off the page that happened
  * to make the images.
  *
  * The page wires the pieces beside it: the labels (`GalleryLabels`), the
  * filters (`GalleryFilters`) and their toolbar (`GalleryToolbar`), the bulk
  * selection (`GallerySelection`), the imports (`GalleryImport`) and each day's
  * group (`GalleryDaySection`); what stays here is which days are read, and the
  * detail view.
  */
class GalleryPage(
    historyService: HistoryService,
    runConfigurationService: RunConfigurationService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    projectService: ProjectService,
    postProcessService: PostProcessService,
    upscalerService: UpscalerService,
    runtimeService: RuntimeService,
    machineService: MachineService,
    /** What each configuration still has to download before it can launch. */
    prerequisites: LaunchPrerequisites,
    /** The path past `/gallery`: the generation open in the detail view, and
      * which of its outputs (`GenerationDetailHost.boundToUrl`).
      */
    section: Signal[List[String]]
) extends Component {

  /** The open detail, by id: it follows the listing, so a deletion closes it
    * and a socket update cannot show a stale copy.
    */
  private val openDetail = Var(Option.empty[GenerationDetailHost.Open])
  private val initialLoadRequested = Var(false)

  /** Starting a project from the gallery's invitations: the projects page's own
    * modal, opening the new workspace.
    */
  private val creatingProject = Var(false)
  private lazy val newProject =
    NewProjectModal(projectService, () => creatingProject.set(false))

  private val labels = GalleryLabels(runConfigurationService)
  private val filters = GalleryFilters(historyService, projectService, labels)
  private val importer = GalleryImport(historyService)

  /** The days a filter has asked for, each once: a day that fails to load is
    * not asked for again on every change, and its own button remains.
    */
  private var readRequested = Set.empty[String]

  /** Everything the loaded days hold — the pool the detail resolves lineage in
    * and the bulk delete reads its dates from.
    */
  private val loaded: Signal[List[Generation]] =
    historyService.generationsByDay.map(_.values.flatten.toList)

  private val selection =
    GallerySelection(loaded, historyService, projectService.projects)

  private val toolbar =
    GalleryToolbar(filters, projectService.projects, selection, importer)

  /** The newest days, until two dozen generations are covered — enough to fill
    * the screen without reading a month of sidecars up front.
    */
  private def initialDays(days: List[HistoryDay]): List[String] =
    days
      .foldLeft((List.empty[String], 0)) { case ((chosen, total), day) =>
        if (chosen.isEmpty || total < 24)
          (chosen :+ day.date, total + day.count)
        else (chosen, total)
      }
      ._1

  lazy val element: HtmlElement = div(
    cls := "content",
    GenerationDetailHost.boundToUrl(Page.Gallery.path, section, openDetail),
    prerequisites.loads,
    projectService.projects.map(_.map(p => p.id -> p.label).toMap)
      --> labels.projectLabels,
    // A detail named by the URL — a refresh, a link, the back button — is not
    // necessarily in the days the gallery loads up front; the day holding it
    // is loaded for it, so it opens straight away rather than when the visitor
    // happens to page back to it. Bound after the
    // services, whose effects are what the command is pushed into.
    openDetail.signal
      .combineWith(loaded)
      .map((open, pool) =>
        open.map(_.generationId).filterNot(id => pool.exists(_.id == id))
      )
      .distinct --> Observer[Option[String]](
      _.foreach(id => historyService.push(Command.LoadGeneration(id)))
    ),
    onMountCallback { _ =>
      historyService.push(Command.LoadDays)
      projectService.push(ProjectService.Command.Load)
      runConfigurationService.push(RunConfigurationService.Command.Load)
      sessionService.push(SessionService.Command.Load)
      postProcessService.push(PostProcessService.Command.LoadJobs)
      upscalerService.push(UpscalerService.Command.Load)
      // The PiD picker needs the architectures to know which configurations
      // are decoders.
      runConfigurationService.architectureService
        .push(drift.frontend.services.ArchitectureService.Command.Load)
    },
    // A finished upscale/resize is a new entry beside its source.
    postProcessService.completions --> Observer[Generation](
      historyService.adopt
    ),
    // Days already loaded on an earlier visit are refreshed by the service
    // itself; only the rest of the initial set is requested here.
    historyService.days.changes
      .withCurrentValueOf(historyService.generationsByDay) --> Observer[
      (List[HistoryDay], Map[String, List[Generation]])
    ] { (days, loaded) =>
      if (!initialLoadRequested.now()) {
        initialLoadRequested.set(true)
        initialDays(days)
          .filterNot(loaded.contains)
          .foreach(date => historyService.push(Command.LoadDay(date)))
      }
    },
    // A gone generation cannot stay ticked; the detail's own closing is the
    // host's job.
    historyService.events --> Observer[Event] {
      case Event.Deleted(id)          => selection.selection.update(_ - id)
      case Event.Imported(generation) =>
        if (importer.opensImported)
          openDetail.set(Some(GenerationDetailHost.Open(generation.id, 0)))
      case Event.Moved(list) =>
        projectService.moved(list)
        selection.selection.update(_ -- list.map(_.id))
    },
    // Images and videos dropped anywhere on the gallery are imported like picked ones.
    onDragOver --> (_.preventDefault()),
    onDrop --> { event =>
      event.preventDefault()
      importer.importFiles(event.dataTransfer.files)
    },
    h1(cls := "title text-primary", "Gallery"),
    hr(),
    ErrorBanner(historyService),
    ErrorBanner(sessionService),
    ErrorBanner(runConfigurationService),
    ErrorBanner(postProcessService),
    ErrorBanner(upscalerService),
    toolbar.element,
    selection.element,
    child <-- creatingProject.signal.map(
      if (_) newProject.element else emptyNode
    ),
    child <-- historyService.daysLoaded
      .combineWith(
        historyService.days.map(_.isEmpty),
        projectService.projectsLoaded,
        projectService.projects.map(_.isEmpty)
      )
      .distinct
      .map {
        case (true, true, _, _) =>
          NewProjectModal.invitation(
            "Nothing generated yet",
            List(
              "Everything you generate lands here, grouped by day, whichever " +
                "model made it: open one to reuse its settings, upscale it, " +
                "redraw or edit it.",
              "The best place to start is a project: it keeps each version " +
                "of your prompt with what it made, and the assistant helps " +
                "you write them. You can also just try a model on its own."
            ),
            () => creatingProject.set(true),
            Some("Just try a model" -> (() => Page.Sandbox.navigate()))
          )
        case (true, false, true, true) =>
          div(
            cls := "notification bg-card mb-4 is-flex is-align-items-center " +
              "is-flex-wrap-wrap",
            styleAttr := "gap: 0.75rem;",
            p(
              cls := "text-secondary mb-0",
              styleAttr := "flex: 1 1 20rem;",
              "None of this is in a project yet. A project keeps every " +
                "version of a prompt beside what it made, so you can compare " +
                "and go back — and the assistant reads its brief."
            ),
            button(
              cls := "button is-primary",
              span(cls := "plus-icon", "+"),
              " Create a project",
              onClick --> (_ => creatingProject.set(true))
            )
          )
        case _ => emptyNode
      },
    // Under a filter every day is read, so each can be judged.
    filters.filtering.combineWith(
      historyService.days,
      historyService.generationsByDay,
      historyService.loadingDays
    ) --> Observer[
      (
          Boolean,
          List[HistoryDay],
          Map[String, List[Generation]],
          Set[String]
      )
    ] { (filtering, days, byDay, loading) =>
      if (filtering)
        days
          .map(_.date)
          .filterNot(date =>
            byDay.contains(date) || loading(date) || readRequested(date)
          )
          .foreach { date =>
            readRequested += date
            historyService.push(Command.LoadDay(date))
          }
    },
    children <-- historyService.days.split(_.date)((date, _, daySignal) =>
      GalleryDaySection(
        date,
        daySignal,
        historyService,
        filters,
        labels,
        selection,
        open => openDetail.set(Some(open))
      ).element
    ),
    child <-- Signal
      .combine(
        filters.filtering,
        filters.filter,
        historyService.days,
        historyService.generationsByDay,
        historyService.loadingDays
      )
      .map { (filtering, matches, days, byDay, loading) =>
        if (!filtering || days.isEmpty) emptyNode
        else if (loading.nonEmpty)
          p(cls := "text-secondary", "Reading the gallery…")
        else if (byDay.values.exists(_.exists(matches))) emptyNode
        else p(cls := "text-secondary", "No generation matches the filters.")
      },
    child <-- historyService.days
      .combineWith(historyService.generationsByDay)
      .map { (days, byDay) =>
        val unloaded = days.map(_.date).filterNot(byDay.contains)
        if (unloaded.isEmpty) emptyNode
        else
          button(
            cls := "button mt-4",
            s"Show all remaining days (${unloaded.size})",
            onClick --> (_ =>
              unloaded
                .foreach(date => historyService.push(Command.LoadDay(date)))
            )
          )
      },
    GenerationDetailHost(
      openDetail,
      loaded,
      historyService,
      runConfigurationService,
      sessionService,
      generationService,
      assistantService,
      postProcessService,
      upscalerService,
      runtimeService,
      machineService,
      prerequisites,
      projectService.projects,
      launchingProject = None,
      onReuseStaged = () => Page.Sandbox.navigate(),
      onAssistantStaged = () => Page.Sandbox.navigateTo(ProjectKind.Text)
    )
  )
}
