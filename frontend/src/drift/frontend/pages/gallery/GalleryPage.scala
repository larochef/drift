package drift.frontend.pages.gallery

import drift.frontend.Page
import drift.frontend.components.{Component, ErrorBanner}
import drift.frontend.pages.generate.readAsDataUrl
import drift.frontend.pages.projects.NewProjectModal
import drift.frontend.services.*
import drift.frontend.services.HistoryService.{Command, Event}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom
import org.scalajs.dom.window

/** The gallery (`specs/12-gallery.md`): every generation drift ever recorded,
  * grouped by day, newest first, rebuilt from the sidecars on disk — so it is
  * complete after a restart and needs no session. Its own sidebar entry since
  * 2026-09-10: it is somewhere you go, not a detour off the page that happened
  * to make the images.
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
    /** What each configuration still has to download before it can launch. */
    prerequisites: LaunchPrerequisites,
    /** The path past `/gallery`: the generation open in the detail view, and
      * which of its outputs (`GenerationDetailHost.boundToUrl`).
      */
    section: Signal[List[String]]
) extends Component {

  private val configurationFilter = Var("")
  private val kindFilter = Var("")
  private val searchVar = Var("")

  /** The open detail, by id: it follows the listing, so a deletion closes it
    * and a socket update cannot show a stale copy.
    */
  private val openDetail = Var(Option.empty[GenerationDetailHost.Open])
  private val initialLoadRequested = Var(false)

  /** Clearing out a day's worth of misfires one confirmation at a time is the
    * wrong tool (François, 2026-09-09): while selecting, a click ticks a card
    * instead of opening it, and one button deletes everything ticked. The ids
    * are enough — the day comes off the generation when the delete goes out.
    */
  private val selecting = Var(false)
  private val selection = Var(Set.empty[String])

  /** Starting a project from the gallery's invitations (François, 2026-09-28):
    * the projects page's own modal, opening the new workspace.
    */
  private val creatingProject = Var(false)
  private lazy val newProject =
    NewProjectModal(projectService, () => creatingProject.set(false))

  private val labels: Signal[Map[String, String]] =
    runConfigurationService.runConfigurations
      .map(_.map(rm => rm.id -> rm.label).toMap)

  /** An imported image, and whatever is made from it, has no configuration
    * (`specs/30`).
    */
  private def labelOf(labels: Map[String, String], id: String): String =
    if (id.isEmpty) "Imported" else labels.getOrElse(id, id)

  // Declared before `filter`, which reads them: a val referenced before its
  // own initialisation is null at bind (the trap of bugs/15's family).
  /** Project labels by id, mirrored for the cards' badges. */
  private val projectLabels = Var(Map.empty[String, String])
  private val projectFilter = Var("")

  /** NSFW projects' images stay out of the grid unless asked for, as NSFW
    * projects stay out of the projects list (François, 2026-09-14); the choice
    * lasts for the page's life.
    */
  private val showNsfw = Var(false)

  /** The projects flagged NSFW. A generation carries its project, and so does
    * anything derived from it, so the flag reaches upscales and redraws too.
    */
  private val nsfwProjectIds: Signal[Set[String]] =
    projectService.projects.map(_.filter(_.nsfw).map(_.id).toSet).distinct

  private val filter: Signal[Generation => Boolean] = Signal
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
    * matches — a day left with nothing does not appear at all (François,
    * 2026-10-02). NSFW projects kept out count as a filter.
    */
  private val filtering: Signal[Boolean] = Signal
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

  /** The days a filter has asked for, each once: a day that fails to load is
    * not asked for again on every change, and its own button remains.
    */
  private var readRequested = Set.empty[String]

  /** Everything the loaded days hold — the pool the detail resolves lineage in
    * and the bulk delete reads its dates from.
    */
  private val loaded: Signal[List[Generation]] =
    historyService.generationsByDay.map(_.values.flatten.toList)

  /** The configurations seen in the loaded days — what the filter offers. */
  private val configurationOptions: Signal[List[(String, String)]] =
    historyService.generationsByDay.combineWith(labels).map { (byDay, labels) =>
      byDay.values.flatten
        .map(_.runConfigurationId)
        // Imports have none, and "" is already "All configurations".
        .filter(_.nonEmpty)
        .toList
        .distinct
        .map(id => id -> labelOf(labels, id))
        .sortBy(_._2)
    }

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

  private def toggleSelecting(): Unit = {
    selection.set(Set.empty)
    selecting.update(!_)
  }

  /** One confirmation for the whole sweep, naming what it costs on disk; the
    * deletes then go out one per generation, the same command a single delete
    * uses.
    */
  private def deleteSelected(loaded: List[Generation]): Unit = {
    val chosen = loaded.filter(generation => selection.now()(generation.id))
    if (chosen.nonEmpty) {
      val files = chosen.map(GenerationDetailHost.fileCount).sum
      val confirmed = window.confirm(
        s"Delete ${chosen.size} generation${
            if (chosen.size == 1) "" else "s"
          }" +
          s" and their $files file(s)?\n\nThis cannot be undone."
      )
      if (confirmed) {
        chosen.foreach(GenerationDetailHost.delete(_, historyService))
        selection.set(Set.empty)
      }
    }
  }

  /** One day's group. Keyed rendering throughout: a card is built once per
    * generation id and survives socket pushes, filter changes and other days
    * loading, so thumbnails are not re-requested on every update.
    */
  private def daySection(
      date: String,
      daySignal: Signal[HistoryDay]
  ): HtmlElement = {
    val loaded: Signal[Option[List[Generation]]] =
      historyService.generationsByDay.map(_.get(date)).distinct
    val loading: Signal[Boolean] =
      historyService.loadingDays.map(_.contains(date)).distinct
    val shown: Signal[List[Generation]] =
      loaded.combineWith(filter).map { (loaded, matches) =>
        loaded.getOrElse(List.empty).filter(matches)
      }
    div(
      cls := "gallery-day",
      dataAttr("date") := date,
      // Under a filter a day shows for what it has that matches: not while it
      // is still being read, and not at all with nothing to show.
      display <-- loaded.combineWith(shown, filtering, loading).map {
        case (Some(_), Nil, _, _)  => "none"
        case (None, _, true, true) => "none"
        case _                     => ""
      },
      div(
        cls := "gallery-day-heading",
        h2(cls := "subtitle text-primary mb-0", date),
        span(
          cls := "tag",
          // What is shown once the day is read; what it holds until then.
          child.text <-- loaded.combineWith(shown, daySignal).map {
            case (Some(_), matching, _) => matching.size.toString
            case (None, _, day)         => day.count.toString
          }
        ),
        child <-- loaded.combineWith(loading, daySignal).map {
          case (None, true, _) =>
            span(cls := "text-secondary is-size-7", "loading…")
          case (None, false, day) =>
            button(
              cls := "button is-small",
              s"Show ${day.count} generation${if (day.count == 1) "" else "s"}",
              onClick --> (_ => historyService.push(Command.LoadDay(date)))
            )
          case (Some(_), _, _) => emptyNode
        },
        child <-- selecting.signal.map {
          case false => emptyNode
          case true  =>
            a(
              cls := "is-size-7",
              "select all of this day",
              onClick.compose(_.sample(shown)) --> Observer[List[Generation]](
                generations => selection.update(_ ++ generations.map(_.id))
              )
            )
        }
      ),
      div(
        cls := "gallery-grid",
        children <-- shown
          .combineWith(labels)
          .map((generations, labels) =>
            // One tile per output: a batch's images each get theirs, side by
            // side (François, 2026-09-14). A generation without outputs still
            // shows as one tile.
            generations.flatMap(generation =>
              (if (generation.outputs.isEmpty) List(0)
               else generation.outputs.indices.toList)
                .map(index => (generation, index, labels))
            )
          )
          .split(tile => (tile._1.id, tile._2)) { (_, initial, _) =>
            val (generation, index, labels) = initial
            val id = generation.id
            GenerationCard(
              generation,
              labelOf(labels, generation.runConfigurationId),
              () => openDetail.set(Some(GenerationDetailHost.Open(id, index))),
              outputIndex = index,
              projectLabel = generation.projectId
                .map(id => projectLabels.signal.map(_.get(id)))
                .getOrElse(Val(None)),
              selecting = selecting.signal,
              selected = selection.signal.map(_(id)),
              onToggleSelected = () =>
                selection.update(current =>
                  if (current(id)) current - id else current + id
                )
            ).element
          }
      )
    )
  }

  private def toolbar: HtmlElement = div(
    cls := "gallery-toolbar mb-4",
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Project"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> projectFilter,
          option(value := "", "All projects"),
          children <-- projectService.projects
            .combineWith(showNsfw.signal)
            .map((projects, nsfwShown) =>
              projects
                .filter(project => nsfwShown || !project.nsfw)
                .map(project =>
                  option(
                    value := project.id,
                    selected <-- projectFilter.signal.map(_ == project.id),
                    project.label
                  )
                )
            )
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "NSFW"),
      label(
        cls := "checkbox text-secondary is-size-7",
        input(
          typ := "checkbox",
          checked <-- showNsfw.signal,
          onChange.mapToChecked.compose(
            _.withCurrentValueOf(nsfwProjectIds)
          ) --> Observer[(Boolean, Set[String])] { case (shown, nsfwProjects) =>
            showNsfw.set(shown)
            // A project that is hidden again cannot stay the one filtered on.
            if (!shown && nsfwProjects(projectFilter.now()))
              projectFilter.set("")
          }
        ),
        " Show NSFW projects"
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Run configuration"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> configurationFilter,
          option(value := "", "All configurations"),
          children <-- configurationOptions.map(_.map { (id, label) =>
            option(
              value := id,
              selected <-- configurationFilter.signal.map(_ == id),
              label
            )
          })
        )
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Kind"),
      div(
        cls := "select is-small",
        select(
          onChange.mapToValue --> kindFilter,
          option(value := "", "Images and videos"),
          option(value := "img_gen", "Images"),
          option(value := "vid_gen", "Videos")
        )
      )
    ),
    div(
      cls := "field",
      styleAttr := "flex: 1 1 16rem;",
      label(cls := "label text-primary is-small", "Search prompts"),
      input(
        cls := "input is-small",
        typ := "search",
        placeholder := "words from the prompt",
        value <-- searchVar.signal,
        onInput.mapToValue --> searchVar
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Import"),
      label(
        cls := "button is-small",
        title := "PNG or JPEG images and WebM, MP4, MOV or Matroska videos " +
          "from your computer, to upscale, redraw, edit or use as inputs — " +
          "or drop them on the gallery",
        input(
          typ := "file",
          accept := "image/png,image/jpeg,video/webm,video/mp4," +
            "video/quicktime,video/x-matroska,.mkv",
          multiple := true,
          styleAttr := "display: none;",
          onChange --> { event =>
            val field = event.target.asInstanceOf[dom.HTMLInputElement]
            importFiles(field.files)
            // The same file picked again must fire again.
            field.value = ""
          }
        ),
        "⤓ Import images or videos"
      )
    ),
    div(
      cls := "field",
      label(cls := "label text-primary is-small", "Cleanup"),
      button(
        cls <-- selecting.signal.map(on =>
          if (on) "button is-small is-primary" else "button is-small"
        ),
        child.text <-- selecting.signal.map(on =>
          if (on) "Done selecting" else "☑ Select"
        ),
        onClick --> (_ => toggleSelecting())
      )
    )
  )

  /** Whether the next import to land opens in the detail view: one picked image
    * is there to be worked on; several are left in the grid.
    */
  private var openImported = false

  /** Images and videos from outside drift, each a gallery entry of its own
    * (`specs/30`). A video is sent as the file it is; anything that does not
    * say it is one goes the images' way, where the server judges the bytes.
    */
  private def importFiles(files: dom.FileList): Unit = {
    val picked = (0 until files.length).map(files(_)).toList
    openImported = picked.size == 1
    picked.foreach(file =>
      if (GalleryPage.isVideo(file))
        historyService.push(Command.ImportVideo(file))
      else
        readAsDataUrl(file)(data =>
          historyService.push(Command.Import(ImageImport(file.name, data)))
        )
    )
  }

  /** The bulk bar, shown only while selecting. */
  private def selectionBar: Node = div(
    child <-- selecting.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "gallery-selection-bar mb-4",
          span(
            cls := "text-primary is-size-7",
            child.text <-- selection.signal.map(chosen =>
              s"${chosen.size} selected"
            )
          ),
          span(
            cls := "text-secondary is-size-7",
            "clicking a card ticks it instead of opening it"
          ),
          button(
            cls := "button is-small",
            "Clear",
            disabled <-- selection.signal.map(_.isEmpty),
            onClick --> (_ => selection.set(Set.empty))
          ),
          ProjectMover(
            projectService.projects,
            targets = loaded.combineWith(selection.signal).map(
              (list, chosen) => list.filter(g => chosen(g.id))
            ),
            state = historyService.moveState,
            onMove = (chosen, projectId) =>
              historyService.push(Command.Move(chosen, projectId)),
            label = count => s"Move $count selected"
          ).element,
          button(
            cls := "button is-small is-danger",
            child.text <-- selection.signal.map(chosen =>
              s"🗑 Delete ${chosen.size} selected"
            ),
            disabled <-- selection.signal.map(_.isEmpty),
            onClick.compose(_.sample(loaded)) --> Observer[List[Generation]](
              deleteSelected
            )
          )
        )
    }
  )

  lazy val element: HtmlElement = div(
    cls := "content",
    GenerationDetailHost.boundToUrl(Page.Gallery.path, section, openDetail),
    historyService.effects,
    runConfigurationService.effects,
    sessionService.effects,
    postProcessService.effects,
    upscalerService.effects,
    projectService.effects,
    prerequisites.effects,
    projectService.projects.map(_.map(p => p.id -> p.label).toMap)
      --> projectLabels,
    // A detail named by the URL — a refresh, a link, the back button — is not
    // necessarily in the days the gallery loads up front; the day holding it
    // is loaded for it, so it opens straight away rather than when the visitor
    // happens to page back to it (François, 2026-09-20). Bound after the
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
      case Event.Deleted(id)          => selection.update(_ - id)
      case Event.Imported(generation) =>
        if (openImported)
          openDetail.set(Some(GenerationDetailHost.Open(generation.id, 0)))
      case Event.Moved(list) =>
        projectService.moved(list)
        selection.update(_ -- list.map(_.id))
    },
    // Images and videos dropped anywhere on the gallery are imported like picked ones.
    onDragOver --> (_.preventDefault()),
    onDrop --> { event =>
      event.preventDefault()
      importFiles(event.dataTransfer.files)
    },
    h1(cls := "title text-primary", "Gallery"),
    hr(),
    ErrorBanner(historyService),
    ErrorBanner(sessionService),
    ErrorBanner(runConfigurationService),
    ErrorBanner(postProcessService),
    ErrorBanner(upscalerService),
    toolbar,
    selectionBar,
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
    filtering.combineWith(
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
      daySection(date, daySignal)
    ),
    child <-- Signal
      .combine(
        filtering,
        filter,
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
      prerequisites,
      projectService.projects,
      launchingProject = None,
      onReuseStaged = () => Page.Sandbox.navigate(),
      onAssistantStaged = () => Page.Sandbox.navigateTo(ProjectKind.Text)
    )
  )
}

object GalleryPage {
  private val videoExtensions = Set("webm", "mp4", "m4v", "mov", "mkv")

  /** Whether a picked or dropped file is a video: by its type, or by its name
    * where the browser gives none (Matroska, often).
    */
  def isVideo(file: dom.File): Boolean =
    file.`type`.startsWith("video/") ||
      videoExtensions.contains(file.name.toLowerCase.split('.').last)
}
