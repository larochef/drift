package drift.frontend.pages.gallery

import drift.frontend.components.{BrowserModal, Component}
import drift.frontend.services.*
import drift.frontend.services.HistoryService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The gallery as a place to pick inputs from
  * (`specs/50-inputs-from-the-gallery.md`): everything it holds that matches
  * its filters, newest first in one grid, over the page that asked. One tile is
  * picked with a click; a slot that takes several ticks them, in the order they
  * are to be used.
  *
  * One picker serves a page: whoever needs an input calls `open` with what the
  * slot takes and what to do with the outputs picked. The page mounts `element`
  * and the history's effects.
  */
class GalleryPicker(
    historyService: HistoryService,
    projectService: ProjectService,
    runConfigurationService: RunConfigurationService
) extends Component {
  import GalleryPicker.*

  private val request = Var(Option.empty[Request])

  // Mirror, to open on an NSFW project with its entries shown.
  private val projectsNow = Var(List.empty[Project])

  private val projectFilter = Var("")
  private val configurationFilter = Var("")
  private val searchVar = Var("")
  private val showNsfw = Var(false)

  /** The days asked for since the picker opened, each once: a day that fails to
    * load is not asked for again on every change.
    */
  private var readRequested = Set.empty[String]

  /** The outputs ticked, in the order they were. */
  private val ticked = Var(List.empty[Picked])

  def open(asked: Request): Unit = {
    val project = asked.projectId.getOrElse("")
    projectFilter.set(project)
    configurationFilter.set("")
    searchVar.set("")
    showNsfw.set(projectsNow.now().exists(p => p.id == project && p.nsfw))
    ticked.set(List.empty)
    readRequested = Set.empty
    request.set(Some(asked))
    historyService.push(Command.LoadDays)
  }

  private def close(): Unit = request.set(None)

  private def pick(asked: Request, picked: List[Picked]): Unit = {
    close()
    if (picked.nonEmpty) asked.onPicked(picked)
  }

  private val labels: Signal[Map[String, String]] =
    runConfigurationService.runConfigurations
      .map(
        _.map(configuration => configuration.id -> configuration.label).toMap
      )

  private val projectLabels: Signal[Map[String, String]] =
    projectService.projects.map(_.map(p => p.id -> p.label).toMap)

  private val filter: Signal[Generation => Boolean] = Signal
    .combine(
      configurationFilter.signal,
      searchVar.signal,
      projectFilter.signal,
      showNsfw.signal,
      projectService.projects.map(_.filter(_.nsfw).map(_.id).toSet).distinct
    )
    .map((configuration, search, project, nsfwShown, nsfwProjects) =>
      GalleryFilter
        .matches(configuration, "", search, project, nsfwShown, nsfwProjects)
    )

  private def filters: HtmlElement = div(
    cls := "is-flex is-flex-wrap-wrap is-align-items-flex-end mb-3",
    styleAttr := "gap: 0.75rem;",
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
    ),
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> configurationFilter,
        option(value := "", "All configurations"),
        children <-- labels.map(
          _.toList
            .sortBy(_._2)
            .map((id, label) =>
              option(
                value := id,
                selected <-- configurationFilter.signal.map(_ == id),
                label
              )
            )
        )
      )
    ),
    input(
      cls := "input is-small",
      styleAttr := "flex: 1 1 12rem;",
      typ := "search",
      placeholder := "Search prompts",
      value <-- searchVar.signal,
      onInput.mapToValue --> searchVar
    ),
    label(
      cls := "checkbox text-secondary is-size-7",
      input(
        typ := "checkbox",
        cls := "mr-1",
        checked <-- showNsfw.signal,
        onChange.mapToChecked --> Observer[Boolean] { shown =>
          showNsfw.set(shown)
          // A project that is hidden again cannot stay the one filtered on.
          if (
            !shown &&
            projectsNow.now().exists(p => p.id == projectFilter.now() && p.nsfw)
          ) projectFilter.set("")
        }
      ),
      "NSFW"
    )
  )

  /** Every output the slot can take among what matches the filters, newest
    * first across the days: the picker reads the whole gallery, so nothing
    * hides behind a day to open (François, 2026-10-02).
    */
  private def tiles(asked: Request): Signal[List[Picked]] =
    Signal
      .combine(
        historyService.days,
        historyService.generationsByDay,
        filter,
        labels
      )
      .map { (days, byDay, matches, labels) =>
        days
          .flatMap(day => byDay.getOrElse(day.date, List.empty))
          .filter(matches)
          .flatMap { generation =>
            val label =
              if (generation.runConfigurationId.isEmpty) "Imported"
              else
                labels.getOrElse(
                  generation.runConfigurationId,
                  generation.runConfigurationId
                )
            generation.outputs
              .filter(output => asked.takes(output))
              .map(Picked(generation, _, label))
          }
      }

  /** Whether every day has been read. */
  private val read: Signal[Boolean] =
    historyService.daysLoaded
      .combineWith(
        historyService.days,
        historyService.generationsByDay,
        historyService.loadingDays
      )
      .map((listed, days, byDay, loading) =>
        listed && loading.isEmpty &&
          days
            .forall(day => byDay.contains(day.date) || readRequested(day.date))
      )
      .distinct

  private def grid(asked: Request): HtmlElement = {
    val shown = tiles(asked)
    div(
      child <-- read.combineWith(shown.map(_.isEmpty).distinct).map {
        case (false, _)   => p(cls := "text-secondary", "Reading the gallery…")
        case (true, true) =>
          p(
            cls := "text-secondary",
            s"No ${asked.noun} of the gallery matches the filters."
          )
        case _ => emptyNode
      },
      div(
        cls := "gallery-grid",
        children <-- shown.split(_.key) { (_, tile, _) =>
          val generation = tile.generation
          GenerationCard(
            generation,
            tile.configurationLabel,
            () => pick(asked, List(tile)),
            outputIndex = generation.outputs.indexOf(tile.output),
            projectLabel = generation.projectId
              .map(id => projectLabels.map(_.get(id)))
              .getOrElse(Val(None)),
            selecting = Val(asked.multiple),
            selected = ticked.signal.map(_.exists(_.key == tile.key)),
            onToggleSelected = () =>
              ticked.update(current =>
                if (current.exists(_.key == tile.key))
                  current.filterNot(_.key == tile.key)
                else current :+ tile
              )
          ).element
        }
      )
    )
  }

  private def modal(asked: Request): HtmlElement =
    BrowserModal(
      title = Val(s"Pick ${asked.title} from the gallery"),
      body = Seq(
        filters,
        grid(asked)
      ),
      onCancel = () => close(),
      footerLeft = Option.when(asked.multiple)(
        div(
          cls := "is-flex is-align-items-center",
          styleAttr := "gap: 0.25rem;",
          children <-- ticked.signal.map(
            _.map(tile =>
              img(
                src := (
                  if (GenerationMediaViewer.isVideo(tile.output))
                    GenerationMediaViewer.stillUrl(tile.output)
                  else tile.output.url
                ),
                styleAttr :=
                  "max-height: 40px; max-width: 60px; border-radius: 3px;"
              )
            )
          )
        )
      ),
      footerRight = div(
        cls := "buttons mb-0",
        BrowserModal.cancelButton(() => close()),
        Option.when(asked.multiple)(
          button(
            cls := "button is-primary",
            disabled <-- ticked.signal.map(_.isEmpty),
            child.text <-- ticked.signal.map(picked =>
              if (picked.isEmpty) "Use" else s"Use ${picked.size}"
            ),
            onClick --> (_ => pick(asked, ticked.now()))
          )
        )
      ),
      cardMods = Seq(styleAttr := "width: min(72rem, 94vw);")
    ).element

  lazy val element: HtmlElement = div(
    projectService.projects --> projectsNow,
    // Opened, the picker reads every day: its grid is the whole gallery.
    historyService.days
      .combineWith(
        request.signal.map(_.isDefined),
        historyService.generationsByDay,
        historyService.loadingDays
      ) --> Observer[
      (
          List[HistoryDay],
          Boolean,
          Map[String, List[Generation]],
          Set[String]
      )
    ] { (days, opened, loaded, loading) =>
      if (opened)
        days
          .map(_.date)
          .filterNot(date =>
            loaded.contains(date) || loading(date) || readRequested(date)
          )
          .foreach { date =>
            readRequested += date
            historyService.push(Command.LoadDay(date))
          }
    },
    child <-- request.signal.map {
      case Some(asked) => modal(asked)
      case None        => emptyNode
    }
  )
}

object GalleryPicker {

  /** What a slot asks of the gallery. */
  case class Request(
      /** What the slot's file input accepts (`MediaAccept`): the media the
        * picker offers.
        */
      accepted: String,
      /** A slot that takes several: tiles are ticked, then used together. */
      multiple: Boolean,
      /** The project to open on, in a workspace. */
      projectId: Option[String],
      onPicked: List[Picked] => Unit
  ) {
    private val images = accepted.contains("image/")
    private val videos = accepted.contains("video/")

    def takes(output: GenerationOutput): Boolean =
      (images && output.mimeType.startsWith("image/")) ||
        (videos && output.mimeType.startsWith("video/"))

    def noun: String =
      if (images && videos) "image or video"
      else if (videos) "video"
      else "image"

    def title: String =
      if (multiple) noun.replace("image", "images").replace("video", "videos")
      else if (images) s"an $noun"
      else s"a $noun"
  }

  /** One output of a gallery entry. */
  case class Picked(
      generation: Generation,
      output: GenerationOutput,
      /** The entry's configuration by name, for what describes it. */
      configurationLabel: String
  ) {
    def key: (String, String) = (generation.id, output.fileName)
  }

  /** How a form asks for the picker: what the slot accepts, whether it takes
    * several, and what to do with what was picked.
    */
  type Open = (String, Boolean, List[Picked] => Unit) => Unit
}
