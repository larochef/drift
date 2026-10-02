package drift.frontend.pages.gallery

import drift.frontend.components.{BrowserModal, Component}
import drift.frontend.services.*
import drift.frontend.services.HistoryService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The gallery as a place to pick inputs from
  * (`specs/50-inputs-from-the-gallery.md`): its tiles, newest day first, under
  * its filters, over the page that asked. One tile is picked with a click; a
  * slot that takes several ticks them, in the order they are to be used.
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

  /** The outputs ticked, in the order they were. */
  private val ticked = Var(List.empty[Picked])

  def open(asked: Request): Unit = {
    val project = asked.projectId.getOrElse("")
    projectFilter.set(project)
    configurationFilter.set("")
    searchVar.set("")
    showNsfw.set(projectsNow.now().exists(p => p.id == project && p.nsfw))
    ticked.set(List.empty)
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

  private def daySection(
      asked: Request,
      date: String,
      daySignal: Signal[HistoryDay]
  ): HtmlElement = {
    val loaded: Signal[Option[List[Generation]]] =
      historyService.generationsByDay.map(_.get(date)).distinct
    val loading: Signal[Boolean] =
      historyService.loadingDays.map(_.contains(date)).distinct
    // One tile per output the slot can take.
    val shown: Signal[List[Picked]] =
      loaded.combineWith(filter, labels).map { (loaded, matches, labels) =>
        loaded
          .getOrElse(List.empty)
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
    div(
      cls := "gallery-day",
      div(
        cls := "gallery-day-heading",
        h2(cls := "subtitle is-6 text-primary mb-0", date),
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
        }
      ),
      child <-- loaded.combineWith(shown).map {
        case (Some(_), Nil) =>
          p(
            cls := "text-secondary is-size-7",
            s"Nothing of this day to pick: no ${asked.noun} matches the " +
              "filters."
          )
        case _ => emptyNode
      },
      div(
        cls := "gallery-grid",
        children <-- shown
          .split(_.key) { (_, tile, _) =>
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
        child <-- historyService.daysLoaded
          .combineWith(historyService.days.map(_.isEmpty))
          .map {
            case (true, true) =>
              p(cls := "text-secondary", "The gallery is empty.")
            case _ => emptyNode
          },
        children <-- historyService.days.split(_.date)((date, _, daySignal) =>
          daySection(asked, date, daySignal)
        )
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
    // The newest day is what a picker is opened for: loaded without a click,
    // the older ones on demand as in the gallery.
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
        days.headOption
          .map(_.date)
          .filterNot(date => loaded.contains(date) || loading(date))
          .foreach(date => historyService.push(Command.LoadDay(date)))
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
