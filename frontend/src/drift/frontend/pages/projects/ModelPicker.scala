package drift.frontend.pages.projects

import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.pages.inference.NewRunConfigurationModal
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** A tool's model picker: its configurations, the live one marked, picking
  * another stopping the live session and launching that one — or, when its
  * weights are not on disk yet, downloading them and leaving the live one
  * running (`specs/46-starter-configurations.md`). Rebuilt only when the live
  * session, the configurations or a download's start or end change, not with
  * progress, which would reset an open select. The workspace's model bar and
  * the Sandbox (`specs/47-sandbox.md`) both use it.
  */
class ModelPicker(
    tool: RuntimeTool,
    heading: Signal[String],
    /** The kind to make, which narrows the list to the models making it. */
    kind: Signal[Option[ProjectKind]],
    /** The page's one action while nothing is loaded: a larger select. */
    prominent: Boolean,
    trailing: Signal[Node],
    sessionService: SessionService,
    sessions: WorkspaceSessions,
    prerequisites: LaunchPrerequisites,
    /** The modal creating a configuration of a tool for a kind, and what to do
      * with the one created — the picker's **+ New**.
      */
    newConfiguration: (
        RuntimeTool,
        Option[ProjectKind],
        RunConfiguration => Unit,
        () => Unit
    ) => HtmlElement,
    /** The project launching, recorded on the session; none from the Sandbox.
      */
    projectId: Option[String],
    /** Whether the live session, named by its id, may be stopped for the one
      * picked — the Sandbox asks before stopping a project's model.
      */
    confirmReplace: String => Boolean
) extends Component {
  // A model picked before it could launch: it stays selected while what it
  // lacks downloads, and launches once nothing is missing (François,
  // 2026-09-28). Picking again, or the empty entry, replaces it.
  private val picked = Var(Option.empty[String])

  /** A configuration created from **+ New**, picked once the cache has a word
    * on each of its models — a model registered with it has none yet, and no
    * word reads as nothing missing.
    */
  private val created = Var(Option.empty[String])
  private val creating = Var(false)

  /** Bumped when a switch is declined, so the select shows the live model again
    * rather than the one the user let go of.
    */
  private val redraw = Var(0)

  private def launch(id: String, live: Option[(String, String)]): Unit = {
    if (live.forall(l => confirmReplace(l._1))) {
      live.foreach((sessionId, _) =>
        sessionService.push(SessionService.Command.Stop(sessionId))
      )
      sessionService.push(SessionService.Command.Launch(id, None, projectId))
    } else redraw.update(_ + 1)
  }

  /** What picking `id` in the select does: launch it, or — lacking weights or a
    * runtime — keep it picked while they come.
    */
  private def choose(
      id: String,
      live: Option[(String, String)],
      missing: Map[String, LaunchPrerequisites.Missing]
  ): Unit =
    if (live.exists(_._2 == id)) picked.set(None)
    else
      missing.get(id) match {
        // Kept until it can launch; the runtime is chosen in the notice
        // beside the picker.
        case Some(lacks) =>
          picked.set(Some(id))
          lacks.weights.foreach(w => prerequisites.download(w.idle))
        case None =>
          picked.set(None)
          launch(id, live)
      }

  lazy val element: HtmlElement = div(
    cls := "field is-grouped is-align-items-center mb-0",
    picked.signal
      .combineWith(prerequisites.byConfiguration, sessions.liveKey(tool))
      --> Observer[
        (
            Option[String],
            Map[String, LaunchPrerequisites.Missing],
            Option[(String, String)]
        )
      ] { (pick, missing, live) =>
        pick.filterNot(missing.contains).foreach { id =>
          picked.set(None)
          if (!live.exists(_._2 == id)) launch(id, live)
        }
      },
    created.signal
      .combineWith(
        prerequisites.unsettled,
        prerequisites.byConfiguration,
        sessions.liveKey(tool)
      )
      --> Observer[
        (
            Option[String],
            Set[String],
            Map[String, LaunchPrerequisites.Missing],
            Option[(String, String)]
        )
      ] { (fresh, unsettled, missing, live) =>
        fresh.filterNot(unsettled.contains).foreach { id =>
          created.set(None)
          choose(id, live, missing)
        }
      },
    span(
      cls := "workspace-picker-label text-primary has-text-weight-semibold mr-2",
      child.text <-- heading
    ),
    child <-- sessions
      .liveKey(tool)
      .combineWith(
        sessions.configurationsOf(tool, kind),
        kind,
        prerequisites.byConfiguration,
        picked.signal,
        redraw.signal
      )
      .map { (live, listed, kind, missing, pick, _) =>
        val liveConfiguration = live.map(_._2)
        val shown = pick.orElse(liveConfiguration)
        // Sessions are not the project's: a live model of the other kind
        // stays listed, marked, so the select names what actually runs.
        val configurations =
          listed.filter((c, fits) => fits || liveConfiguration.contains(c.id))
        div(
          cls := "control",
          select(
            cls := (if (prominent) "select is-medium" else "select"),
            onChange.mapToValue --> Observer[String] { id =>
              if (id.isEmpty) picked.set(None)
              else choose(id, live, missing)
            },
            option(
              value := "",
              selected := shown.isEmpty,
              if (configurations.isEmpty)
                kind
                  .fold("no configuration")(k => s"no ${k.noun} configuration")
              else "choose a model…"
            ),
            configurations.map((c, fits) =>
              option(
                value := c.id,
                selected := shown.contains(c.id),
                c.label + (
                  // the upstream engine goes without saying (`specs/43`)
                  if (c.runner == RuntimeEngine.upstream(tool)) ""
                  else s" · ${c.runner.displayName}"
                ) + (
                  missing
                    .get(c.id)
                    .fold("")(lacks =>
                      s" · ⬇ ${LaunchPrerequisites.describe(lacks)}"
                    )
                ) + (
                  if (!liveConfiguration.contains(c.id)) ""
                  else if (fits) " (live)"
                  else
                    kind
                      .fold(" (live)")(k => s" (live, not a ${k.noun} model)")
                )
              )
            )
          )
        )
      },
    // No runtime to run this tool's models: said once beside the picker, with
    // the install (`specs/46`).
    child <-- sessions
      .configurationsOf(tool, kind)
      .combineWith(prerequisites.byConfiguration)
      .map((listed, missing) =>
        // Said once for the whole picker: every build its configurations can
        // take, and a pick on another engine switches all that run on it.
        // Only the ones it offers — a video model is not an image project's
        // concern.
        LaunchPrerequisites.RuntimeNeed
          .merge(
            listed
              .collect { case (c, true) => c }
              .flatMap(c => missing.get(c.id).flatMap(_.runtime))
          )
          .fold(emptyNode)(need =>
            div(
              cls := "control ml-2",
              LaunchOrDownload.runtimeNotice(need, prerequisites)
            )
          )
      ),
    div(
      cls := "control",
      button(
        // The size of the session's controls beside it, not a size above.
        cls := (if (prominent) "button is-medium is-info"
                else "button is-small is-info"),
        span(cls := "plus-icon", "+"),
        " New",
        title <-- heading.map(name =>
          s"Create a configuration for the ${name.toLowerCase} and pick it"
        ),
        onClick --> (_ => creating.set(true))
      )
    ),
    child <-- creating.signal.combineWith(kind).map {
      case (false, _)   => emptyNode
      case (true, kind) =>
        newConfiguration(
          tool,
          kind,
          configuration => {
            prerequisites.refreshCache()
            created.set(Some(configuration.id))
          },
          () => creating.set(false)
        )
    },
    child <-- sessions.liveStatus(tool).map {
      case Some(status) =>
        span(
          cls := (status match {
            case SessionStatus.Ready => "tag is-success is-small ml-2"
            case _                   => "tag is-info is-small ml-2"
          }),
          status.toString.toLowerCase
        )
      case None => emptyNode
    },
    div(cls := "control", child <-- trailing)
  )
}

object ModelPicker {

  /** A picker's **+ New** (François, 2026-09-28): the run configurations page's
    * own modal, offering the architectures of the picker's tool that make the
    * kind asked for.
    */
  def newConfiguration(
      runConfigurationService: RunConfigurationService,
      loraService: LoraService,
      assistantService: AssistantService,
      runtimeService: RuntimeService,
      browsers: BrowserServices
  )(
      tool: RuntimeTool,
      kind: Option[ProjectKind],
      onCreated: RunConfiguration => Unit,
      onClose: () => Unit
  ): HtmlElement =
    NewRunConfigurationModal(
      runConfigurationService,
      runConfigurationService.architectures.map(
        _.filter(a => a.tool == tool && kind.forall(_.accepts(a)))
      ),
      loraService,
      browsers,
      assistantService.library.ofKind(PromptKind.AssistantSystem),
      runtimeService.runtimes,
      onClose = onClose,
      onCreated = onCreated,
      heading = tool match {
        case RuntimeTool.LlamaCpp => "New chat configuration"
        case _                    =>
          kind.fold("New run configuration")(k =>
            s"New ${k.noun} configuration"
          )
      }
    ).element
}
