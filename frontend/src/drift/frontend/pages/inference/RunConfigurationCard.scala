package drift.frontend.pages.inference

import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

class RunConfigurationCard(
    rm: RunConfiguration,
    archName: String,
    blockers: List[LaunchBlocker],
    commandLine: Option[String],
    /** What resolving the parameters had to say for the runtime this card would
      * launch on (`specs/16-parameter-resolution.md`) — a dropped flag is never
      * left unexplained.
      */
    notes: List[ResolutionNote],
    session: Option[Session],
    /** Launchable runtimes, default first — the launch-time choice. */
    runtimes: List[Runtime],
    /** The runtime to preselect: the user's earlier pick, else the default. */
    selectedRuntimeId: Option[String],
    onSelectRuntime: String => Unit,
    onDelete: String => Unit,
    onEdit: RunConfiguration => Unit,
    /** What it lacks before it can launch — a runtime, weights — with what
      * supplies it (`specs/46`): the launch control offers that instead.
      */
    missing: Signal[Option[LaunchPrerequisites.Missing]],
    prerequisites: LaunchPrerequisites,
    onLaunch: String => Unit,
    onStop: String => Unit,
    /** Where the live session runs — the Sandbox or its project's name
      * (`specs/47-sandbox.md`).
      */
    runningIn: Option[String],
    /** Opens the page the live session runs in. */
    onOpen: Session => Unit
) extends Component {
  private val showCommand = Var(false)
  private val ready = blockers.isEmpty
  private val live = session.exists(_.status.isActive)
  private val onlyWeightsMissing = blockers.nonEmpty && blockers.forall {
    case LaunchBlocker.WeightsNotCached(_, _, _) => true
    case _                                       => false
  }

  private def statusTag(s: Session): HtmlElement = {
    val (colour, label) = s.status match {
      case SessionStatus.Starting => ("is-info", "starting…")
      case SessionStatus.Ready    =>
        ("is-success", s.port.map(p => s"ready on :$p").getOrElse("ready"))
      case SessionStatus.Failed => ("is-danger", "failed")
      // No colour: the neutral chip is right for a session that simply ended
      // (and is-dark is 11% lightness — invisible on this dark design).
      case SessionStatus.Stopped => ("", "stopped")
    }
    span(cls := s"tag is-small mr-2 $colour", label)
  }

  /** The runtime to launch on, rendered only beside the Launch button. The
    * element is rebuilt whenever the page re-renders, so the current pick is
    * restored from `selectedRuntimeId` rather than trusted to the DOM.
    */
  private def runtimeSelect: Option[HtmlElement] =
    Option.when(runtimes.nonEmpty)(
      select(
        cls := "select is-small",
        title := "Runtime to launch on",
        runtimes.map(runtime =>
          option(
            value := runtime.id,
            defaultSelected := selectedRuntimeId.contains(runtime.id),
            runtime.label
          )
        ),
        onChange.mapToValue --> (id => onSelectRuntime(id))
      )
    )

  private def launchOrStopControls: Seq[HtmlElement] =
    if (live)
      session.toSeq.flatMap(s =>
        Seq(
          span(
            cls := "text-secondary is-size-7 mr-2",
            s"Running in ${runningIn.getOrElse("the Sandbox")}"
          ),
          button(
            cls := "button is-link is-small mr-2",
            "Open",
            onClick --> (_ => onOpen(s))
          ),
          button(
            cls := "button is-warning is-small",
            "⏹ Stop",
            onClick --> (_ => onStop(s.id))
          )
        )
      )
    // A missing runtime or missing weights take the launch's place
    // (`specs/46-starter-configurations.md`); only a start or an end of an
    // install or a download rebuilds it, never a progress tick.
    else if (ready || onlyWeightsMissing)
      Seq(
        LaunchOrDownload(
          missing,
          prerequisites,
          div(
            cls := "run-configuration-launch-controls",
            runtimeSelect,
            button(
              cls := "button is-primary is-small",
              "▶ Launch",
              onClick --> (_ => onLaunch(rm.id))
            )
          )
        ).element
      )
    else Seq.empty

  /** Launching lives in its own footer strip, apart from the header's
    * edit/delete: it is the everyday action, they are the occasional ones.
    * Absent entirely while the configuration cannot launch.
    */
  private def launchFooter: Option[HtmlElement] = {
    val controls = launchOrStopControls
    Option.when(controls.nonEmpty)(
      div(cls := "card-footer run-configuration-launch", controls)
    )
  }

  lazy val element: HtmlElement = div(
    // No bottom margin: the inference page's grid gap spaces the cards.
    cls := "card bg-card run-configuration-card",
    div(
      cls := "card-header",
      paddingRight := "0.75rem",
      p(cls := "card-header-title text-primary", rm.label),
      span(
        cls := "tag is-small mr-2",
        cls := (if (ready) "is-success" else "is-warning"),
        if (ready) "ready" else "incomplete"
      ),
      session.map(statusTag),
      button(
        cls := "button is-info is-small card-header-action",
        "\u270F\uFE0F Edit",
        onClick --> (_ => onEdit(rm))
      ),
      button(
        cls := "button is-danger is-small card-header-action",
        "\uD83D\uDDD1\uFE0F Delete",
        onClick --> (_ =>
          if (window.confirm("Delete this configuration?")) onDelete(rm.id)
        )
      )
    ),
    div(
      cls := "card-content",
      p(cls := "text-secondary mb-2", s"ID: ${rm.id}"),
      p(
        cls := "text-secondary mb-2",
        s"Architecture: $archName · runs on ${rm.runner.displayName}"
      ),
      session
        .filter(_.status == SessionStatus.Failed)
        .flatMap(_.error)
        .map(reason =>
          div(
            cls := "notification is-danger is-light py-2 px-3 mb-3",
            p(
              cls := "has-text-weight-bold is-size-7 mb-1",
              "Launch failed:"
            ),
            // The reason may carry the log tail, so keep the line breaks.
            pre(
              cls := "is-size-7 text-break",
              styleAttr := "white-space: pre-wrap; word-break: break-all; background: transparent; padding: 0;",
              reason
            )
          )
        ),
      if (ready) emptyNode
      else
        div(
          cls := "notification is-warning is-light py-2 px-3 mb-3",
          p(cls := "has-text-weight-bold is-size-7 mb-1", "Cannot launch yet:"),
          ul(
            cls := "is-size-7",
            blockers.map(b => li(b.message))
          )
        ),
      // Above the command line, because they explain what is missing from it.
      // A live session shows what its own launch resolved instead: the runtime
      // may have changed since.
      (session.map(_.parameterNotes).filter(_.nonEmpty) match {
        case Some(live) => live
        case None       => notes.map(_.message)
      }) match {
        case Nil      => emptyNode
        case messages =>
          div(
            cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-3",
            p(
              cls := "has-text-weight-bold mb-1",
              // Not "this runtime changes the command line" any more: a note
              // can also be a flag the user removed
              // (`specs/16-parameter-resolution.md`).
              "How this command line was resolved"
            ),
            ul(messages.map(message => li(message)))
          )
      },
      commandLine match {
        case None       => emptyNode
        case Some(line) =>
          div(
            cls := "mb-3",
            button(
              cls := "button is-small",
              child.text <-- showCommand.signal.map(open =>
                if (open) "\u25BC Command line" else "\u25B6 Command line"
              ),
              onClick --> (_ => showCommand.update(!_))
            ),
            child <-- showCommand.signal.map {
              case false => emptyNode
              case true  =>
                div(
                  cls := "mt-2",
                  pre(
                    cls := "is-size-7 text-break",
                    styleAttr := "white-space: pre-wrap; word-break: break-all;",
                    line
                  ),
                  button(
                    cls := "button is-small is-info mt-1",
                    "Copy",
                    onClick --> (_ =>
                      org.scalajs.dom.window.navigator.clipboard
                        .writeText(line)
                    )
                  )
                )
            }
          )
      },
      if (rm.assignments.nonEmpty) {
        table(
          cls := "table is-fullwidth is-narrow bg-table-header text-primary",
          thead(
            tr(
              th(cls := "text-secondary", "Checkpoint"),
              th(cls := "text-secondary", "Assigned Model")
            )
          ),
          tbody(rm.assignments.toList.sortBy(_._1).map {
            (checkpoint, modelId) =>
              tr(
                td(cls := "text-primary", checkpoint),
                td(cls := "text-primary", modelId)
              )
          })
        )
      } else emptyNode,
      if (rm.overriddenParameters.nonEmpty || rm.removedParameters.nonEmpty) {
        div(
          cls := "mt-3",
          p(
            cls := "has-text-weight-bold text-secondary",
            "Override parameters"
          ),
          ul(
            rm.overriddenParameters.toList.sortBy(_._1).map { case (k, v) =>
              li(code(s"$k $v"))
            } ++ rm.removedParameters.sorted.map(flag =>
              li(
                code(flag),
                span(cls := "tag is-warning is-light ml-2", "not passed")
              )
            )
          )
        )
      } else emptyNode
    ),
    launchFooter
  )
}
