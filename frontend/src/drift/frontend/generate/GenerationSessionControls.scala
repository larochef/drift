package drift.frontend.generate

import drift.frontend.components.Component
import drift.frontend.services.LogService

import com.raquo.laminar.api.L.*

/** The session's Log, Restart and Stop buttons, which the page places in its
  * model bar, beside the picker that names the model — a workspace and the
  * Sandbox alike.
  */
final class GenerationSessionControls(
    sessionId: String,
    /** The session's own output (`specs/13-log-streaming.md`) - what the model
      * is doing while it loads, and why it stopped when it did.
      */
    logService: LogService,
    /** Stops the session and launches the configuration again. */
    onRestart: () => Unit,
    onStop: () => Unit
) extends Component {

  private val showLog = Var(false)
  private val followLog = Var(true)

  lazy val element: HtmlElement = div(
    cls := "control is-flex",
    button(
      cls := "button mr-2 is-small",
      child.text <-- showLog.signal.map(open => if (open) "▼ Log" else "▶ Log"),
      onClick --> { _ =>
        val opening = !showLog.now()
        showLog.set(opening)
        if (opening) logService.follow(sessionId) else logService.stop()
      }
    ),
    button(
      cls := "button mr-2 is-small",
      "↻ Restart",
      title := "Stop the model and start it again on the same runtime, " +
        "keeping this form: it picks up LoRAs installed since it " +
        "started, and starts afresh if it misbehaves",
      onClick --> (_ => onRestart())
    ),
    button(
      cls := "button is-warning is-small",
      "⏹ Stop session",
      onClick --> (_ => onStop())
    )
  )

  /** The log the first button opens, for the panel to place over its form. */
  lazy val log: Modifier[HtmlElement] = child <-- showLog.signal.map {
    case true =>
      SessionLog(logService, followLog, () => showLog.set(false)).element
    case false => emptyNode
  }
}
