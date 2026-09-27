package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.frontend.services.LogService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The session's output, followed live (`specs/13-log-streaming.md`). Opening
  * it replays the server's buffer first, so arriving late still shows the load
  * that already happened. Following can be paused, which is the only way to
  * read something that a chatty load would otherwise scroll away.
  */
class SessionLog(
    logService: LogService,
    /** Whether the body scrolls with new lines — the panel's, so a pause holds
      * across closing and reopening the log.
      */
    followLog: Var[Boolean],
    onClose: () => Unit
) extends Component {

  lazy val element: HtmlElement = div(
    cls := "box bg-card p-3 mb-3 session-log",
    div(
      cls := "level is-mobile mb-2",
      div(
        cls := "level-left",
        span(cls := "text-secondary is-size-7", "Session output"),
        span(
          cls := "tag is-small ml-2",
          child.text <-- logService.lines.map(lines => s"${lines.size} lines")
        )
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-small mr-2",
          child.text <-- followLog.signal.map(on =>
            if (on) "⏸ Pause" else "▶ Follow"
          ),
          onClick --> (_ => followLog.update(!_))
        ),
        button(
          cls := "button is-small mr-2",
          "Copy all",
          onClick.compose(_.sample(logService.lines)) --> Observer[
            List[LogLine]
          ](lines =>
            org.scalajs.dom.window.navigator.clipboard
              .writeText(lines.map(_.text).mkString("\n"))
          )
        ),
        button(
          cls := "button is-small",
          "Close",
          onClick --> (_ => onClose())
        )
      )
    ),
    pre(
      cls := "session-log-body is-size-7",
      // One text node rather than an element per line: a load prints thousands,
      // and a keyed list of them costs far more than the text it shows. The
      // rate is bounded too - re-rendering the body per captured line is work
      // nobody can read at that speed.
      child.text <-- logService.lines
        .map(_.map(_.text).mkString("\n"))
        .changes
        .throttle(400)
        .toSignal(""),
      // Autoscroll while following, and not once paused - the pause exists to
      // hold a line still.
      inContext(node =>
        logService.lines --> Observer[List[LogLine]](_ =>
          if (followLog.now())
            node.ref.scrollTop = node.ref.scrollHeight.toDouble
        )
      )
    )
  )
}
