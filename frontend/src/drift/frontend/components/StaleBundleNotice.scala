package drift.frontend.components

import drift.frontend.services.StatusSocketService

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** Says so when the status socket sends what this page cannot read: the backend
  * has been restarted on a build whose messages are newer than the bundle the
  * browser holds. Every topic travels in one message, so nothing live arrives
  * at all in that state — and the page looks idle rather than broken, which is
  * how a finished runtime install can sit there saying "queued" (François,
  * 2026-09-20). Reloading is the whole cure, so the notice carries the button.
  */
class StaleBundleNotice(statusSocket: StatusSocketService) extends Component {
  lazy val element: HtmlElement = div(
    child <-- statusSocket.undecodable.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "notification is-warning is-light mb-4",
          span(
            "drift is running a build this page cannot read — nothing here " +
              "updates by itself until it is reloaded."
          ),
          button(
            cls := "button is-small ml-2",
            "Reload",
            onClick --> (_ => dom.window.location.reload())
          )
        )
    }
  )
}
