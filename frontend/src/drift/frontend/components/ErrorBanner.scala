package drift.frontend.components

import drift.frontend.services.ServiceErrors

import com.raquo.laminar.api.L.*

/** Renders the last failed command of a service, so a save that did not go
  * through says so instead of leaving the user to guess.
  */
class ErrorBanner(service: ServiceErrors) extends Component {
  lazy val element: HtmlElement = div(
    child <-- service.lastError.map {
      case Some(message) =>
        div(
          cls := "notification is-danger is-light mb-4",
          button(cls := "delete", onClick --> (_ => service.clearError())),
          message
        )
      case None => emptyNode
    }
  )
}
