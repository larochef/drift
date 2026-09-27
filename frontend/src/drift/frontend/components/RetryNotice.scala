package drift.frontend.components

import com.raquo.laminar.api.L.*

/** A load that failed, said in words, beside the button that asks again
  * (`specs/24`): for what a user can simply retry — a site overloaded, a
  * network blip — where the service's banner would leave them to find the
  * action again.
  */
class RetryNotice(message: String, onRetry: () => Unit) extends Component {
  lazy val element: HtmlElement = div(
    cls := "notification is-danger is-light mt-3 is-flex " +
      "is-align-items-center is-justify-content-space-between",
    span(message),
    button(
      cls := "button is-small ml-3",
      "Retry",
      onClick --> (_ => onRetry())
    )
  )
}
