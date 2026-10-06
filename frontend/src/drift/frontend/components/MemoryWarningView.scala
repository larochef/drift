package drift.frontend.components

import com.raquo.laminar.api.L.*

/** What a session says when its model loaded with almost no memory left
  * (`specs/07-launch-and-supervision.md`): the reason a generation crawls,
  * where the user is looking when it does.
  */
class MemoryWarningView(warning: Signal[Option[String]]) extends Component {

  lazy val element: HtmlElement = div(
    child <-- warning.distinct.map {
      case Some(message) =>
        div(
          cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-3",
          message
        )
      case None => emptyNode
    }
  )
}
