package drift.frontend.pages.gallery

import drift.frontend.components.Component

import com.raquo.laminar.api.L.*

/** One line per live session — each holds a model in memory beside any
  * post-processing job — with a button to stop it; never stopped on its own.
  */
class LiveSessionNotice(
    /** Live sessions, id → label. */
    liveSessions: Signal[List[(String, String)]],
    onStopSession: String => Unit
) extends Component {

  lazy val element: HtmlElement = div(
    children <-- liveSessions.map(_.map { (id, label) =>
      p(
        cls := "text-secondary is-size-7 mb-1",
        s"$label is loaded and holds its memory beside these jobs. ",
        button(
          cls := "button is-small is-light",
          "■ Stop it",
          onClick --> (_ => onStopSession(id))
        )
      )
    })
  )
}
