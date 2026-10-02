package drift.frontend.pages.assistant

import drift.frontend.components.Component
import drift.frontend.services.AssistantService

import com.raquo.laminar.api.L.*

/** The live assistant's controls in a model bar, as the image model has its own
  * there: the chat drawer's toggle, Stop, and what the server applied — out of
  * the chat, which keeps the room (François, 2026-09-29). A conversation that
  * is the page — a text project's, the Sandbox's Text tab — has no drawer.
  */
class AssistantSessionControls(
    service: AssistantService,
    onStop: () => Unit,
    /** Whether the chat drawer is open, where there is one. */
    drawerOpen: Option[Var[Boolean]]
) extends Component {

  lazy val element: HtmlElement = div(
    cls := "is-flex is-align-items-center",
    styleAttr := "gap: 0.5rem;",
    drawerOpen.map(open =>
      button(
        cls := "button is-small",
        cls("is-active") <-- open.signal,
        child.text <-- open.signal
          .map(shown => if (shown) "💬 Hide chat" else "💬 Show chat"),
        onClick --> (_ => open.update(!_))
      )
    ),
    button(
      cls := "button is-small is-warning",
      "⏹ Stop session",
      onClick --> (_ => onStop())
    ),
    AssistantSessionFacts(service).element
  )
}
