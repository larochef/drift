package drift.frontend.pages.assistant

import drift.frontend.components.Component
import drift.frontend.services.AssistantService.Turn

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** A reply's reasoning: folded away once the reply has begun, and while the
  * model is still thinking, its last lines as they come — what it is busy with,
  * before a word of the answer. Its Markdown is rendered as it comes
  * (`ReplyText.shown`), the plain text until the first rendering.
  *
  * One element for the turn's whole life, fed by signals: built anew with each
  * chunk, as the rest of a streaming turn is, a block the user had opened
  * closed again at the next word.
  */
class ThinkingView(turn: Signal[Turn]) extends Component {

  private val reasoning = turn.map(_.reasoning).distinct

  /** Thinking still: streaming, and nothing of the reply yet. */
  private val thinking =
    turn
      .map(t => t.streaming && t.text.isEmpty && t.reasoning.nonEmpty)
      .distinct

  /** The reasoning as it reads now: rendered, or plain until it is. */
  private def text: Signal[HtmlElement] =
    ReplyText.shown(
      reasoning,
      turn.map(_.streaming),
      "assistant-text is-size-7",
      rendering => div(cls := "is-size-7", rendering)
    )

  /** Whether the user has the whole of it open, which the tail then repeats. */
  private val opened = Var(false)
  private val onToggle = eventProp[dom.Event]("toggle")

  lazy val element: HtmlElement = div(
    display <-- reasoning.map(text => if (text.isEmpty) "none" else ""),
    detailsTag(
      cls := "mb-2",
      onToggle --> (event =>
        opened.set(
          event.target.asInstanceOf[dom.Element].hasAttribute("open")
        )
      ),
      summaryTag(
        cls := "text-secondary is-size-7",
        child.text <-- thinking
          .combineWith(reasoning.map(_.length))
          .map { case (still, length) =>
            s"${if (still) "thinking…" else "thinking"} ($length characters)"
          }
      ),
      div(child <-- text)
    ),
    child <-- thinking.combineWith(opened.signal).map {
      case (true, false) =>
        div(
          cls := "assistant-thinking-tail text-secondary mb-2",
          div(child <-- text)
        )
      case _ => emptyNode
    }
  )
}
