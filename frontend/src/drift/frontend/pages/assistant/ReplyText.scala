package drift.frontend.pages.assistant

import drift.frontend.components.{Component, RichText}
import drift.frontend.services.ApiClient
import drift.shared.renderAssistantMarkdown

import scala.collection.mutable
import scala.util.Success

import com.raquo.laminar.api.L.*

/** A model's text (`specs/20`), its Markdown as the backend rendered and
  * cleaned it — as it streams too, rendered again as it grows
  * (`ReplyText.live`). Until the first answer comes, or if none does, the plain
  * text shows.
  *
  * One element for the turn's life, fed by signals: a rendering stays on screen
  * until the next one replaces it, whatever else of the turn is redrawn.
  */
class ReplyText(text: Signal[String], streaming: Signal[Boolean])
    extends Component {

  lazy val element: HtmlElement =
    div(child <-- ReplyText.shown(text.distinct, streaming))
}

object ReplyText {

  /** Rendered replies by their text: a turn is rebuilt whenever its signal
    * emits, and a conversation is drawn again on every visit, so each text is
    * asked for once per page load.
    */
  private val cache = mutable.Map.empty[String, String]

  private val render = ApiClient.stream(renderAssistantMarkdown)

  /** How often a text still growing is rendered again, in milliseconds. */
  private val LiveInterval = 700

  /** A text that may still be growing (`growing`), rendered as it grows: asked
    * again every `LiveInterval` while it changes, each answer replacing the
    * last, and once more when it has settled. `None` until the first answer.
    */
  def live(
      text: Signal[String],
      growing: Signal[Boolean]
  ): Signal[Option[String]] =
    growing.distinct
      .flatMapSwitch { still =>
        if (still)
          EventStream
            .periodic(LiveInterval)
            .sample(text)
            .distinct
            .filter(_.nonEmpty)
            .flatMapSwitch(fresh)
        else
          EventStream
            .fromValue(())
            .sample(text)
            .filter(_.nonEmpty)
            .flatMapSwitch(kept)
      }
      .map(Some(_))
      .toSignal(None)

  private def fresh(text: String): EventStream[String] =
    render(text).recoverToTry.collect { case Success(html) => html }

  /** A settled text's rendering, asked once per page load. */
  private def kept(text: String): EventStream[String] =
    cache.get(text) match {
      case Some(html) => EventStream.fromValue(html)
      case None       =>
        fresh(text).map { html =>
          cache(text) = html
          html
        }
    }

  /** `text` as it reads now: its last rendering, redrawn only when a new one
    * comes, or the plain text until the first. `plainClass` styles the plain
    * text, `wrap` holds a rendering.
    */
  def shown(
      text: Signal[String],
      growing: Signal[Boolean],
      plainClass: String = "assistant-text",
      wrap: HtmlElement => HtmlElement = identity
  ): Signal[HtmlElement] =
    live(text, growing).flatMapSwitch {
      case Some(html) => Val(wrap(RichText(html).element))
      case None       => text.map(value => pre(cls := plainClass, value))
    }
}
