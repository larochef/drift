package drift.frontend.pages.assistant

import drift.frontend.components.{Component, RichText}
import drift.frontend.services.ApiClient
import drift.shared.renderAssistantMarkdown

import scala.collection.mutable
import scala.util.Success

import com.raquo.laminar.api.L.*

/** A model's text (`specs/20`): plain while it streams, then its Markdown as
  * the backend rendered and cleaned it. Until that answer comes, or if it never
  * does, the plain text stays.
  */
class ReplyText(text: String, streaming: Boolean) extends Component {

  private def plain: HtmlElement = pre(cls := "assistant-text", text)

  lazy val element: HtmlElement =
    if (streaming || text.isEmpty) plain
    else
      div(child <-- ReplyText.rendered(text).map {
        case Some(html) => RichText(html).element
        case None       => plain
      })
}

object ReplyText {

  /** Rendered replies by their text: a turn is rebuilt whenever its signal
    * emits, and a conversation is drawn again on every visit, so each text is
    * asked for once per page load.
    */
  private val cache = mutable.Map.empty[String, String]

  private val render = ApiClient.stream(renderAssistantMarkdown)

  private def rendered(text: String): Signal[Option[String]] =
    cache.get(text) match {
      case Some(html) => Val(Some(html))
      case None       =>
        render(text).recoverToTry
          .collect { case Success(html) =>
            cache(text) = html
            Some(html)
          }
          .toSignal(None)
    }
}
