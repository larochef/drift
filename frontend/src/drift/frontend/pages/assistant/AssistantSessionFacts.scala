package drift.frontend.pages.assistant

import drift.frontend.components.Component
import drift.frontend.services.AssistantService

import com.raquo.laminar.api.L.*

/** What the assistant's server applied: its context size and whether it sees
  * images. The chat's own header shows it on the Assistant page; a workspace
  * shows it in its model bar, beside the model it describes.
  */
class AssistantSessionFacts(service: AssistantService) extends Component {

  lazy val element: HtmlElement = span(
    cls := "text-secondary is-size-7",
    child <-- service.properties
      .combineWith(service.propertiesError)
      .map {
        case (Some(properties), _) =>
          span(
            s"context ${properties.contextSize} tokens · ",
            if (properties.vision)
              span(cls := "tag is-link is-small", "vision")
            else span(cls := "tag is-small", "text only")
          )
        case (None, Some(error)) =>
          span(cls := "has-text-danger", error)
        case _ => span("reading what the server applied…")
      }
  )
}
