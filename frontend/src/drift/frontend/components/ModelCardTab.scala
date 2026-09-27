package drift.frontend.components

import com.raquo.laminar.api.L.*

/** What a repository's card tab has to show. */
enum ModelCardContent {

  /** Nothing has arrived yet. */
  case Pending

  /** The repository has no README. */
  case Absent

  /** The site shows the card only to accounts that accepted its terms. */
  case Gated

  /** The README, rendered and cleaned by the backend. */
  case Rendered(html: String)
}

/** The Model card tab of an opened repository (`specs/24`, `37`): the way to
  * the repository on its own site, and its README as the backend rendered and
  * cleaned it. A failure says why beside a Retry that re-sends what failed.
  */
class ModelCardTab(
    /** "HuggingFace", "ModelScope" -- and the host they answer on. */
    siteName: String,
    siteHost: String,
    pageUrl: String,
    card: Signal[ModelCardContent],
    loading: Signal[Boolean],
    failure: Signal[Option[String]],
    onRetry: () => Unit
) extends Component {
  lazy val element: HtmlElement = div(
    p(
      cls := "is-size-7 mb-3",
      a(
        href := pageUrl,
        target := "_blank",
        rel := "noopener noreferrer",
        s"Open on $siteHost ↗"
      )
    ),
    child <-- card.combineWith(loading, failure).map {
      case (_, true, _) =>
        p(cls := "text-secondary", "Loading the model card...")
      case (_, _, Some(reason)) =>
        RetryNotice(s"The model card did not load. $reason", onRetry).element
      case (ModelCardContent.Rendered(html), _, _) => RichText(html).element
      case (ModelCardContent.Gated, _, _)          =>
        p(
          cls := "text-secondary",
          s"This repository is gated: $siteName shows its card only to " +
            s"accounts that accepted its terms. Accept them on $siteHost, " +
            "with the account whose token is set in Settings, to read it here."
        )
      case (ModelCardContent.Absent, _, _) =>
        p(cls := "text-secondary", "This repository has no model card.")
      case (ModelCardContent.Pending, _, _) => emptyNode
    }
  )
}
