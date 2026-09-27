package drift.frontend.pages.generate

import drift.frontend.components.Component

import com.raquo.laminar.api.L.*

/** What the last reuse seeded, and what it could not carry, above the form —
  * dismissable, and composed from the form state's two parts so a warning
  * raised late does not replace the headline.
  */
class ReuseNotice(state: GenerationFormState) extends Component {
  import state.{reuseNoticeText, reuseWarnings}

  lazy val element: HtmlElement = div(
    child <-- reuseNoticeText.signal.combineWith(reuseWarnings.signal).map {
      (text, warnings) =>
        if (text.isEmpty && warnings.isEmpty) emptyNode
        else
          div(
            cls := (if (warnings.isEmpty)
                      "notification is-info is-light py-2 px-3 is-size-7 mb-3 reuse-notice"
                    else
                      "notification is-warning is-light py-2 px-3 is-size-7 mb-3 reuse-notice"),
            button(
              cls := "delete is-small",
              onClick --> { _ =>
                reuseNoticeText.set(None)
                reuseWarnings.set(List.empty)
              }
            ),
            text.map(p(cls := "mb-0", _)).getOrElse(emptyNode),
            warnings.map(warning =>
              p(cls := "mb-0 has-text-weight-semibold", s"⚠ $warning")
            )
          )
    }
  )
}
