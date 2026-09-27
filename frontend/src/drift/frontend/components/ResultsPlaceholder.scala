package drift.frontend.components

import com.raquo.laminar.api.L.*

/** The "loading..." bar and the "nothing here" line that every browser list
  * shows around its results.
  *
  * `show` gates the empty message: a search view only says "No models found"
  * once a search has actually run, otherwise an untouched browser accuses
  * itself of having found nothing.
  */
class ResultsPlaceholder(
    busy: Signal[Boolean],
    isEmpty: Signal[Boolean],
    busyText: String,
    emptyText: String,
    show: Signal[Boolean] = Val(true)
) extends Component {
  lazy val element: HtmlElement = div(
    child <-- busy.map { b =>
      if (b) progressTag(cls := "mb-3 is-primary", busyText) else emptyNode
    },
    child <-- Signal.combine(busy, isEmpty, show).map { (b, empty, visible) =>
      if (!b && empty && visible) p(cls := "text-secondary", emptyText)
      else emptyNode
    }
  )
}
