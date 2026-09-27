package drift.frontend.components

import com.raquo.laminar.api.L.*

/** The tab strip of an opened result (`specs/24`): Files · About · Images on
  * Civitai, Examples · Files · Model card on the repository sites. The chosen
  * tab is the browser's own `Var` -- it stays chosen for the next result
  * opened, so reading cards or galleries in a row costs no click each.
  */
class BrowserTabs[T](current: Var[T], tabs: (T, String)*) extends Component {
  lazy val element: HtmlElement = div(
    cls := "tabs mb-3",
    ul(
      tabs.map { (target, text) =>
        li(
          cls("is-active") <-- current.signal.map(_ == target),
          a(text, onClick --> (_ => current.set(target)))
        )
      }
    )
  )
}
