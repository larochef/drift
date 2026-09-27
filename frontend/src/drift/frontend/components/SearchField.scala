package drift.frontend.components

import com.raquo.laminar.api.L.*

/** The query input plus Search button shared by the model browsers. The input
  * is controlled (`value <--`), so a programmatic change to `query` is
  * reflected -- the HuggingFace browser used `defaultValue` and silently was
  * not.
  */
class SearchField(
    query: Var[String],
    busy: Signal[Boolean],
    onSearch: () => Unit,
    hint: String = "Search models..."
) extends Component {
  lazy val element: HtmlElement =
    div(
      cls := "field has-addons mb-3",
      div(
        cls := "control is-expanded",
        input(
          cls := "input",
          typ := "text",
          placeholder := hint,
          value <-- query.signal,
          onInput.mapToValue --> query,
          onKeyDown.filter(_.keyCode == 13).mapTo(()) --> (_ => onSearch())
        )
      ),
      div(
        cls := "control",
        button(
          cls := "button is-primary",
          "Search",
          disabled <-- busy,
          onClick --> (_ => onSearch())
        )
      )
    )
}
