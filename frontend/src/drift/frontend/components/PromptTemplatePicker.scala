package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** A select over the templates of one kind (`specs/32-prompt-library.md`), used
  * where a prompt is about to be sent: the assistant panel and the redraw
  * panel. `none` labels an extra first choice for "no template", when a caller
  * allows one.
  */
class PromptTemplatePicker(
    templates: Signal[List[PromptTemplate]],
    choice: Var[Option[String]],
    none: Option[String] = None,
    small: Boolean = true
) extends Component {

  lazy val element: HtmlElement = div(
    cls := "select",
    cls("is-small") := small,
    select(
      children <-- templates.map { all =>
        none.toList.map(label =>
          option(
            value := "",
            selected <-- choice.signal.map(_.isEmpty),
            label
          )
        ) ++ all.map(t =>
          option(
            value := t.id,
            selected <-- choice.signal.map(_.contains(t.id)),
            t.label + (if (t.builtIn) " (built-in)" else "")
          )
        )
      },
      onChange.mapToValue.map(v => Option(v).filter(_.nonEmpty)) --> choice
    )
  )
}
