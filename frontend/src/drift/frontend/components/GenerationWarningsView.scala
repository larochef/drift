package drift.frontend.components

import com.raquo.laminar.api.L.*

/** What a generation went on without — a LoRA the server left unapplied, whole
  * or in part: said beside the picture, which was made but is not the one that
  * was asked for. Nothing when there is nothing to say.
  */
class GenerationWarningsView(warnings: List[String]) extends Component {

  lazy val element: HtmlElement = div(
    Option.when(warnings.nonEmpty)(
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-3",
        p(
          cls := "has-text-weight-bold mb-1",
          "⚠ Made without everything that was asked"
        ),
        warnings.map(warning => p(warning))
      )
    )
  )
}
