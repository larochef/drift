package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The chain: the original a generation was made from, and what was made from
  * it — both resolved in the loaded generations, so they follow the history.
  */
class GenerationLineage(
    generation: Generation,
    parent: Signal[Option[Generation]],
    derivatives: Signal[List[Generation]],
    onOpen: String => Unit
) extends Component {

  lazy val element: HtmlElement = div(
    child <-- parent.combineWith(derivatives).map { (parentNow, derived) =>
      if (generation.derivation.isEmpty && derived.isEmpty) emptyNode
      else
        div(
          cls := "gallery-lineage mt-3",
          generation.derivation.map { d =>
            div(
              cls := "mb-2",
              span(cls := "text-secondary is-size-7 mr-2", "Made from"),
              parentNow match {
                case Some(p) =>
                  button(
                    cls := "button is-small",
                    s"Open original (${RecordedParameters.titleOf(p).take(40)})",
                    onClick --> (_ => onOpen(p.id))
                  )
                case None =>
                  span(
                    cls := "text-primary is-size-7",
                    s"${d.parentId} — not in the loaded days, or deleted"
                  )
              }
            )
          },
          if (derived.isEmpty) emptyNode
          else
            div(
              span(cls := "text-secondary is-size-7 mr-2", "Made from this"),
              div(
                cls := "buttons",
                derived.map(entry =>
                  button(
                    cls := "button is-small",
                    entry.derivation
                      .map(RecordedParameters.operationOf)
                      .getOrElse(entry.id),
                    onClick --> (_ => onOpen(entry.id))
                  )
                )
              )
            )
        )
    }
  )
}
