package drift.frontend.pages.generate

import drift.frontend.components.Component

import com.raquo.laminar.api.L.*

/** One folded parameter group (`specs/14`): the title always shows, its fields
  * only when opened, and a closed group that is actually doing something says
  * so beside its title — folding hires away must not hide why a run is slow.
  */
class CollapsibleSection(
    title: String,
    /** What the closed group is doing, or None when it changes nothing. */
    summary: Signal[Option[String]],
    content: => HtmlElement
) extends Component {

  private val open: Signal[Boolean] =
    CollapsibleSection.openSections.signal.map(_.contains(title)).distinct

  lazy val element: HtmlElement =
    div(
      cls := "box bg-table-header py-2 px-3 mb-3",
      div(
        cls := "is-flex is-align-items-baseline cursor-pointer",
        onClick --> (_ =>
          CollapsibleSection.openSections.update(open =>
            if (open.contains(title)) open - title else open + title
          )
        ),
        span(
          cls := "text-secondary is-size-7 mr-2",
          child.text <-- open.map(isOpen => if (isOpen) "▾" else "▸")
        ),
        label(
          cls := "label text-primary is-small mb-0 cursor-pointer",
          title
        ),
        child <-- open.combineWith(summary).map {
          case (false, Some(text)) =>
            span(cls := "text-secondary is-size-7 ml-2", text)
          case _ => emptyNode
        }
      ),
      child <-- open.map {
        case true  => div(cls := "mt-2", content)
        case false => emptyNode
      }
    )
}

object CollapsibleSection {

  /** Which folded parameter sections are open (`specs/14`). On the companion,
    * so it is one setting per user rather than per panel: the panel is rebuilt
    * whenever the model picker switches configuration, and a section the user
    * opened must not close because of that — nor because a version was applied
    * to the form (`specs/19`: a generation must not move the form).
    */
  private val openSections = Var(Set.empty[String])
}
