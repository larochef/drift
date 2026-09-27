package drift.frontend.components

import drift.shared.ArchitectureTags

import com.raquo.laminar.api.L.*

/** The All / <tag> chips shared by the architectures and the run configurations
  * built on them (François, 2026-09-10).
  *
  * Built from the tags actually in use rather than a fixed list, so a tag
  * invented on an architecture becomes a filter without anything else changing.
  */
class TagFilter(
    selected: Var[Option[String]],
    tags: Signal[List[String]]
) extends Component {

  private def chip(tag: Option[String], label: String): HtmlElement =
    button(
      cls <-- selected.signal.map(current =>
        if (current == tag) "button is-small is-primary is-selected"
        else "button is-small"
      ),
      label,
      onClick --> (_ => selected.set(tag))
    )

  lazy val element: HtmlElement = div(
    cls := "buttons has-addons kind-filter",
    // A tag that disappears - the last architecture carrying it was deleted or
    // retagged - must not leave the list filtered by something invisible.
    tags --> Observer[List[String]](inUse =>
      if (selected.now().exists(tag => !inUse.contains(tag)))
        selected.set(None)
    ),
    children <-- tags.map(inUse =>
      chip(None, "All") :: inUse.map(tag => chip(Some(tag), tag))
    )
  )
}

object TagFilter {

  /** The tags in use across some architectures, in the order the filters show
    * them.
    */
  def inUse(architectures: List[drift.shared.Architecture]): List[String] =
    ArchitectureTags.ordered(architectures.flatMap(_.tags).toSet)
}
