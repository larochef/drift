package drift.frontend.pages.gallery

import drift.frontend.components.Component

import com.raquo.laminar.api.L.*

/** One section of the detail view's right column. */
case class DetailSection(
    id: String,
    title: String,
    /** A count beside the title, where there is one worth seeing. */
    badge: Signal[Option[String]],
    body: HtmlElement
)

/** The right column of the detail view: tabs, one open at a time (François,
  * 2026-09-19). Parameters, the post-processing rows and the chain this image
  * belongs to each want the full height and only one is wanted at once;
  * stacked, they pushed the batch filmstrip and the actions off the bottom of
  * the window.
  *
  * Every body is built **once** and hidden rather than removed. The redraw
  * panel holds what has been typed into it — instructions, strength, the tile
  * and window sizes — and rebuilding it on every switch would throw that away.
  * This is why the tab strip is not `BrowserTabs`, which leaves the bodies to
  * its caller and so invites a `child <--` that rebuilds them.
  */
class DetailTabs(sections: List[DetailSection], open: Var[String])
    extends Component {

  private def tab(section: DetailSection): HtmlElement =
    li(
      cls("is-active") <-- open.signal.map(_ == section.id),
      a(
        onClick --> (_ => open.set(section.id)),
        span(section.title),
        child.maybe <-- section.badge.map(
          _.map(text => span(cls := "tag is-small ml-2", text))
        )
      )
    )

  lazy val element: HtmlElement = div(
    cls := "detail-tabs",
    // A section can be absent — Inputs, when the generation had none — and the
    // tab kept from the last image would then show nothing at all.
    onMountCallback(_ =>
      if (!sections.exists(_.id == open.now()))
        sections.headOption.foreach(first => open.set(first.id))
    ),
    div(cls := "tabs is-small is-boxed mb-2", ul(sections.map(tab))),
    sections.map(section =>
      div(
        cls := "detail-section-body",
        cls("is-hidden") <-- open.signal.map(_ != section.id),
        section.body
      )
    )
  )
}
