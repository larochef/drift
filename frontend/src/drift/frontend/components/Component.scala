package drift.frontend.components

import com.raquo.laminar.api.L.*
import com.raquo.laminar.modifiers.RenderableNode

trait Component {

  /** A `def`, so object pages can build a fresh element per render — frontroute
    * refuses to re-initialize location state on a re-inserted element, which is
    * exactly what a cached `lazy val` element causes on the second visit. Class
    * pages, constructed fresh per navigation, override with `lazy val`.
    */
  def element: HtmlElement
}

object Component {
  given RenderableNode[Component] = RenderableNode(c => c.element)
}
