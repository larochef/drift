package drift.frontend.components

import scala.collection.mutable

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** What sits behind an overlay stops scrolling while the overlay is open
  * (François, 2026-09-11: "I keep scrolling another container than the one I'd
  * want"). Every scrollable ancestor of the overlay is locked — the page
  * itself, and any panel it opens from, such as the browser modal's body under
  * a Civitai viewer — and each gets its own overflow back on close. Overlays
  * nest, so locks are counted per element. Every modal and popin mounts it.
  */
object ScrollLock {

  /** An element held by `count` overlays, with its inline style before the
    * first.
    */
  private case class Held(count: Int, overflow: String, gutter: String)

  private val held = mutable.Map.empty[dom.html.Element, Held]

  def whileMounted: Modifier[HtmlElement] =
    onMountUnmountCallbackWithState[HtmlElement, List[dom.html.Element]](
      mount = context => lock(context.thisNode.ref),
      unmount = (_, locked) => locked.foreach(unlock)
    )

  /** The page's root, always — it scrolls without saying so in its style — and
    * every ancestor whose overflow scrolls.
    */
  private def scrollableAncestors(
      overlay: dom.Element
  ): List[dom.html.Element] = {
    val root = dom.document.documentElement
    Iterator
      .iterate[dom.Node](overlay.parentNode)(_.parentNode)
      .takeWhile(_ != null)
      .collect { case element: dom.html.Element => element }
      .filter(element =>
        element == root ||
          Set("auto", "scroll")
            .contains(dom.window.getComputedStyle(element).overflowY)
      )
      .toList
  }

  private def lock(overlay: dom.Element): List[dom.html.Element] = {
    val targets = scrollableAncestors(overlay)
    targets.foreach { element =>
      held.get(element) match {
        case Some(existing) =>
          held(element) = existing.copy(count = existing.count + 1)
        case None =>
          held(element) = Held(
            1,
            element.style.overflow,
            element.style.getPropertyValue("scrollbar-gutter")
          )
          // A scrollbar that vanishes shifts everything beside it: its room
          // stays reserved while the lock holds.
          if (element.scrollHeight > element.clientHeight)
            element.style.setProperty("scrollbar-gutter", "stable")
          element.style.overflow = "hidden"
      }
    }
    targets
  }

  private def unlock(targets: List[dom.html.Element]): Unit =
    targets.foreach { element =>
      held.get(element).foreach {
        case Held(1, overflow, gutter) =>
          held.remove(element)
          element.style.overflow = overflow
          if (gutter.isEmpty) element.style.removeProperty("scrollbar-gutter")
          else element.style.setProperty("scrollbar-gutter", gutter)
        case existing =>
          held(element) = existing.copy(count = existing.count - 1)
      }
    }
}
