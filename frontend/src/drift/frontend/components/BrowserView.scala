package drift.frontend.components

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** Which of a browser's two views shows -- the result list or one opened result
  * -- and where the list was scrolled while the opened one has the body, so
  * Back lands on the tile the browse left off at instead of the top of the
  * search (François, 2026-09-17).
  *
  * The state is its own object rather than `ResultBrowser`'s own, because the
  * cards are built by the site's browser and must be able to open one: `open()`
  * from a card's click, `back()` from the footer's button, and the shell
  * renders whichever view `showingDetail` names.
  */
class BrowserView {

  private val detail = Var(false)
  private var container: Option[dom.Element] = None
  private var listScrollTop: Double = 0

  /** The opened result's view is showing; otherwise, the result list. */
  val showingDetail: Signal[Boolean] = detail.signal

  /** `ResultBrowser` binds this to the modal's body: the element that scrolls.
    */
  val binder: Mod[HtmlElement] = onMountUnmountCallback[HtmlElement](
    mount = context => container = Some(context.thisNode.ref),
    unmount = _ => container = None
  )

  /** Remembers where the list stands and opens the detail view at its own top
    * -- one opened from halfway down a search otherwise opens halfway down.
    */
  def open(): Unit = {
    listScrollTop = container.map(_.scrollTop).getOrElse(0)
    container.foreach(_.scrollTop = 0)
    detail.set(true)
  }

  /** The list again, scrolled where it was left -- one frame later, since it
    * has no height to scroll within before it is re-rendered and painted.
    */
  def back(): Unit = {
    detail.set(false)
    val remembered = listScrollTop
    dom.window.requestAnimationFrame(_ =>
      container.foreach(_.scrollTop = remembered)
    )
  }
}
