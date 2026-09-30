package drift.frontend.components

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** Ends a `<video>`'s download when the element leaves the page. A browser
  * keeps a removed player's range request open, half read, until the element is
  * garbage-collected; drift serves HTTP/1.1, so a few of those hold its six
  * connections and the next page's videos wait (tens of seconds) for them.
  * Removing the source and reloading drops the download at once (bug 37). The
  * source comes back if the same element is mounted again.
  */
object VideoRelease {

  private val Released = "data-released-src"

  val onUnmount: Modifier[HtmlElement] = onMountUnmountCallback(
    mount = context =>
      context.thisNode.ref match {
        case video: dom.HTMLVideoElement
            if !video.hasAttribute("src") && video.hasAttribute(Released) =>
          video.setAttribute("src", video.getAttribute(Released))
          video.removeAttribute(Released)
        case _ => ()
      },
    unmount = element =>
      element.ref match {
        case video: dom.HTMLVideoElement if video.hasAttribute("src") =>
          video.pause()
          video.setAttribute(Released, video.getAttribute("src"))
          video.removeAttribute("src")
          video.load()
        case _ => ()
      }
  )
}
