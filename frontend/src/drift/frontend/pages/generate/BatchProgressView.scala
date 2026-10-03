package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** Where a batch has got to, above the bar of the image in flight
  * ([[drift.frontend.components.LogProgressView]]): "Image 2 of 4" over a bar
  * of the images already done, the way a tiled job counts its tiles over the
  * tile's own bar. Whole images only — an image can run several passes, so
  * folding its steps into the batch's bar could only be a guess.
  *
  * Nothing for a single image.
  */
class BatchProgressView(batch: Signal[Option[BatchProgress]])
    extends Component {

  // Not in Laminar's default bundle, so defined here.
  private val progressTag = htmlTag("progress")

  /** `<progress value>` - Laminar knows `value` only as an input property. */
  private val progressValueAttr = htmlAttr("value", StringAsIsCodec)

  lazy val element: HtmlElement = div(
    child <-- batch.distinct.map {
      case Some(position) if position.total > 1 =>
        div(
          cls := "mt-2",
          p(
            cls := "is-size-7 text-secondary",
            s"Image ${position.image} of ${position.total}"
          ),
          progressTag(
            cls := "progress is-small is-info mt-1 mb-0",
            progressValueAttr := position.completed.toString,
            maxAttr := position.total.toString
          )
        )
      case _ => emptyNode
    }
  )
}
