package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** Where a video's generation has got to, above the bar of the stage in flight
  * ([[drift.frontend.components.LogProgressView]]): "Step 2 of 5: sampling"
  * over a bar of the stages already done, as [[BatchProgressView]] counts a
  * batch's images. Whole stages only — they differ too much in length for a
  * fraction of one to mean a fraction of the whole.
  *
  * Nothing for an engine that announces no stages.
  */
class StageProgressView(stage: Signal[Option[StageProgress]])
    extends Component {

  // Not in Laminar's default bundle, so defined here.
  private val progressTag = htmlTag("progress")

  /** `<progress value>` - Laminar knows `value` only as an input property. */
  private val progressValueAttr = htmlAttr("value", StringAsIsCodec)

  lazy val element: HtmlElement = div(
    child <-- stage.distinct.map {
      case Some(position) if position.total > 1 =>
        div(
          cls := "mt-2",
          p(
            cls := "is-size-7 text-secondary",
            s"Step ${position.stage} of ${position.total}: ${position.name}"
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
