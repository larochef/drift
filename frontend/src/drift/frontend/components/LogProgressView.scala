package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What a run is doing, read out of its log (`specs/13-log-streaming.md`): a
  * session's, on the inference page, and a post-processing job's, in the
  * gallery — the same bar and the same last line, since both come from the same
  * sd-cpp output through the same parser.
  *
  * Whichever bar redrew most recently is the one shown, whatever its kind: a
  * session prints many bars, not one - several loads while it starts, a
  * sampling bar per pass, and more tensors loading lazily between passes.
  * Filtering to one kind would blank the bar for the others, and stitching them
  * into a single number could only be a guess, so each is shown with its own
  * label and totals.
  *
  * With no bar in flight there is no animated placeholder: Bulma's
  * indeterminate progress animates `background-position`, which repaints the
  * element every frame for as long as it is on screen (François, 2026-09-09:
  * the interface went laggy until the tab lost focus). The last log line says
  * more than a moving stripe anyway.
  */
class LogProgressView(
    /** The bar in flight, and the last line that was not one. */
    state: Signal[(Option[SessionProgress], Option[String])]
) extends Component {

  // Not in Laminar's default bundle, so defined here.
  private val progressTag = htmlTag("progress")

  /** `<progress value>` - Laminar knows `value` only as an input property. */
  private val progressValueAttr =
    htmlAttr("value", com.raquo.laminar.codecs.StringAsIsCodec)

  lazy val element: HtmlElement = div(
    child <-- state.distinct
      .map {
        case (Some(progress), _) =>
          div(
            cls := "mt-2",
            progressTag(
              cls := "progress is-small is-info mb-1",
              progressValueAttr := progress.done.toString,
              maxAttr := progress.total.toString
            ),
            p(
              cls := "is-size-7 text-secondary",
              // The pass, where there is more than one: wan 2.2 runs its
              // high-noise expert and then its low-noise one, and the two bars
              // are otherwise identical.
              s"${progress.kind.label}" +
                progress.note.map(note => s" ($note)").getOrElse("") +
                s" ${progress.done}/${progress.total}" +
                s" (${progress.percent}%) - ${progress.detail}"
            )
          )
        case (None, Some(activity)) =>
          p(cls := "is-size-7 text-secondary mt-2 text-break", activity)
        case _ => emptyNode
      }
  )
}
