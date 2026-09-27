package drift.frontend.pages.cache

import drift.frontend.components.Component
import drift.frontend.services.ConversionService
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** The conversions of this drift run, on the Model Cache page: queued and
  * running ones with their tensor bar and a Cancel, failed ones with the tail
  * of sd-cli's output, finished ones naming the registered model.
  */
class ConversionJobList(conversionService: ConversionService)
    extends Component {
  import ConversionJobList.*

  private def outputName(job: ConversionJob): String =
    job.outputPath.split('/').last

  /** A bar makes sense while tensors are counted: dequantizing or converting.
    */
  private def counting(job: ConversionJob): Boolean =
    job.state == ConversionState.Converting ||
      job.state == ConversionState.Dequantizing

  private def row(job: ConversionJob): HtmlElement = job.state match {
    case ConversionState.Queued | ConversionState.Dequantizing |
        ConversionState.Converting | ConversionState.Registering =>
      div(
        cls := "notification is-info is-light py-2 px-3 is-size-7 mb-1",
        div(
          cls := "level is-mobile mb-1",
          div(
            cls := "level-left",
            span(
              s"Converting ${job.sourceLabel} → ${outputName(job)} (${job.targetType})" +
                ((job.state, job.progress) match {
                  case (ConversionState.Dequantizing, Some(progress)) =>
                    s" — dequantizing, tensor ${progress.completed} of ${progress.total}"
                  case (ConversionState.Converting, Some(progress)) =>
                    s" — tensor ${progress.completed} of ${progress.total}" +
                      (if (job.detail.nonEmpty) s", ${job.detail}" else "")
                  case _ =>
                    if (job.detail.nonEmpty) s" — ${job.detail}" else ""
                })
            )
          ),
          div(
            cls := "level-right",
            button(
              cls := "button is-small is-warning",
              "Cancel",
              onClick --> { _ =>
                conversionService.push(ConversionService.Command.Cancel(job.id))
              }
            )
          )
        ),
        job.progress match {
          case Some(progress) if counting(job) =>
            progressTag(
              cls := "progress is-small is-info mb-0",
              valueAttr := progress.completed.toString,
              maxAttr := progress.total.toString
            )
          case _ => progressTag(cls := "progress is-small is-info mb-0")
        }
      )
    case ConversionState.Failed =>
      div(
        cls := "notification is-danger is-light py-2 px-3 is-size-7 mb-1",
        p(
          cls := "has-text-weight-bold mb-1",
          s"Converting ${job.sourceLabel} → ${outputName(job)} failed: ${job.error
              .getOrElse("no reason given")}"
        ),
        if (job.logTail.isEmpty) emptyNode
        else
          pre(
            cls := "is-size-7",
            styleAttr := "white-space: pre-wrap; word-break: break-all; background: transparent; padding: 0; max-height: 12rem; overflow: auto;",
            job.logTail.mkString("\n")
          )
      )
    case ConversionState.Cancelled =>
      div(
        cls := "notification is-light py-2 px-3 is-size-7 mb-1",
        s"Converting ${job.sourceLabel} → ${outputName(job)} cancelled."
      )
    case ConversionState.Completed =>
      div(
        cls := "notification is-success is-light py-2 px-3 is-size-7 mb-1",
        s"✓ ${outputName(job)} registered as model ${job.modelId.getOrElse("?")} in family ${job.familyId}."
      )
  }

  lazy val element: HtmlElement = div(
    children <-- conversionService.jobs.map(_.map(row))
  )
}

object ConversionJobList {
  private val progressTag = htmlTag("progress")
  private val valueAttr = htmlAttr("value", StringAsIsCodec)
  private val maxAttr = htmlAttr("max", StringAsIsCodec)
}
