package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Runtime installs in progress, or ended, with a cancel while one runs. On the
  * page rather than in the add modal: an install outlives the form that started
  * it, and closing the modal must not hide it.
  */
class RuntimeInstallJobs(runtimeService: RuntimeService) extends Component {

  lazy val element: HtmlElement = div(
    children <-- runtimeService.installs.map { jobs =>
      jobs.values.toList.sortBy(_.runtimeId).map { job =>
        val percent = job.totalBytes
          .filter(_ > 0)
          .map(total => s"${(job.downloadedBytes * 100 / total).toInt}%")
          .getOrElse(LaunchBlocker.humanBytes(job.downloadedBytes))
        div(
          cls := "level is-mobile mb-1 is-marginless",
          div(
            cls := "level-left",
            div(
              p(cls := "text-primary is-size-7", job.runtimeId),
              p(
                cls := "text-secondary is-size-7",
                job.state match {
                  case RuntimeInstallState.Downloading =>
                    s"downloading ${job.step} — $percent"
                  case RuntimeInstallState.Unpacking =>
                    s"unpacking ${job.step}"
                  case RuntimeInstallState.Validating => "validating"
                  case other => other.toString.toLowerCase
                }
              ),
              job.error match {
                case Some(error) =>
                  p(cls := "has-text-danger is-size-7", error)
                case None => emptyNode
              }
            )
          ),
          div(
            cls := "level-right",
            span(
              cls := (job.state match {
                case RuntimeInstallState.Completed => "tag is-success is-small"
                case RuntimeInstallState.Failed    => "tag is-danger is-small"
                case RuntimeInstallState.Cancelled => "tag is-dark is-small"
                case _                             => "tag is-info is-small"
              }),
              job.state.toString.toLowerCase
            ),
            if (job.state.isActive)
              button(
                cls := "button is-small ml-1",
                "✕",
                title := "Cancel install",
                onClick --> { _ =>
                  runtimeService.push(
                    RuntimeService.Command.CancelInstall(job.runtimeId)
                  )
                }
              )
            else emptyNode
          )
        )
      }
    }
  )
}
