package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object PostProcessService {
  enum Command {
    case LoadJobs
    case Upscale(date: String, fileName: String, request: UpscaleRequest)
    case Pid(date: String, fileName: String, request: PidUpscaleRequest)
    case SeedVr2(
        date: String,
        fileName: String,
        request: SeedVr2UpscaleRequest
    )
    case Redraw(date: String, fileName: String, request: RedrawRequest)
    case Edit(date: String, fileName: String, request: EditRequest)
    case Cancel(jobId: String)
    case Pause(jobId: String, force: Boolean)
    case Resume(jobId: String)
  }
}

/** Post-hoc upscale jobs (`specs/15-post-hoc-resize.md`). The backend runs them
  * and cancels them; the status socket mirrors the job list. A job's completion
  * is announced once on `completions` with the derived gallery entry, which the
  * gallery adopts into its listing.
  */
class PostProcessService(statusSocket: StatusSocketService)
    extends ServiceErrors {
  import PostProcessService.Command

  private val jobsFn = ApiClient.stream(listPostProcessJobs)
  private val upscaleFn = ApiClient.stream(upscaleOutput)
  private val pidFn = ApiClient.stream(pidUpscaleOutput)
  private val seedVr2Fn = ApiClient.stream(seedVr2UpscaleOutput)
  private val redrawFn = ApiClient.stream(redrawOutput)
  private val editFn = ApiClient.stream(editOutput)
  private val cancelFn = ApiClient.stream(cancelPostProcessJob)
  private val pauseFn = ApiClient.stream(pausePostProcessJob)
  private val resumeFn = ApiClient.stream(resumePostProcessJob)

  private val _jobs = Var(List.empty[PostProcessJob])

  /** Every job of this drift run, newest first. */
  val jobs: Signal[List[PostProcessJob]] = _jobs.signal

  private val _dismissed = Var(Set.empty[String])

  /** The jobs whose card is still open: `jobs` minus the finished ones closed
    * with the card's × (`specs/15-post-hoc-resize.md`). The job itself is kept
    * — its result is in the gallery — but a card left over from the first
    * redraw of an output must not be read as the second one's (François,
    * 2026-09-20). Browser-session state: a reload brings the cards back.
    */
  val visibleJobs: Signal[List[PostProcessJob]] =
    _jobs.signal
      .combineWith(_dismissed.signal)
      .map((all, closed) => all.filterNot(job => closed.contains(job.id)))

  def dismiss(jobId: String): Unit = _dismissed.update(_ + jobId)

  private val completionBus = new EventBus[Generation]
  val completions: EventStream[Generation] = completionBus.events
  private var announced = Set.empty[String]

  private val cmdBus = new EventBus[Command]

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def applyAll(listing: List[PostProcessJob]): Unit = {
    _jobs.set(listing.sortBy(-_.startedAt))
    listing
      .filter(_.state == PostProcessState.Completed)
      .flatMap(_.result)
      .filterNot(result => announced.contains(result.id))
      .foreach { result =>
        announced += result.id
        completionBus.writer.onNext(result)
      }
  }

  private def upsert(job: PostProcessJob): Unit =
    applyAll(_jobs.now().filterNot(_.id == job.id) :+ job)

  // A refused job answers as Failed with its reason — surfaced like any
  // other failed command, on top of the job row the detail view shows.
  private def submitted(action: String): Observer[Try[PostProcessJob]] =
    Observer {
      case Success(job) =>
        job.error.filter(_ => job.state == PostProcessState.Failed) match {
          case Some(reason) => reportFailure(action, reason)
          case None         => clearError()
        }
        upsert(job)
      case Failure(err) => reportFailure(action, err)
    }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.LoadJobs => () }
      .flatMapSwitch(_ => jobsFn(()).recoverToTry) --> Observer[
      Try[List[PostProcessJob]]
    ] {
      case Success(listing) => clearError(); applyAll(listing)
      case Failure(err) => reportFailure("Listing post-processing jobs", err)
    },
    cmdBus.events
      .collect { case Command.Upscale(date, fileName, request) =>
        (date, fileName, request)
      }
      .flatMapMerge(input => upscaleFn(input).recoverToTry) --> submitted(
      "Upscaling the image"
    ),
    cmdBus.events
      .collect { case Command.Pid(date, fileName, request) =>
        (date, fileName, request)
      }
      .flatMapMerge(input => pidFn(input).recoverToTry) --> submitted(
      "PiD upscaling the image"
    ),
    cmdBus.events
      .collect { case Command.SeedVr2(date, fileName, request) =>
        (date, fileName, request)
      }
      .flatMapMerge(input => seedVr2Fn(input).recoverToTry) --> submitted(
      "SeedVR2 upscaling"
    ),
    cmdBus.events
      .collect { case Command.Redraw(date, fileName, request) =>
        (date, fileName, request)
      }
      .flatMapMerge(input => redrawFn(input).recoverToTry) --> submitted(
      "Redrawing the image"
    ),
    cmdBus.events
      .collect { case Command.Edit(date, fileName, request) =>
        (date, fileName, request)
      }
      .flatMapMerge(input => editFn(input).recoverToTry) --> submitted(
      "Editing the image"
    ),
    cmdBus.events
      .collect { case Command.Pause(jobId, force) => (jobId, force) }
      .flatMapMerge((jobId, force) =>
        pauseFn((jobId, force)).map(jobId -> _).recoverToTry
      )
      --> Observer[Try[(String, Boolean)]] {
        // The paused job arrives on the socket once its tile ends.
        case Success((_, true))      => clearError()
        case Success((jobId, false)) =>
          reportFailure(
            "Pausing the job",
            s"'$jobId' is not a running tiled job any more."
          )
        case Failure(err) => reportFailure("Pausing the job", err)
      },
    cmdBus.events
      .collect { case Command.Resume(jobId) => jobId }
      .flatMapMerge(jobId => resumeFn(jobId).recoverToTry) --> submitted(
      "Resuming the job"
    ),
    cmdBus.events
      .collect { case Command.Cancel(jobId) => jobId }
      .flatMapMerge(jobId => cancelFn(jobId).map(jobId -> _).recoverToTry)
      --> Observer[Try[(String, Boolean)]] {
        // The job itself arrives on the socket a moment later, cancelled.
        case Success((_, true))      => clearError()
        case Success((jobId, false)) =>
          reportFailure(
            "Cancelling the job",
            s"'$jobId' was not running any more."
          )
        case Failure(err) => reportFailure("Cancelling the job", err)
      },
    statusSocket.postProcessJobs --> Observer[List[PostProcessJob]](applyAll)
  )
}
