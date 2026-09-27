package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object DownloadService {
  enum Command {
    case Load
    case Start(modelId: String)
    case Cancel(modelId: String)
  }
  enum Event {

    /** A download reached `Completed` — the cue for pages to reload the cache
      * status so indicators and launch blockers update.
      */
    case Finished(modelId: String)
  }
}

class DownloadService(statusSocket: StatusSocketService) extends ServiceErrors {
  import DownloadService.{Command, Event}

  private val listFn = ApiClient.stream(drift.shared.listDownloads)
  private val startFn = ApiClient.stream(drift.shared.startDownload)
  private val cancelFn = ApiClient.stream(drift.shared.cancelDownload)

  private val _jobs = Var(Map.empty[String, DownloadJob])
  val jobs: Signal[Map[String, DownloadJob]] = _jobs.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  /** Replaces the map with the polled list, emitting `Finished` for every job
    * that went from active to completed since the previous picture.
    */
  private def applyListing(listing: List[DownloadJob]): Unit = {
    val previous = _jobs.now()
    val next = listing.map(job => job.modelId -> job).toMap
    _jobs.set(next)
    next.values.foreach { job =>
      val wasActive = previous.get(job.modelId).forall(_.state.isActive)
      if (job.state == DownloadState.Completed && wasActive)
        evtBus.writer.onNext(Event.Finished(job.modelId))
    }
  }

  private def applyOne(job: DownloadJob): Unit = {
    val previous = _jobs.now().get(job.modelId)
    _jobs.update(_ + (job.modelId -> job))
    if (
      job.state == DownloadState.Completed &&
      previous.forall(_.state.isActive)
    ) evtBus.writer.onNext(Event.Finished(job.modelId))
  }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[DownloadJob]]] {
        case Success(listing) =>
          clearError()
          applyListing(listing)
        case Failure(err) => reportFailure("Listing downloads", err)
      },
    cmdBus.events
      .collect { case Command.Start(modelId) => modelId }
      .flatMapMerge(modelId => startFn(modelId).recoverToTry)
      --> Observer[Try[DownloadJob]] {
        case Success(job) =>
          clearError()
          applyOne(job)
        case Failure(err) => reportFailure("Starting the download", err)
      },
    cmdBus.events
      .collect { case Command.Cancel(modelId) => modelId }
      .flatMapMerge(modelId => cancelFn(modelId).recoverToTry)
      --> Observer[Try[DownloadJob]] {
        case Success(job) =>
          clearError()
          applyOne(job)
        case Failure(err) => reportFailure("Cancelling the download", err)
      },
    // The socket pushes the job list whenever it changes, so progress moves
    // without the user touching anything — through `applyListing`, so
    // `Finished` still fires for completions this page did not start.
    statusSocket.downloads --> Observer[List[DownloadJob]](applyListing)
  )
}
