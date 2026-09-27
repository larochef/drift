package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object ConversionService {
  enum Command {
    case Load
    case Start(request: ConversionRequest)
    case Cancel(id: String)
  }
  enum Event {

    /** A conversion reached `Completed` — the cue for the Model Cache page to
      * reload the files and the models, since both gained an entry.
      */
    case Finished(job: ConversionJob)

    /** The backend refused a `Start`, with its reason — shown by the modal that
      * asked, beside the form, rather than in the page banner.
      */
    case Refused(reason: String)
  }
}

/** Model conversion (`specs/25-model-conversion.md`): the jobs of this drift
  * run, live off the status socket, and the commands that start and cancel
  * them. Inspecting a file is a plain request the modal makes on its own.
  */
class ConversionService(statusSocket: StatusSocketService)
    extends ServiceErrors {
  import ConversionService.{Command, Event}

  private val listFn = ApiClient.stream(drift.shared.listConversions)
  private val startFn =
    ApiClient.streamWithFailureReason(drift.shared.startConversion)
  private val cancelFn = ApiClient.stream(drift.shared.cancelConversion)

  /** What a cached file holds; fails with the backend's reason. */
  val inspect: String => EventStream[ModelFileInfo] =
    ApiClient.streamWithFailureReason(drift.shared.inspectCachedFile)

  private val _jobs = Var(List.empty[ConversionJob])

  /** Newest first, as the backend lists them. */
  val jobs: Signal[List[ConversionJob]] = _jobs.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  /** Replaces the list, emitting `Finished` for every job that went from active
    * to completed since the previous picture.
    */
  private def applyListing(listing: List[ConversionJob]): Unit = {
    val previous = _jobs.now().map(job => job.id -> job).toMap
    _jobs.set(listing)
    listing.foreach { job =>
      val wasActive = previous.get(job.id).forall(_.state.isActive)
      if (job.state == ConversionState.Completed && wasActive)
        evtBus.writer.onNext(Event.Finished(job))
    }
  }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[ConversionJob]]] {
        case Success(listing) =>
          clearError()
          applyListing(listing)
        case Failure(err) => reportFailure("Listing conversions", err)
      },
    cmdBus.events
      .collect { case Command.Start(request) => request }
      .flatMapMerge(request => startFn(request).recoverToTry)
      --> Observer[Try[ConversionJob]] {
        case Success(job) =>
          clearError()
          _jobs.update(job :: _.filterNot(_.id == job.id))
        case Failure(err) =>
          evtBus.writer.onNext(Event.Refused(err.getMessage))
      },
    cmdBus.events
      .collect { case Command.Cancel(id) => id }
      .flatMapMerge(id => cancelFn(id).map(id -> _).recoverToTry)
      --> Observer[Try[(String, Boolean)]] {
        case Success((_, true))   => clearError()
        case Success((id, false)) =>
          reportFailure(
            "Cancelling the conversion",
            s"'$id' was not running any more."
          )
        case Failure(err) => reportFailure("Cancelling the conversion", err)
      },
    // The socket pushes the list whenever a job moves, so the bars advance
    // without the user touching anything — through `applyListing`, so
    // `Finished` fires for completions this page did not start.
    statusSocket.conversions --> Observer[List[ConversionJob]](applyListing)
  )
}
