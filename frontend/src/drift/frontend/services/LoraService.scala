package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object LoraService {
  enum Command {
    case Load

    /** The official LoRAs the backend offers (`specs/33-lora-sources.md`). */
    case LoadCatalog
    case Install(
        architectureId: String,
        source: LoraInstallSource,
        grouping: LoraGrouping = LoraGrouping.Together
    )

    /** Re-attaches an orphan cache folder to an architecture — the entity is
      * rebuilt from the folder's sidecar (`path` is the folder or any file in
      * it, as the Model Cache page lists it).
      */
    case Adopt(path: String, architectureId: String)

    /** Makes one LoRA of two halves published apart
      * (`specs/33-lora-sources.md`), under a name of the user's choosing.
      */
    case Pair(loraId: String, otherId: String, label: String)

    /** Saves user-editable fields; an nsfw flip moves the cache folder. */
    case Update(lora: Lora)
    case Delete(loraId: String)
    case LoadJobs
  }
  enum Event {

    /** An orphan folder became an installed LoRA — the cache view's reference
      * index is stale now.
      */
    case Adopted(lora: Lora)
  }
}

/** The LoRA collection and its downloads (`specs/09-lora-management.md`). */
class LoraService(statusSocket: StatusSocketService) extends ServiceErrors {
  import LoraService.{Command, Event}

  private val listFn = ApiClient.stream(drift.shared.listLoras)
  private val catalogFn = ApiClient.stream(listLoraCatalog)
  private val installFn = ApiClient.stream(installLora)
  private val adoptFn = ApiClient.stream(adoptLora)
  private val pairFn = ApiClient.stream(pairLoras)
  private val updateFn = ApiClient.stream(updateLora)
  private val deleteFn = ApiClient.stream(deleteLora)
  private val jobsFn = ApiClient.stream(listLoraDownloads)

  private val _loras = Var(List.empty[Lora])
  val loras: Signal[List[Lora]] = _loras.signal

  /** The current value, for event handlers — `Signal.now()` is not public. */
  def lorasNow: List[Lora] = _loras.now()

  private val _listed = Var(false)

  /** The collection once a listing has arrived: `None` before, which an empty
    * collection is not — a recipe's LoRAs wait for the first, never for the
    * collection to be non-empty.
    */
  val loadedLoras: Signal[Option[List[Lora]]] =
    _listed.signal
      .combineWith(_loras.signal)
      .map((listed, all) => Option.when(listed)(all))

  def loadedLorasNow: Option[List[Lora]] =
    Option.when(_listed.now())(_loras.now())

  private val _catalog = Var(List.empty[Lora])

  /** What drift offers to install, installed entries included — an entry and
    * the LoRA installed from it share their id.
    */
  val catalog: Signal[List[Lora]] = _catalog.signal

  private val _jobs = Var(List.empty[LoraDownloadJob])
  val jobs: Signal[List[LoraDownloadJob]] = _jobs.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def upsert(lora: Lora): Unit =
    _loras.update(all =>
      (all.filterNot(_.id == lora.id) :+ lora).sortBy(_.label)
    )

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[Lora]]] {
        case Success(listing) =>
          clearError()
          _loras.set(listing.sortBy(_.label))
          _listed.set(true)
        case Failure(err) => reportFailure("Listing LoRAs", err)
      },
    cmdBus.events
      .collect { case Command.LoadCatalog => () }
      .flatMapSwitch(_ => catalogFn(()).recoverToTry)
      --> Observer[Try[List[Lora]]] {
        case Success(listing) => _catalog.set(listing.sortBy(_.label))
        case Failure(err) => reportFailure("Listing the official LoRAs", err)
      },
    cmdBus.events
      .collect { case Command.Install(architectureId, source, grouping) =>
        InstallLoraRequest(architectureId, source, grouping)
      }
      .flatMapMerge(request => installFn(request).recoverToTry)
      --> Observer[Try[InstallLoraResponse]] {
        case Success(response) =>
          response.error match {
            case Some(reason) => reportFailure("Installing the LoRA", reason)
            case None         =>
              clearError()
              response.lora.foreach(upsert)
              push(Command.LoadJobs)
          }
        case Failure(err) => reportFailure("Installing the LoRA", err)
      },
    cmdBus.events
      .collect { case Command.Adopt(path, architectureId) =>
        AdoptLoraRequest(path, architectureId)
      }
      .flatMapMerge(request => adoptFn(request).recoverToTry)
      --> Observer[Try[InstallLoraResponse]] {
        case Success(response) =>
          response.error match {
            case Some(reason) => reportFailure("Adopting the LoRA", reason)
            case None         =>
              clearError()
              response.lora.foreach { lora =>
                upsert(lora)
                evtBus.writer.onNext(Event.Adopted(lora))
              }
              push(Command.LoadJobs)
          }
        case Failure(err) => reportFailure("Adopting the LoRA", err)
      },
    cmdBus.events
      .collect { case Command.Pair(loraId, otherId, label) =>
        (otherId, PairLoraRequest(loraId, otherId, label))
      }
      .flatMapMerge((otherId, request) =>
        pairFn(request).map(response => (otherId, response)).recoverToTry
      )
      --> Observer[Try[(String, InstallLoraResponse)]] {
        case Success((otherId, response)) =>
          response.error match {
            case Some(reason) => reportFailure("Pairing the LoRAs", reason)
            case None         =>
              clearError()
              // The other half is gone: its files live in the one kept.
              _loras.update(_.filterNot(_.id == otherId))
              response.lora.foreach(upsert)
              push(Command.LoadJobs)
          }
        case Failure(err) => reportFailure("Pairing the LoRAs", err)
      },
    cmdBus.events
      .collect { case Command.Update(lora) => lora }
      .flatMapMerge(lora => updateFn((lora.id, lora)).recoverToTry)
      --> Observer[Try[Option[Lora]]] {
        case Success(Some(saved)) =>
          clearError()
          upsert(saved)
        case Success(None) =>
          reportFailure("Saving the LoRA", "it no longer exists")
        case Failure(err) => reportFailure("Saving the LoRA", err)
      },
    cmdBus.events
      .collect { case Command.Delete(loraId) => loraId }
      .flatMapMerge(id => deleteFn(id).map(ok => (id, ok)).recoverToTry)
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, _)) =>
          clearError()
          _loras.update(_.filterNot(_.id == id))
        case Failure(err) => reportFailure("Deleting the LoRA", err)
      },
    cmdBus.events
      .collect { case Command.LoadJobs => () }
      .flatMapSwitch(_ => jobsFn(()).recoverToTry)
      --> Observer[Try[List[LoraDownloadJob]]] {
        case Success(listing) => _jobs.set(listing)
        // Job polling is decoration; a failed poll reports nothing.
        case Failure(_) => ()
      },
    // The socket pushes the job list whenever it changes, so progress and
    // completion arrive on their own.
    statusSocket.loraDownloads --> Observer[List[LoraDownloadJob]](_jobs.set)
  )
}
