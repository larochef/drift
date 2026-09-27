package drift.frontend.services

import drift.shared.FileEntry

import scala.util.*

import com.raquo.laminar.api.L.*

object FileService {
  enum Command {
    case LoadHome
    case ListDirectory(path: String)
  }
  enum Event {
    case HomeLoaded(path: String)
  }
}

/** Server-side file system browsing, behind the same command/effects contract
  * as the other services so `FileBrowser` does not have to speak HTTP itself.
  */
class FileService extends ServiceErrors {
  import FileService.{Command, Event}

  private val homeFn = ApiClient.stream(drift.shared.getHomeDirectory)
  private val listFn = ApiClient.stream(drift.shared.listDirectory)

  private val _entries = Var(List.empty[FileEntry])
  val entries: Signal[List[FileEntry]] = _entries.signal

  private val _loading = Var(false)
  val loading: Signal[Boolean] = _loading.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  // Recovering matters most here: an unrecovered failure used to end the
  // navigation stream, so one unreadable directory made every later click in
  // the browser a no-op until the modal was reopened.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.LoadHome => () }
      .flatMapSwitch { _ =>
        _loading.set(true)
        homeFn(()).recoverToTry
      } --> Observer[Try[String]] {
      case Success(path) =>
        clearError()
        _loading.set(false)
        evtBus.writer.onNext(Event.HomeLoaded(path))
      case Failure(err) =>
        _loading.set(false)
        reportFailure("Reading the home directory", err)
    },
    cmdBus.events
      .collect { case Command.ListDirectory(path) => path }
      .flatMapSwitch { path =>
        _loading.set(true)
        listFn(path).recoverToTry.map(t => (path, t))
      } --> Observer[(String, Try[List[FileEntry]])] {
      case (_, Success(es)) =>
        clearError()
        _entries.set(es)
        _loading.set(false)
      case (path, Failure(err)) =>
        _entries.set(Nil)
        _loading.set(false)
        reportFailure(s"Reading '$path'", err)
    }
  )
}
