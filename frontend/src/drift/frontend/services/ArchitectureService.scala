package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object ArchitectureService {
  enum Command {
    case Load
    case Create(a: Architecture)
    case Delete(id: String)
    case Update(id: String, a: Architecture)
  }
  enum Event {
    case Created(a: Architecture)
    case Updated(id: String)
    case Deleted(id: String)
  }
}

class ArchitectureService extends ServiceErrors {
  import ArchitectureService.{Command, Event}

  private val listArchitecturesFn =
    ApiClient.stream(drift.shared.listArchitectures)
  private val createArchitectureFn =
    ApiClient.stream(drift.shared.createArchitecture)
  private val updateArchitectureFn =
    ApiClient.stream(drift.shared.updateArchitecture)
  private val deleteArchitectureFn =
    ApiClient.stream(drift.shared.deleteArchitecture)

  private def list: EventStream[List[Architecture]] = listArchitecturesFn(())
  private def create(a: Architecture): EventStream[Architecture] =
    createArchitectureFn(a)
  private def update(
      id: String,
      a: Architecture
  ): EventStream[Option[Architecture]] = updateArchitectureFn((id, a))
  private def delete(id: String): EventStream[Boolean] =
    deleteArchitectureFn(id)

  private val _architectures = Var(List.empty[Architecture])
  val architectures: Signal[List[Architecture]] = _architectures.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  // See ModelService.effects for why writes merge instead of switching.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => list.recoverToTry)
      --> Observer[Try[List[Architecture]]] {
        case Success(archs) =>
          clearError()
          _architectures.set(archs)
        case Failure(err) => reportFailure("Loading architectures", err)
      },
    cmdBus.events
      .collect { case Command.Create(a) => a }
      .flatMapMerge(a => create(a).recoverToTry)
      --> Observer[Try[Architecture]] {
        case Success(a) =>
          clearError()
          _architectures.update(_ :+ a)
          evtBus.writer.onNext(Event.Created(a))
        case Failure(err) => reportFailure("Creating the architecture", err)
      },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge { id => delete(id).map(d => (id, d)).recoverToTry }
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, true)) =>
          clearError()
          _architectures.update(_.filterNot(_.id == id))
          evtBus.writer.onNext(Event.Deleted(id))
        case Success((id, false)) =>
          reportFailure(
            "Deleting the architecture",
            s"the server refused to delete '$id' (built-in entries are protected)."
          )
        case Failure(err) => reportFailure("Deleting the architecture", err)
      },
    cmdBus.events
      .collect { case Command.Update(id, a) => (id, a) }
      .flatMapMerge { case (id, a) =>
        update(id, a).map(saved => (id, a, saved)).recoverToTry
      } --> Observer[Try[(String, Architecture, Option[Architecture])]] {
      case Success((id, a, Some(_))) =>
        clearError()
        _architectures.update(_.map(aa => if (aa.id == id) a else aa))
        evtBus.writer.onNext(Event.Updated(id))
      case Success((id, _, None)) =>
        reportFailure("Saving the architecture", s"'$id' no longer exists.")
      case Failure(err) => reportFailure("Saving the architecture", err)
    }
  )
}
