package drift.frontend.services

import drift.shared.Model

import scala.util.*

import com.raquo.laminar.api.L.*

object ModelService {
  enum Command {
    case Load
    case Create(m: Model)
    case Update(id: String, m: Model)
    case Delete(id: String)
  }
  enum Event {
    case Created(m: Model)
    case Updated(m: Model)
    case Deleted(id: String)
  }
}

class ModelService extends ServiceErrors {
  import ModelService.{Command, Event}

  private val listModelsFn = ApiClient.stream(drift.shared.listModels)
  private val createModelFn = ApiClient.stream(drift.shared.createModel)
  private val updateModelFn = ApiClient.stream(drift.shared.updateModel)
  private val deleteModelFn = ApiClient.stream(drift.shared.deleteModel)

  private def listModels: EventStream[List[Model]] = listModelsFn(())
  private def createModel(m: Model): EventStream[Model] = createModelFn(m)
  private def updateModel(id: String, m: Model): EventStream[Option[Model]] =
    updateModelFn((id, m))
  private def deleteModel(id: String): EventStream[Boolean] = deleteModelFn(id)

  private val _allModels = Var(List.empty[Model])
  val allModels: Signal[List[Model]] = _allModels.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  // Writes use flatMapMerge, not flatMapSwitch: switching unsubscribes from the
  // request already in flight, so two quick saves would leave the first one's
  // result unapplied even though the server stored it. Load keeps switching --
  // there the newest list is the only one worth having.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listModels.recoverToTry)
      --> Observer[Try[List[Model]]] {
        case Success(models) =>
          clearError()
          _allModels.set(models)
        case Failure(err) => reportFailure("Loading models", err)
      },
    cmdBus.events
      .collect { case Command.Create(m) => m }
      .flatMapMerge(m => createModel(m).recoverToTry)
      --> Observer[Try[Model]] {
        case Success(m) =>
          clearError()
          _allModels.update(_ :+ m)
          evtBus.writer.onNext(Event.Created(m))
        case Failure(err) => reportFailure("Creating the model", err)
      },
    cmdBus.events
      .collect { case Command.Update(id, m) => (id, m) }
      .flatMapMerge { case (id, m) =>
        updateModel(id, m).map(saved => (id, m, saved)).recoverToTry
      } --> Observer[Try[(String, Model, Option[Model])]] {
      // The saved copy, not the one sent: the server drops parameters that
      // override nothing (`specs/16-parameter-resolution.md`), and the list
      // has to show what is stored.
      case Success((id, _, Some(saved))) =>
        clearError()
        _allModels.update(
          _.map(existing => if (existing.id == id) saved else existing)
        )
        evtBus.writer.onNext(Event.Updated(saved))
      case Success((id, _, None)) =>
        reportFailure("Saving the model", s"'$id' no longer exists.")
      case Failure(err) => reportFailure("Saving the model", err)
    },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge { id =>
        deleteModel(id).map(deleted => (id, deleted)).recoverToTry
      } --> Observer[Try[(String, Boolean)]] {
      case Success((id, true)) =>
        clearError()
        _allModels.update(_.filterNot(_.id == id))
        evtBus.writer.onNext(Event.Deleted(id))
      case Success((id, false)) =>
        reportFailure(
          "Deleting the model",
          s"the server refused to delete '$id' (built-in entries are protected)."
        )
      case Failure(err) => reportFailure("Deleting the model", err)
    }
  )
}
