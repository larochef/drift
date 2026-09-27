package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object RunConfigurationService {
  enum Command {
    case Load
    case Create(rm: RunConfiguration)
    case Delete(id: String)
    case Update(id: String, rm: RunConfiguration)
  }
  enum Event {
    case Created(rm: RunConfiguration)
    case Updated(id: String)
    case Deleted(id: String)
  }
}

class RunConfigurationService(
    val modelService: ModelService,
    val architectureService: ArchitectureService
) extends ServiceErrors {
  import RunConfigurationService.{Command, Event}

  private val listRunConfigurationsFn =
    ApiClient.stream(drift.shared.listRunConfigurations)
  private val createRunConfigurationFn =
    ApiClient.stream(drift.shared.createRunConfiguration)
  private val updateRunConfigurationFn =
    ApiClient.stream(drift.shared.updateRunConfiguration)
  private val deleteRunConfigurationFn =
    ApiClient.stream(drift.shared.deleteRunConfiguration)

  private def listRunConfigurations: EventStream[List[RunConfiguration]] =
    listRunConfigurationsFn(())
  private def createRunConfiguration(
      r: RunConfiguration
  ): EventStream[RunConfiguration] =
    createRunConfigurationFn(r)
  private def updateRunConfiguration(
      id: String,
      r: RunConfiguration
  ): EventStream[Option[RunConfiguration]] = updateRunConfigurationFn((id, r))
  private def deleteRunConfiguration(id: String): EventStream[Boolean] =
    deleteRunConfigurationFn(id)

  private val _runConfigurations = Var(List.empty[RunConfiguration])

  val runConfigurations: Signal[List[RunConfiguration]] =
    _runConfigurations.signal
  val architectures: Signal[List[Architecture]] =
    architectureService.architectures
  val allModels: Signal[List[Model]] = modelService.allModels

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  // See ModelService.effects for why writes merge instead of switching.
  val effects: Modifier[HtmlElement] = Seq(
    modelService.effects,
    architectureService.effects,
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listRunConfigurations.recoverToTry)
      --> Observer[Try[List[RunConfiguration]]] {
        case Success(configurations) =>
          clearError()
          _runConfigurations.set(configurations)
        case Failure(err) => reportFailure("Loading run configurations", err)
      },
    cmdBus.events
      .collect { case Command.Create(rm) => rm }
      .flatMapMerge(rm => createRunConfiguration(rm).recoverToTry)
      --> Observer[Try[RunConfiguration]] {
        case Success(rm) =>
          clearError()
          _runConfigurations.update(_ :+ rm)
          evtBus.writer.onNext(Event.Created(rm))
        case Failure(err) =>
          reportFailure("Creating the run configuration", err)
      },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge { id =>
        deleteRunConfiguration(id).map(deleted => (id, deleted)).recoverToTry
      } --> Observer[Try[(String, Boolean)]] {
      case Success((id, true)) =>
        clearError()
        _runConfigurations.update(_.filterNot(_.id == id))
        evtBus.writer.onNext(Event.Deleted(id))
      case Success((id, false)) =>
        reportFailure(
          "Deleting the run configuration",
          s"the server refused to delete '$id'."
        )
      case Failure(err) => reportFailure("Deleting the run configuration", err)
    },
    cmdBus.events
      .collect { case Command.Update(id, rm) => (id, rm) }
      .flatMapMerge { case (id, rm) =>
        updateRunConfiguration(id, rm)
          .map(saved => (id, rm, saved))
          .recoverToTry
      } --> Observer[Try[
      (String, RunConfiguration, Option[RunConfiguration])
    ]] {
      // The saved copy, not the one sent: the server drops overrides that
      // override nothing (`specs/16-parameter-resolution.md`).
      case Success((id, _, Some(saved))) =>
        clearError()
        _runConfigurations.update(_.map(r => if (r.id == id) saved else r))
        evtBus.writer.onNext(Event.Updated(id))
      case Success((id, _, None)) =>
        reportFailure(
          "Saving the run configuration",
          s"'$id' no longer exists."
        )
      case Failure(err) => reportFailure("Saving the run configuration", err)
    }
  )
}
