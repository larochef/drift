package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

object RunConfigurationService {
  type Command = EntityService.Command[RunConfiguration]
  val Command: EntityService.Command.type = EntityService.Command
  type Event = EntityService.Event[RunConfiguration]
  val Event: EntityService.Event.type = EntityService.Event
}

class RunConfigurationService(
    val modelService: ModelService,
    val architectureService: ArchitectureService
) extends EntityService[RunConfiguration](
      ApiClient.stream(listRunConfigurations),
      ApiClient.stream(createRunConfiguration),
      ApiClient.stream(updateRunConfiguration),
      ApiClient.stream(deleteRunConfiguration),
      _.id,
      EntityWording(
        loading = "Loading run configurations",
        creating = "Creating the run configuration",
        saving = "Saving the run configuration",
        deleting = "Deleting the run configuration",
        missing = EntityWording.noLongerExists,
        refused = id => s"the server refused to delete '$id'."
      )
    ) {
  val runConfigurations: Signal[List[RunConfiguration]] = entities.signal
  val architectures: Signal[List[Architecture]] =
    architectureService.architectures
  val allModels: Signal[List[Model]] = modelService.allModels
}
