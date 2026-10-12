package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

object ModelService {
  type Command = EntityService.Command[Model]
  val Command: EntityService.Command.type = EntityService.Command
  type Event = EntityService.Event[Model]
  val Event: EntityService.Event.type = EntityService.Event
}

class ModelService
    extends EntityService[Model](
      ApiClient.stream(listModels),
      ApiClient.stream(createModel),
      ApiClient.stream(updateModel),
      ApiClient.stream(deleteModel),
      _.id,
      EntityWording(
        loading = "Loading models",
        creating = "Creating the model",
        saving = "Saving the model",
        deleting = "Deleting the model",
        missing = EntityWording.noLongerExists,
        refused = EntityWording.builtInsAreProtected
      )
    ) {
  val allModels: Signal[List[Model]] = entities.signal
}
