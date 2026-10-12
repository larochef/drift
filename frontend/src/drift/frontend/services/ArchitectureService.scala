package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

object ArchitectureService {
  type Command = EntityService.Command[Architecture]
  val Command: EntityService.Command.type = EntityService.Command
  type Event = EntityService.Event[Architecture]
  val Event: EntityService.Event.type = EntityService.Event
}

class ArchitectureService
    extends EntityService[Architecture](
      ApiClient.stream(listArchitectures),
      ApiClient.stream(createArchitecture),
      ApiClient.stream(updateArchitecture),
      ApiClient.stream(deleteArchitecture),
      _.id,
      EntityWording(
        loading = "Loading architectures",
        creating = "Creating the architecture",
        saving = "Saving the architecture",
        deleting = "Deleting the architecture",
        missing = EntityWording.noLongerExists,
        refused = EntityWording.builtInsAreProtected
      )
    ) {
  val architectures: Signal[List[Architecture]] = entities.signal
}
