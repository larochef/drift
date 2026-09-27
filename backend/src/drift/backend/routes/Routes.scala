package drift.backend.routes

import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** Built-ins are the reference file's: a create cannot claim the flag, an
  * update of one is answered with the stored copy, a delete refused
  * (`specs/32-prompt-library.md`).
  */
def promptTemplateEndpoints(
    storage: StorageService
): List[ServerEndpoint[Any, Identity]] =
  endpointsFor[PromptTemplate](
    storage,
    "prompt-templates",
    _.id,
    listPromptTemplates,
    getPromptTemplate,
    createPromptTemplate,
    updatePromptTemplate,
    deletePromptTemplate,
    canDelete = !_.builtIn,
    mergeUpdate = (stored, incoming) =>
      if (stored.builtIn) stored else incoming.copy(builtIn = false),
    onCreate = _.copy(builtIn = false)
  )

def architectureEndpoints(
    storage: StorageService
): List[ServerEndpoint[Any, Identity]] =
  endpointsFor[Architecture](
    storage,
    "architectures",
    _.id,
    listArchitectures,
    getArchitecture,
    createArchitecture,
    updateArchitecture,
    deleteArchitecture,
    canDelete = !_.builtIn
  )

/** Only what actually overrides something is stored
  * (`specs/16-parameter-resolution.md`): a parameter repeating the default of
  * every architecture that can assign the model changes no command line, and
  * the effective list is computed at launch anyway.
  */
def modelEndpoints(
    storage: StorageService
): List[ServerEndpoint[Any, Identity]] = {
  def pruned(model: Model): Model =
    ParameterOverrides.prunedModel(
      model,
      storage.list[Architecture]("architectures")
    )

  endpointsFor[Model](
    storage,
    "models",
    _.id,
    listModels,
    getModel,
    createModel,
    updateModel,
    deleteModel,
    canDelete = !_.builtIn,
    mergeUpdate = (_, incoming) => pruned(incoming),
    onCreate = pruned
  )
}

/** Pruned like a model, against its architecture and its assigned models —
  * never against a runtime, which is chosen at launch
  * (`specs/16-parameter-resolution.md`).
  */
def runConfigurationEndpoints(
    storage: StorageService
): List[ServerEndpoint[Any, Identity]] = {
  def pruned(configuration: RunConfiguration): RunConfiguration =
    ParameterOverrides.prunedConfiguration(
      configuration,
      storage.get[Architecture]("architectures", configuration.architectureId),
      storage.list[Model]("models")
    )

  endpointsFor[RunConfiguration](
    storage,
    "run-configurations",
    _.id,
    listRunConfigurations,
    getRunConfiguration,
    createRunConfiguration,
    updateRunConfiguration,
    deleteRunConfiguration,
    mergeUpdate = (_, incoming) => pruned(incoming),
    onCreate = pruned
  )
}
