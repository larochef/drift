package drift.backend.routes

import drift.backend.runtime.{RuntimeCatalog, RuntimeManager}
import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def runtimeEndpoints(
    storage: StorageService,
    manager: RuntimeManager,
    catalog: RuntimeCatalog
): List[ServerEndpoint[Any, Identity]] =
  endpointsFor[Runtime](
    storage,
    "runtimes",
    _.id,
    listRuntimes,
    getRuntime,
    createRuntime,
    updateRuntime,
    deleteRuntime,
    onDeleted = manager.cleanupDeleted
  ) ++ List(
    validateRuntime.serverLogicSuccess[Identity](manager.revalidate),
    getRuntimeSelection.serverLogicSuccess[Identity](_ => manager.selection),
    setRuntimeSelection.serverLogicSuccess[Identity](manager.saveSelection),
    listRuntimeReleases.serverLogicSuccess[Identity](catalog.releases),
    listTheRockTargets.serverLogicSuccess[Identity](_ =>
      catalog.theRockTargets
    ),
    resolveTheRock.serverLogicSuccess[Identity]((gfx, rocm) =>
      manager.resolveTheRock(gfx, rocm)
    ),
    installRuntime.serverLogicSuccess[Identity](manager.install),
    runnerOffer.serverLogicSuccess[Identity](_ => manager.runnerOffer),
    installRunner.serverLogicSuccess[Identity](manager.installRunner),
    installLatestRuntime.serverLogicSuccess[Identity](request =>
      manager.installLatest(
        request.tool,
        request.backend,
        request.gfxTarget,
        request.theRockVersion
      )
    ),
    upgradeRuntime.serverLogicSuccess[Identity]((id, request) =>
      manager.upgrade(id, request.theRockVersion)
    ),
    changeRuntimeTheRock.serverLogicSuccess[Identity]((id, request) =>
      manager.changeTheRock(id, request.theRockVersion)
    ),
    listRuntimeInstalls.serverLogicSuccess[Identity](_ => manager.listInstalls),
    cancelRuntimeInstall.serverLogicSuccess[Identity](manager.cancelInstall)
  )
