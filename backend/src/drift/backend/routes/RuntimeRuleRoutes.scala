package drift.backend.routes

import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** The runtime rules drift ships (`specs/16-parameter-resolution.md`), so the
  * command-line preview resolves exactly what the launcher will.
  */
def runtimeRuleEndpoints(
    storage: StorageService
): List[ServerEndpoint[Any, Identity]] = List(
  listRuntimeRules.serverLogicSuccess[Identity](_ => storage.runtimeRules)
)
