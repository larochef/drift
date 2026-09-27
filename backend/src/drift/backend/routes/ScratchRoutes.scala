package drift.backend.routes

import drift.backend.sdserver.GenerationManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** Free play's write side (`specs/22-free-play-and-scratch-generations.md`):
  * keeping one result, and throwing the rest away.
  */
def scratchEndpoints(
    manager: GenerationManager
): List[ServerEndpoint[Any, Identity]] = List(
  keepScratchGeneration.serverLogicSuccess[Identity] {
    (generationId, request) =>
      manager.keep(generationId, request.projectId)
  },
  clearScratch.serverLogicSuccess[Identity](_ => manager.clearScratch())
)
