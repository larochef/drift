package drift.backend.routes

import drift.backend.conversion.ConversionManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** Model conversion (`specs/25-model-conversion.md`): inspect a cached file,
  * queue its conversion, list and cancel the jobs.
  */
def conversionEndpoints(
    manager: ConversionManager
): List[ServerEndpoint[Any, Identity]] = List(
  inspectCachedFile.serverLogic[Identity](manager.inspect),
  startConversion.serverLogic[Identity](manager.start),
  listConversions.serverLogicSuccess[Identity](_ => manager.listJobs),
  cancelConversion.serverLogicSuccess[Identity](manager.cancel)
)
