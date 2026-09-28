package drift.backend.routes

import drift.backend.upscale.UpscalerManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** The manager owns everything here: installs queue transfers and delete
  * removes the weight file along with the entity.
  */
def upscalerEndpoints(
    manager: UpscalerManager
): List[ServerEndpoint[Any, Identity]] = List(
  listUpscalers.serverLogicSuccess[Identity](_ => manager.list),
  installUpscaler.serverLogicSuccess[Identity](manager.installFromUrl),
  installUpscalerFromCivitai.serverLogicSuccess[Identity](
    manager.installFromCivitai
  ),
  deleteUpscaler.serverLogicSuccess[Identity](manager.delete),
  listUpscalerDownloads.serverLogicSuccess[Identity](_ => manager.listJobs),
  cancelUpscalerDownload.serverLogicSuccess[Identity](manager.cancelDownload)
)
