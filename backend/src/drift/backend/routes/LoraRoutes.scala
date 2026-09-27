package drift.backend.routes

import drift.backend.lora.LoraManager
import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** Not `endpointsFor`: update moves the cache folder on an nsfw flip and delete
  * removes it, so the manager owns both.
  */
def loraEndpoints(
    storage: StorageService,
    manager: LoraManager
): List[ServerEndpoint[Any, Identity]] = List(
  listLoras.serverLogicSuccess[Identity](_ => storage.list[Lora]("loras")),
  listLoraCatalog.serverLogicSuccess[Identity](_ => manager.catalog),
  installLora.serverLogicSuccess[Identity](manager.install),
  adoptLora.serverLogicSuccess[Identity](manager.adopt),
  updateLora.serverLogicSuccess[Identity]((id, lora) =>
    manager.update(id, lora)
  ),
  pairLoras.serverLogicSuccess[Identity](manager.pair),
  deleteLora.serverLogicSuccess[Identity](manager.delete),
  listLoraDownloads.serverLogicSuccess[Identity](_ => manager.listJobs)
)
