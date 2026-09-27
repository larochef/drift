package drift.backend.routes

import drift.backend.cache.*
import drift.backend.storage.StorageService
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def cacheEndpoints(
    storage: StorageService,
    cache: ModelCache,
    inventory: CacheInventory
): List[ServerEndpoint[Any, Identity]] = List(
  listCachedFiles.serverLogicSuccess[Identity](_ => inventory.list),
  deleteCachedFile.serverLogicSuccess[Identity](inventory.delete),
  listCivitaiCachedFileIds.serverLogicSuccess[Identity](
    cache.civitaiCachedFileIds
  ),
  cacheStatus.serverLogicSuccess[Identity] { _ =>
    storage.list[Model]("models").map { model =>
      cache.resolve(model.source) match {
        case CacheEntry.Present(path, bytes) =>
          ModelCacheStatus(
            model.id,
            CacheState.Cached,
            Some(path.toString),
            Some(bytes)
          )
        case CacheEntry.Absent =>
          ModelCacheStatus(model.id, CacheState.Missing)
        case CacheEntry.BrokenLocal(path) =>
          ModelCacheStatus(model.id, CacheState.Broken, Some(path.toString))
      }
    }
  }
)
