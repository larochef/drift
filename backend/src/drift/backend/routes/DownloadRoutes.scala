package drift.backend.routes

import drift.backend.download.DownloadManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def downloadEndpoints(
    manager: DownloadManager
): List[ServerEndpoint[Any, Identity]] = List(
  listDownloads.serverLogicSuccess[Identity](_ => manager.list),
  startDownload.serverLogicSuccess[Identity](manager.start),
  cancelDownload.serverLogicSuccess[Identity](manager.cancel)
)
