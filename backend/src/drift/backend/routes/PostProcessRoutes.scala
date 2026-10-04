package drift.backend.routes

import drift.backend.postprocess.PostProcessManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def postProcessEndpoints(
    manager: PostProcessManager
): List[ServerEndpoint[Any, Identity]] = List(
  upscaleOutput.serverLogicSuccess[Identity]((date, file, request) =>
    manager.upscale(date, file, request)
  ),
  pidUpscaleOutput.serverLogicSuccess[Identity]((date, file, request) =>
    manager.pidUpscale(date, file, request)
  ),
  seedVr2UpscaleOutput.serverLogicSuccess[Identity]((date, file, request) =>
    manager.seedVr2Upscale(date, file, request)
  ),
  redrawOutput.serverLogicSuccess[Identity]((date, file, request) =>
    manager.redraw(date, file, request)
  ),
  editOutput.serverLogicSuccess[Identity]((date, file, request) =>
    manager.edit(date, file, request)
  ),
  cancelPostProcessJob.serverLogicSuccess[Identity](id =>
    manager.cancelJob(id)
  ),
  pausePostProcessJob.serverLogicSuccess[Identity]((id, force) =>
    manager.pauseJob(id, force)
  ),
  resumePostProcessJob.serverLogicSuccess[Identity](id =>
    manager.resumeJob(id)
  ),
  listPostProcessJobs.serverLogicSuccess[Identity](_ => manager.listJobs),
  getPostProcessPicture.serverLogic[Identity] { (id, side, _) =>
    manager.picture(id, side).map(_ -> "image/png").toRight(())
  }
)
