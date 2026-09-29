package drift.backend.routes

import drift.backend.sdserver.GenerationManager
import drift.backend.session.SessionManager
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def sessionEndpoints(
    manager: SessionManager,
    /** Free play does not outlive its session
      * (`specs/22-free-play-and-scratch-generations.md`), and the dependency
      * only runs this way round — so the sweep is chained here rather than
      * inside `SessionManager`.
      */
    generations: GenerationManager
): List[ServerEndpoint[Any, Identity]] = List(
  listSessions.serverLogicSuccess[Identity](_ => manager.list),
  launchSession.serverLogicSuccess[Identity](request =>
    manager.launch(
      request.runConfigurationId,
      request.runtimeId,
      request.projectId
    )
  ),
  stopSession.serverLogicSuccess[Identity] { sessionId =>
    val stopped = manager.stop(sessionId)
    // Only the generation sessions: an assistant session makes no scratch
    // outputs, and stopping one must not throw away the image results sitting
    // beside it.
    if (stopped.exists(_.tool == RuntimeTool.SdCpp)) generations.clearScratch()
    stopped
  },
  // The scratch outputs stay: the same model comes back, and what it made is
  // still the user's to keep.
  restartSession.serverLogicSuccess[Identity](manager.restart)
)
