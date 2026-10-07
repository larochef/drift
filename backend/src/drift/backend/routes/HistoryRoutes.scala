package drift.backend.routes

import drift.backend.sdserver.*
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def historyEndpoints(
    history: GenerationHistory,
    imports: GenerationImports,
    moves: GenerationMoves
): List[ServerEndpoint[Any, Identity]] = List(
  listHistoryDays.serverLogicSuccess[Identity](_ => history.days),
  listHistoryDay.serverLogicSuccess[Identity](history.day),
  findHistoryGenerationDay.serverLogicSuccess[Identity](history.dayOf),
  deleteHistoryGeneration.serverLogicSuccess[Identity]((date, generationId) =>
    history.delete(date, generationId)
  ),
  importHistoryImage.serverLogic[Identity](imports.importImage),
  importHistoryVideo.serverLogic[Identity]((fileName, upload) =>
    imports.importVideo(fileName, upload.toPath)
  ),
  moveHistoryGenerations.serverLogic[Identity](moves.move)
)
