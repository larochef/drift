package drift.backend.routes

import drift.backend.huggingface.HuggingFaceApi
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

def huggingFaceEndpoints(
    api: HuggingFaceApi
): List[ServerEndpoint[Any, Identity]] = List(
  searchHuggingFaceModels.serverLogicSuccess[Identity] {
    (query, limit, offset, sort, direction, baseModel) =>
      api.searchModels(
        query,
        limit.getOrElse(20),
        offset.getOrElse(0),
        sort,
        direction,
        baseModel
      )
  },
  getHuggingFaceModelDetail.serverLogicSuccess[Identity] { (author, name) =>
    api.getModelDetail(author, name)
  },
  getHuggingFaceModelCard.serverLogic[Identity] { (author, name) =>
    api.getModelCard(author, name)
  }
)
