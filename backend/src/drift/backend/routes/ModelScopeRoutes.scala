package drift.backend.routes

import drift.backend.modelscope.ModelScopeApi
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** The ModelScope browser's endpoints (`specs/37-modelscope.md`), relaying
  * ModelScope through drift's own client of it; a failure answers 502 with
  * ModelScope's reason, as the HuggingFace card does.
  */
def modelScopeEndpoints(
    api: ModelScopeApi
): List[ServerEndpoint[Any, Identity]] = List(
  searchModelScope.serverLogic[Identity] {
    (query, page, sort, baseModels, lorasOnly) =>
      api.search(query, page, sort, baseModels, lorasOnly)
  },
  getModelScopeModelDetail.serverLogic[Identity] { (owner, name) =>
    api.detail(s"$owner/$name")
  },
  getModelScopeModelCard.serverLogic[Identity] { (owner, name) =>
    api.card(s"$owner/$name")
  }
)
