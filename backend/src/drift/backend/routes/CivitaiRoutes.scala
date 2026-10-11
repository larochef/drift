package drift.backend.routes

import drift.backend.civitai.CivitaiClient
import drift.backend.sites.ModelDescriptions
import drift.shared.*

import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

/** The detail as the browser gets it, descriptions cleaned for insertion
  * (`specs/24`). The LoRA and download paths call the client directly and keep
  * reading the raw detail.
  */
private def cleanedForBrowser(
    detail: CivitaiModelDetail
): CivitaiModelDetail = {
  def clean(html: Option[String]): Option[String] =
    html
      .map(ModelDescriptions.fromHtml(_, "https://civitai.com/"))
      .filter(_.trim.nonEmpty)
  detail.copy(
    description = clean(detail.description),
    modelVersions = detail.modelVersions.map(version =>
      version.copy(description = clean(version.description))
    )
  )
}

def civitaiEndpoints(
    civitaiClient: CivitaiClient
): List[ServerEndpoint[Any, Identity]] = List(
  searchCivitaiModels.serverLogicSuccess[Identity] {
    (query, limit, cursor, sort, modelType, baseModels, nsfw) =>
      civitaiClient.search(
        query,
        modelType,
        baseModels,
        limit.getOrElse(20),
        cursor,
        sort,
        nsfw.getOrElse(false)
      )
  },
  getCivitaiModelDetail.serverLogicSuccess[Identity] { modelId =>
    civitaiClient.getModelDetail(modelId).map(cleanedForBrowser)
  },
  listCivitaiImages.serverLogic[Identity] {
    (modelId, modelVersionId, sort, cursor, nsfw) =>
      civitaiClient.images(
        modelId,
        modelVersionId,
        sort,
        cursor,
        nsfw.getOrElse(false)
      )
  }
)
