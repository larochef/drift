package drift.backend.routes

import drift.shared.*

import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import sttp.client3.{basicRequest, HttpURLConnectionBackend}
import sttp.model.Uri.*
import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint

private val hfClient = HttpURLConnectionBackend()

/** `token` is resolved per request, like the download path's: a gated
  * repository hides its README from anonymous calls too (`specs/24`).
  */
def huggingFaceEndpoints(
    token: () => Option[String]
): List[ServerEndpoint[Any, Identity]] = List(
  searchHuggingFaceModels.serverLogicSuccess[Identity] {
    (query, limit, offset, sort, direction, baseModel) =>
      searchModels(
        query,
        limit.getOrElse(20),
        offset.getOrElse(0),
        sort,
        direction,
        baseModel
      )
  },
  getHuggingFaceModelDetail.serverLogicSuccess[Identity] { (author, name) =>
    getModelDetail(author, name)
  },
  getHuggingFaceModelCard.serverLogic[Identity] { (author, name) =>
    getModelCard(author, name, token())
  }
)

private def hfRequest = basicRequest
  .header("Accept", "application/json")
  .header("User-Agent", "sd-wrapper/0.1.0")

private val ListingFields = List(
  "author",
  "downloads",
  "likes",
  "tags",
  "pipeline_tag",
  "library_name",
  "cardData",
  "siblings"
)

private def searchModels(
    query: String,
    limit: Int,
    offset: Int,
    sort: Option[String],
    direction: Option[Int],
    baseModel: Option[String]
): List[HuggingFaceModelInfo] = {
  try {
    val sortVal = sort.getOrElse("downloads")
    val dirVal = direction.getOrElse(-1)
    val parameters = List(
      "search" -> query,
      "limit" -> limit.toString,
      "skip" -> offset.toString,
      "sort" -> sortVal,
      "direction" -> dirVal.toString
    ) ++
      // HuggingFace ANDs repeated filters, so one base model per search.
      baseModel.map(repo => "filter" -> s"base_model:adapter:$repo") ++
      // An expanded listing carries only the fields it names: the card and
      // the files are where the examples are (`specs/36`).
      ListingFields.map(field => "expand[]" -> field)
    val response = hfRequest
      .get(uri"https://huggingface.co/api/models".addParams(parameters*))
      .send(hfClient)
    response.body match {
      case Right(json) =>
        val examples = HuggingFaceExamples.ofListing(json)
        readFromString[List[HuggingFaceModelInfo]](json).map { model =>
          val found = examples.getOrElse(model.id, Nil)
          model.copy(previews = found.take(4), exampleCount = found.size)
        }
      case Left(_) => List.empty
    }
  } catch {
    case _: Exception => List.empty
  }
}

private def getModelDetail(
    author: String,
    name: String
): Option[HuggingFaceModelDetail] = {
  try {
    val response = hfRequest
      // `?blobs=true` is what makes `siblings[].lfs.sha256` appear; without it the
      // listing carries only filenames and nothing can be verified or cache-keyed.
      .get(uri"https://huggingface.co/api/models/$author/$name?blobs=true")
      .send(hfClient)
    response.body match {
      case Right(json) =>
        Some(
          readFromString[HuggingFaceModelDetail](json)
            .copy(examples = HuggingFaceExamples.ofModel(json))
        )
      case Left(_) => None
    }
  } catch {
    case _: Exception => None
  }
}

/** The repository's README as a rendered, cleaned card (`specs/24`). A gated
  * repository gates its README too — anonymously even FLUX.1-schnell's answers
  * 401 (2026-09-11) — so the token goes along, and a refusal comes back as
  * `gated` rather than as an error. Relative links and images resolve to the
  * repository's files. Any other failure is the reason, which the card tab
  * shows beside a Retry.
  */
private def getModelCard(
    author: String,
    name: String,
    token: Option[String]
): Either[String, HuggingFaceModelCard] =
  try {
    val request = basicRequest
      .header("User-Agent", "sd-wrapper/0.1.0")
      .get(uri"https://huggingface.co/$author/$name/raw/main/README.md")
    val response =
      token.fold(request)(t => request.auth.bearer(t)).send(hfClient)
    response.code.code match {
      case 200 =>
        Right(
          HuggingFaceModelCard(html =
            response.body.toOption
              .map(markdown =>
                ModelDescriptions.fromMarkdown(
                  ModelDescriptions.withGallery(
                    markdown,
                    // Only when the card asks for it: one more request.
                    if (ModelDescriptions.hasGallery(markdown))
                      cardGallery(author, name, token)
                    else Nil
                  ),
                  s"https://huggingface.co/$author/$name/resolve/main/"
                )
              )
              .filter(_.trim.nonEmpty)
          )
        )
      case 401 | 403 => Right(HuggingFaceModelCard(gated = true))
      case 404       => Right(HuggingFaceModelCard())
      case other     => Left(s"HuggingFace answered $other.")
    }
  } catch {
    case NonFatal(error) =>
      Left(s"HuggingFace could not be reached: ${rootReason(error)}")
  }

/** The card's `widget` gallery, read from the model info rather than from the
  * README's YAML header, which drift does not parse. None when it cannot be
  * had: the card then renders without it.
  */
private def cardGallery(
    author: String,
    name: String,
    token: Option[String]
): List[ModelExample] =
  try {
    val request =
      hfRequest.get(uri"https://huggingface.co/api/models/$author/$name")
    token
      .fold(request)(t => request.auth.bearer(t))
      .send(hfClient)
      .body
      .toOption
      .map(HuggingFaceExamples.galleryOf)
      .getOrElse(Nil)
  } catch { case NonFatal(_) => Nil }

/** What went wrong in a failed request to another site (`specs/24`), in its
  * root cause's words — the client wraps "Read timed out" in "Exception when
  * sending request: GET …" — or the cause's type when it has none.
  */
private[routes] def rootReason(error: Throwable): String = {
  val root =
    Iterator.iterate(error)(_.getCause).takeWhile(_ != null).toList.last
  Option(root.getMessage)
    .filter(_.nonEmpty)
    .getOrElse(root.getClass.getSimpleName)
}
