package drift.backend.huggingface

import drift.backend.sites.{rootReason, ModelDescriptions}
import drift.shared.*

import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.typesafe.scalalogging.Logger
import sttp.client3.{basicRequest, HttpURLConnectionBackend}
import sttp.model.Uri.*

/** HuggingFace's own calls: the search, a repository's files and its card.
  * `token` is resolved per request, like the download path's: a gated
  * repository hides its README from anonymous calls too (`specs/24`).
  */
class HuggingFaceApi(token: () => Option[String]) {
  private val hfClient = HttpURLConnectionBackend()
  private val hfLogger = Logger[HuggingFaceApi]

  private def hfRequest = basicRequest
    .header("Accept", "application/json")
    .header("User-Agent", "drift")

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

  def searchModels(
      query: String,
      limit: Int,
      offset: Int,
      sort: Option[String],
      direction: Option[Int],
      baseModel: Option[String]
  ): Either[String, List[HuggingFaceModelInfo]] = {
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
          Right(
            readFromString[List[HuggingFaceModelInfo]](json).map { model =>
              val found = examples.getOrElse(model.id, Nil)
              model.copy(previews = found.take(4), exampleCount = found.size)
            }
          )
        case Left(answer) =>
          hfLogger.warn(
            s"HuggingFace search answered ${response.code}: ${answer.take(300)}"
          )
          Left(s"HuggingFace answered ${response.code}.")
      }
    } catch {
      case NonFatal(error) =>
        hfLogger.warn("HuggingFace search failed", error)
        Left(s"HuggingFace could not be reached: ${rootReason(error)}")
    }
  }

  /** A repository's files and examples; none for one HuggingFace does not have,
    * and the reason when it could not be asked.
    */
  def getModelDetail(
      author: String,
      name: String
  ): Either[String, Option[HuggingFaceModelDetail]] =
    try {
      val response = hfRequest
        // `?blobs=true` is what makes `siblings[].lfs.sha256` appear; without it the
        // listing carries only filenames and nothing can be verified or cache-keyed.
        .get(uri"https://huggingface.co/api/models/$author/$name?blobs=true")
        .send(hfClient)
      response.body match {
        case Right(json) =>
          Right(
            Some(
              readFromString[HuggingFaceModelDetail](json)
                .copy(examples = HuggingFaceExamples.ofModel(json))
            )
          )
        case Left(_) if response.code.code == 404 => Right(None)
        case Left(_) => Left(s"HuggingFace answered ${response.code}.")
      }
    } catch {
      case NonFatal(error) =>
        Left(s"HuggingFace could not be reached: ${rootReason(error)}")
    }

  /** The repository's README as a rendered, cleaned card (`specs/24`). A gated
    * repository gates its README too — anonymously even FLUX.1-schnell's
    * answers 401 (2026-09-11) — so the token() goes along, and a refusal comes
    * back as `gated` rather than as an error. Relative links and images resolve
    * to the repository's files. Any other failure is the reason, which the card
    * tab shows beside a Retry.
    */
  def getModelCard(
      author: String,
      name: String
  ): Either[String, HuggingFaceModelCard] =
    try {
      val request = basicRequest
        .header("User-Agent", "drift")
        .get(uri"https://huggingface.co/$author/$name/raw/main/README.md")
      val response =
        token().fold(request)(t => request.auth.bearer(t)).send(hfClient)
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
                        cardGallery(author, name)
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
      name: String
  ): List[ModelExample] =
    try {
      val request =
        hfRequest.get(uri"https://huggingface.co/api/models/$author/$name")
      token()
        .fold(request)(t => request.auth.bearer(t))
        .send(hfClient)
        .body
        .toOption
        .map(HuggingFaceExamples.galleryOf)
        .getOrElse(Nil)
    } catch { case NonFatal(_) => Nil }
}
