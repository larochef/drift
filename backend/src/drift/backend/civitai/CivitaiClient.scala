package drift.backend.civitai

import drift.backend.sites.rootReason
import drift.shared.*

import scala.util.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromString,
  JsonValueCodec
}
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import com.typesafe.scalalogging.StrictLogging
import sttp.client3.{basicRequest, HttpURLConnectionBackend}
import sttp.model.Uri
import sttp.model.Uri.*

/** `token` is resolved per request, like the download path's — a token saved in
  * Settings applies without a restart. Sending it matters beyond rate limits:
  * Civitai hides some models (early access, account-gated content) from
  * anonymous API calls.
  */
class CivitaiClient(token: () => Option[String] = () => None)
    extends StrictLogging {
  private val backend = HttpURLConnectionBackend()

  /** Posts per gallery page (`specs/24`): ten, below the version's own images —
    * the search is slow and often overloaded, and ranks somewhat at random.
    */
  private val ImagePageSize = 10

  def search(
      query: Option[String],
      modelType: Option[String],
      baseModels: List[String],
      limit: Int,
      cursor: Option[String],
      sort: Option[String],
      nsfw: Boolean
  ): Either[String, CivitaiSearchResult] = {
    try {
      val sortValue = sort.getOrElse("Most Downloaded")
      val typeParam = toCivitaiType(modelType.getOrElse("CHECKPOINT"))

      // Civitai's text-search index does not support `baseModels`: sending
      // `query` and `baseModels` together answers an empty list (verified
      // live 2026-09-03 — query=turbo + baseModels=Flux.2 D returned 0 while
      // either alone returned pages). With a query, the base-model scoping is
      // applied here instead, on the versions of the returned items.
      val filterLocally = query.isDefined && baseModels.nonEmpty

      // Their query search also ranks by opaque relevance (ignoring `sort`)
      // and pages *differently* at different limits — four 25-item pages walk
      // a poorer set than one 100-item call and can miss models entirely. So when filtering locally, ask for the largest pages.
      val civitaiPageLimit = if (filterLocally) 100 else limit

      // The search parameter is `query` — `q` is silently ignored, which made
      // every search return the unfiltered most-downloaded list.
      // `nsfw` is always sent explicitly, never left absent: the absent-param
      // default hides models the site shows, some of them SFW-flagged. The browser's "Include NSFW" checkbox picks the value.
      def pageUri(pageCursor: Option[String]): Uri = {
        var uri = uri"https://civitai.com/api/v1/models"
          .addParam("types", typeParam)
          .addParam("sort", sortValue)
          .addParam("limit", civitaiPageLimit.toString)
          .addParam("nsfw", nsfw.toString)
        query.foreach(q => uri = uri.addParam("query", q))
        pageCursor.foreach(c => uri = uri.addParam("cursor", c))
        if (!filterLocally)
          baseModels.foreach(bm => uri = uri.addParam("baseModels", bm))
        uri
      }

      def fetchPage(
          pageCursor: Option[String]
      ): (List[CivitaiModelListInfo], Option[String]) = {
        val uri = pageUri(pageCursor)
        logger.debug(s"Civitai search URI: $uri")
        val response = authorized(
          basicRequest.header("Accept", "application/json").get(uri)
        ).send(backend)
        response.body match {
          case Right(json) =>
            Try(readFromString[CivitaiSearchResponse](json)) match {
              case Success(parsed) =>
                val next = parsed.metadata.flatMap(_.nextCursor).map(_.value)
                logger.debug(
                  s"Civitai search: ${parsed.items.size} items, nextCursor=$next"
                )
                (parsed.items, next)
              case Failure(err) =>
                logger.error(
                  s"Civitai search decode error: $err | json=${json.take(500)}"
                )
                throw CivitaiClient.Refused(
                  "Civitai's answer could not be read"
                )
            }
          case Left(body) =>
            logger.error(s"Civitai search response error: $body")
            throw CivitaiClient.Refused(
              s"Civitai answered ${response.code}: ${civitaiReason(body)}"
            )
        }
      }

      def matchesBaseModels(item: CivitaiModelListInfo): Boolean =
        !filterLocally ||
          item.modelVersions.exists(_.baseModel.exists(baseModels.contains))

      var (items, nextCursor) = fetchPage(cursor)
      var kept = items.filter(matchesBaseModels)
      // A generic query can filter most of a page away while matches sit
      // deeper; keep following the cursor until the requested count is met
      // (bounded) rather than showing a thin or empty page.
      var extraPages = 0
      while (kept.size < limit && nextCursor.isDefined && extraPages < 3) {
        extraPages += 1
        val (more, next) = fetchPage(nextCursor)
        kept = kept ++ more.filter(matchesBaseModels)
        nextCursor = next
      }
      Right(
        CivitaiSearchResult(
          items = kept.map(withoutDescriptions),
          nextCursor = nextCursor
        )
      )
    } catch {
      case CivitaiClient.Refused(reason) => Left(reason)
      case NonFatal(error)               =>
        logger.error(s"Civitai search exception", error)
        Left(s"Civitai could not be reached: ${rootReason(error)}")
    }
  }

  /** The search grid never shows a description, and descriptions are raw HTML
    * written by strangers — the bulk of a page, too. They stay out of its
    * payload (`specs/24`); the opened model's come cleaned, from the detail
    * endpoint.
    */
  private def withoutDescriptions(
      item: CivitaiModelListInfo
  ): CivitaiModelListInfo =
    item.copy(
      description = None,
      modelVersions = item.modelVersions.map(_.copy(description = None))
    )

  /** One page of the images and videos people posted with a model, or with one
    * of its versions (`specs/24`). `nsfw` is always sent, and as a *level*
    * (probed 2026-09-11): the absent parameter answered 503 and `true` returned
    * only NSFW posts, while `X` returns every level and `None` only safe ones.
    * A failure comes back as the reason, in Civitai's words when it gives some
    * — the gallery shows it beside a Retry, rather than an empty gallery that
    * reads as a model nobody used.
    */
  def images(
      modelId: Int,
      modelVersionId: Option[Int],
      sort: Option[String],
      cursor: Option[String],
      nsfw: Boolean
  ): Either[String, CivitaiImagePage] = {
    var uri = uri"https://civitai.com/api/v1/images"
      .addParam("modelId", modelId.toString)
      .addParam("limit", ImagePageSize.toString)
      .addParam("sort", sort.getOrElse("Most Reactions"))
      .addParam("nsfw", if (nsfw) "X" else "None")
    modelVersionId.foreach(id =>
      uri = uri.addParam("modelVersionId", id.toString)
    )
    cursor.foreach(c => uri = uri.addParam("cursor", c))
    try {
      val response = authorized(
        basicRequest.header("Accept", "application/json").get(uri)
      ).send(backend)
      response.body match {
        case Right(json) =>
          Try(readFromString[CivitaiImagesResponse](json)) match {
            case Success(page) =>
              Right(
                CivitaiImagePage(
                  page.items,
                  page.metadata.flatMap(_.nextCursor).map(_.value)
                )
              )
            case Failure(error) =>
              logger.error(
                s"Civitai images decode error: $error | json=${json.take(500)}"
              )
              Left(
                "Civitai's answer could not be read: its format may have changed."
              )
          }
        case Left(body) =>
          logger.warn(
            s"Civitai images answered ${response.code}: ${body.take(500)}"
          )
          Left(s"Civitai answered ${response.code}: ${civitaiReason(body)}")
      }
    } catch {
      // The throwable goes as SLF4J's exception argument: interpolated, it
      // became one anyway and the message read "unreachable: {}".
      case NonFatal(error) =>
        logger.warn("Civitai images unreachable", error)
        Left(s"Civitai could not be reached: ${rootReason(error)}")
    }
  }

  /** Where a file of a version is fetched from. */
  def downloadAddress(versionId: String, fileId: String): String =
    s"https://civitai.com/api/download/models/$versionId?fileId=$fileId"

  /** What goes with a request to Civitai: the token, when there is one. */
  def downloadHeaders: Map[String, String] =
    token().map(saved => "Authorization" -> s"Bearer $saved").toMap

  /** The same for an address that may not be Civitai's: the token goes to
    * Civitai only — sent to github.com or huggingface.co it would leak.
    */
  def headersFor(url: String): Map[String, String] = {
    val civitai =
      try {
        val host = Option(java.net.URI.create(url).getHost).getOrElse("")
        host == "civitai.com" || host.endsWith(".civitai.com")
      } catch { case NonFatal(_) => false }
    if (civitai) downloadHeaders else Map.empty
  }

  private def authorized(
      request: sttp.client3.Request[Either[String, String], Any]
  ): sttp.client3.Request[Either[String, String], Any] =
    token().fold(request)(t => request.auth.bearer(t))

  /** A model with its versions and files, or why it could not be had: no such
    * model, an answer that does not read, a site that does not answer.
    */
  def getModelDetail(modelId: Int): Either[String, CivitaiModelDetail] =
    try {
      val uri = uri"https://civitai.com/api/v1/models/$modelId"
        .addParam("metadata", "false")
      val response = authorized(
        basicRequest.header("Accept", "application/json").get(uri)
      ).send(backend)
      response.body match {
        case Right(json) =>
          logger.debug(s"Civitai modelDetail raw json: ${json.take(500)}")
          Try(readFromString[List[CivitaiModelDetail]](json)).toOption
            .flatMap(_.headOption)
            .map(Right(_))
            .getOrElse(
              Try(readFromString[CivitaiModelDetail](json)).toEither.left
                .map { err =>
                  logger.warn(
                    s"Civitai modelDetail decode failed: ${err.getMessage}"
                  )
                  s"Civitai's answer for model $modelId could not be read"
                }
            )
        case Left(_) if response.code.code == 404 =>
          Left(s"Civitai has no model $modelId")
        case Left(body) =>
          Left(s"Civitai answered ${response.code}: ${civitaiReason(body)}")
      }
    } catch {
      case NonFatal(error) =>
        Left(s"Civitai could not be reached: ${rootReason(error)}")
    }
}

object CivitaiClient {

  /** A search page Civitai did not give, with the reason to pass on. */
  final private case class Refused(reason: String)
      extends RuntimeException(reason)
}

/** Civitai's error body: `{"error": "…"}`, `message` on some routes. */
private case class CivitaiError(
    error: Option[String] = None,
    message: Option[String] = None
)
private given JsonValueCodec[CivitaiError] = JsonCodecMaker.make

/** What Civitai said went wrong, in its own words when its body carries them
  * (`{"error":"Image search is temporarily overloaded — please retry."}`, seen
  * 2026-09-11), else the start of the body — an HTML error page says nothing
  * worth showing.
  */
private def civitaiReason(body: String): String =
  Try(readFromString[CivitaiError](body)).toOption
    .flatMap(parsed => parsed.error.orElse(parsed.message))
    .getOrElse(
      if (body.isBlank || body.trim.startsWith("<")) "no reason given"
      else body.trim.replaceAll("\\s+", " ").take(200)
    )

private val typeCaseMap = Map(
  "CHECKPOINT" -> "Checkpoint",
  "LORA" -> "LORA",
  "TEXTUALINVERSION" -> "TextualInversion",
  "VAE" -> "VAE",
  "UPSCALER" -> "Upscaler",
  "CONTROLNET" -> "Controlnet",
  "POSES" -> "Poses"
)

private def toCivitaiType(frontendType: String): String =
  typeCaseMap.getOrElse(frontendType.toUpperCase, frontendType)
