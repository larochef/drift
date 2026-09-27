package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

case class CivitaiModelListInfo(
    id: Int,
    name: String,
    description: Option[String] = None,
    `type`: String,
    stats: CivitaiModelListStats = CivitaiModelListStats(),
    creator: Option[CivitaiCreator] = None,
    tags: List[String] = List.empty,
    modelVersions: List[CivitaiModelVersion] = List.empty
) {

  /** The showcase Civitai ships with a search result, in the shape the three
    * browsers show their tiles (`specs/24`): the still pictures first, so a
    * tile opens on one rather than on a video, then the rest. The first is the
    * tile's preview and the others stand in when it does not load.
    */
  def examples: List[ModelExample] = {
    val shown = modelVersions.headOption.toList
      .flatMap(_.images)
      .filterNot(_.minor)
    val (pictures, rest) = shown.partition(_.`type` == "image")
    (pictures ++ rest)
      .map(image => ModelExample(image.url, image.`type` == "video"))
  }

  /** How many of them there are — what a tile's corner chip says. */
  def exampleCount: Int = examples.size
}

object CivitaiModelListInfo {

  /** Wire shape of a list entry. Every field carries a default so a response
    * that omits one still decodes, and `files`/`images` are kept because some
    * entries expose them at the root instead of under `modelVersions`.
    */
  private case class Raw(
      id: Int = 0,
      name: String = "",
      description: Option[String] = None,
      `type`: String = "",
      stats: CivitaiModelListStats = CivitaiModelListStats(),
      creator: Option[CivitaiCreator] = None,
      tags: List[String] = List.empty,
      modelVersions: List[CivitaiModelVersion] = List.empty,
      files: List[CivitaiModelFile] = List.empty,
      images: List[CivitaiImage] = List.empty
  )

  private def fromRaw(r: Raw): CivitaiModelListInfo = CivitaiModelListInfo(
    id = r.id,
    name = r.name,
    description = r.description,
    `type` = r.`type`,
    stats = r.stats,
    creator =
      r.creator.map(c => c.copy(username = c.username.filter(_.nonEmpty))),
    tags = r.tags,
    modelVersions =
      if (r.modelVersions.nonEmpty) r.modelVersions
      else if (r.files.nonEmpty || r.images.nonEmpty)
        List(
          CivitaiModelVersion(
            id = 0,
            name = "default",
            files = r.files,
            images = r.images
          )
        )
      else List.empty
  )

  private val rawCodec: JsonValueCodec[Raw] = JsonCodecMaker.make
  private val derived: JsonValueCodec[CivitaiModelListInfo] =
    JsonCodecMaker.make

  /** Decodes through the permissive wire shape, encodes the model as-is. */
  given codec: JsonValueCodec[CivitaiModelListInfo] =
    new JsonValueCodec[CivitaiModelListInfo] {
      def nullValue: CivitaiModelListInfo = null
      def decodeValue(
          in: JsonReader,
          default: CivitaiModelListInfo
      ): CivitaiModelListInfo =
        fromRaw(rawCodec.decodeValue(in, rawCodec.nullValue))
      def encodeValue(x: CivitaiModelListInfo, out: JsonWriter): Unit =
        derived.encodeValue(x, out)
    }

  given listCodec: JsonValueCodec[List[CivitaiModelListInfo]] =
    JsonCodecMaker.make
}

case class CivitaiModelListStats(
    downloadCount: Int = 0,
    thumbsUpCount: Int = 0,
    commentCount: Int = 0
)
object CivitaiModelListStats {

  /** Civitai serves a literal `null` count on some entries (seen on Flux.2
    * Klein models). jsoniter applies a field default only when the key is
    * *absent* — a null `Int` throws, and one such entry failed the decode of
    * the whole search page, which surfaced as "No models found" (bugs/20). So
    * decode through nullable counts and read null as 0.
    */
  private case class Raw(
      downloadCount: Option[Int] = None,
      thumbsUpCount: Option[Int] = None,
      commentCount: Option[Int] = None
  )
  private val rawCodec: JsonValueCodec[Raw] = JsonCodecMaker.make
  private val derived: JsonValueCodec[CivitaiModelListStats] =
    JsonCodecMaker.make

  given codec: JsonValueCodec[CivitaiModelListStats] =
    new JsonValueCodec[CivitaiModelListStats] {
      def nullValue: CivitaiModelListStats = CivitaiModelListStats()
      def decodeValue(
          in: JsonReader,
          default: CivitaiModelListStats
      ): CivitaiModelListStats = {
        val raw = rawCodec.decodeValue(in, rawCodec.nullValue)
        CivitaiModelListStats(
          downloadCount = raw.downloadCount.getOrElse(0),
          thumbsUpCount = raw.thumbsUpCount.getOrElse(0),
          commentCount = raw.commentCount.getOrElse(0)
        )
      }
      def encodeValue(x: CivitaiModelListStats, out: JsonWriter): Unit =
        derived.encodeValue(x, out)
    }
}

case class CivitaiCreator(
    username: Option[String] = None
)

case class CivitaiModelDetail(
    id: Int,
    name: String,
    `type`: String,
    description: Option[String] = None,
    creator: Option[CivitaiCreator] = None,
    tags: List[String] = List.empty,
    /** Civitai's own adult flag — what files a LoRA under sfw/ or nsfw/. */
    nsfw: Option[Boolean] = None,
    nsfwLevel: Option[Int] = None,
    stats: Option[CivitaiModelListStats] = None,
    modelVersions: List[CivitaiModelVersion] = List.empty
)
object CivitaiModelDetail {
  given Schema[CivitaiFileMetadata] = Schema.derived
  given Schema[CivitaiModelFile] = Schema.derived
  given Schema[CivitaiImage] = Schema.derived
  given Schema[CivitaiPaidAccess] = Schema.derived
  given Schema[CivitaiModelVersion] = Schema.derived
  given Schema[CivitaiCreator] = Schema.derived
  given Schema[CivitaiModelListStats] = Schema.derived
  given Schema[CivitaiModelDetail] = Schema.derived
  given JsonValueCodec[CivitaiModelDetail] = JsonCodecMaker.make
  given JsonValueCodec[List[CivitaiModelDetail]] = JsonCodecMaker.make
  given JsonValueCodec[Option[CivitaiModelDetail]] = JsonCodecMaker.make
}

/** `id` is optional on purpose: the search endpoint returns it, the
  * model-detail endpoint does not. Requiring it made every `getModelDetail`
  * decode fail, which went unnoticed because the browser seeds its file list
  * from the search payload.
  */
case class CivitaiImage(
    id: Option[Int] = None,
    url: String = "",
    nsfwLevel: Int = 0,
    width: Int = 0,
    height: Int = 0,
    hash: String = "",
    `type`: String = "",
    minor: Boolean = false,
    poi: Boolean = false,
    hasMeta: Boolean = false,
    hasPositivePrompt: Boolean = false,
    onSite: Boolean = false,
    remixOfId: Option[Int] = None
)

/** What Civitai charges for a version, when it charges (`paidAccess` on the
  * version; `null` for everything free). `permanent` is bought-forever;
  * otherwise it is early access, free again once `endsAt` passes — the version
  * also repeats that date as `earlyAccessDeadline`.
  */
case class CivitaiPaidAccess(
    permanent: Boolean = false,
    endsAt: Option[String] = None
)

case class CivitaiModelVersion(
    id: Int,
    name: String,
    baseModel: Option[String] = None,
    publishedAt: Option[String] = None,
    trainedWords: List[String] = List.empty,
    files: List[CivitaiModelFile] = List.empty,
    images: List[CivitaiImage] = List.empty,
    /** The version's notes, HTML: cleaned when it comes from the detail
      * endpoint (`specs/24`), never sent by the search one.
      */
    description: Option[String] = None,
    /** Set when this version costs money. It is per **version**, not per model:
      * a model's other versions are commonly free, which is why the browser
      * says so beside the version rather than on the tile. The model carries
      * `hasActivePaidAccess` for "one of these is paid", which does not say
      * which.
      */
    paidAccess: Option[CivitaiPaidAccess] = None
) {

  /** Whether Civitai asks for money before this version can be downloaded. */
  def isPaid: Boolean = paidAccess.isDefined

  /** The date early access ends, when it is a date — after it the version is
    * free, and drift's download of it would start working on its own.
    */
  def paidUntil: Option[String] =
    paidAccess.filterNot(_.permanent).flatMap(_.endsAt)
}

/** Civitai's per-file metadata. `fp` is the precision/quantization — "fp16",
  * "bf16", "fp8", "int8", "nf4" — and the only way to tell apart files a
  * version publishes under the *same name* (seen live: one checkpoint offered
  * as int8 and bf16, identical filename). `size` is "full" or "pruned".
  */
case class CivitaiFileMetadata(
    fp: Option[String] = None,
    size: Option[String] = None,
    format: Option[String] = None
)

case class CivitaiModelFile(
    id: Int,
    name: String,
    sizeKB: Option[Double] = None,
    primary: Option[Boolean] = None,
    /** Ready-made download URL; keyed by *version* id, which is why the version
      * cannot be reconstructed from the model id alone.
      */
    downloadUrl: Option[String] = None,
    hashes: Map[String, String] = Map.empty,
    metadata: Option[CivitaiFileMetadata] = None
) {

  /** The content SHA256 Civitai publishes, when it does. */
  def sha256: Option[String] = hashes.get("SHA256").map(_.toLowerCase)

  def isPrimary: Boolean = primary.getOrElse(true)

  /** The precision/quantization to display: the metadata `fp` when present,
    * else whatever the filename gives away (fp8, bf16, a GGUF quant like
    * q4_k_m…). Lowercase, or empty when nothing is stated anywhere.
    */
  def quantization: Option[String] =
    metadata
      .flatMap(_.fp)
      .map(_.toLowerCase)
      .orElse(
        CivitaiModelFile.QuantInFileName
          .findFirstMatchIn(name.toLowerCase)
          .map(_.group(1))
      )

  /** The full/pruned variant, when Civitai states it. */
  def sizeVariant: Option[String] =
    metadata.flatMap(_.size).map(_.toLowerCase)
}
object CivitaiModelFile {

  /** Precision markers as they appear in file names: an explicit float type,
    * int8/nf4, or a GGUF quant (`q8_0`, `q4_k_m`, `iq4_xs`…).
    */
  private val QuantInFileName =
    """(?:^|[^a-z0-9])(fp8|fp16|fp32|bf16|int8|nf4|i?q\d(?:_[a-z0-9]+)*)(?:$|[^a-z0-9])""".r
}

/** Civitai pages with an opaque cursor (`metadata.nextCursor`), not an offset —
  * `offset` is silently ignored by their API, which is why paging is
  * cursor-based end to end.
  */
case class CivitaiSearchMetadata(nextCursor: Option[CivitaiCursor] = None)

/** A page cursor as Civitai sends it: a string on some queries
  * (`"30|1781016975943"`, images by version) and a bare number on others
  * (`137607241`, images by model — probed 2026-09-11). Read as a string, the
  * number failed the whole page, so either shape becomes the text sent back.
  */
case class CivitaiCursor(value: String)
object CivitaiCursor {
  given JsonValueCodec[CivitaiCursor] = new JsonValueCodec[CivitaiCursor] {
    def nullValue: CivitaiCursor = null
    def decodeValue(in: JsonReader, default: CivitaiCursor): CivitaiCursor = {
      val quoted = in.isNextToken('"')
      in.rollbackToken()
      CivitaiCursor(
        if (quoted) in.readString(null) else in.readBigDecimal(null).toString
      )
    }
    def encodeValue(x: CivitaiCursor, out: JsonWriter): Unit =
      out.writeVal(x.value)
  }
}

case class CivitaiSearchResponse(
    items: List[CivitaiModelListInfo],
    metadata: Option[CivitaiSearchMetadata] = None
)
object CivitaiSearchResponse {
  given JsonValueCodec[CivitaiSearchResponse] = JsonCodecMaker.make
}

/** What drift's own search endpoint answers: one page plus the cursor that
  * fetches the next one (absent on the last page).
  */
case class CivitaiSearchResult(
    items: List[CivitaiModelListInfo],
    nextCursor: Option[String] = None
)
object CivitaiSearchResult {
  // Explicit schemas like CivitaiModelDetail's: auto-derivation diverges once
  // the list entries sit inside an envelope.
  given Schema[CivitaiFileMetadata] = Schema.derived
  given Schema[CivitaiModelFile] = Schema.derived
  given Schema[CivitaiImage] = Schema.derived
  given Schema[CivitaiPaidAccess] = Schema.derived
  given Schema[CivitaiModelVersion] = Schema.derived
  given Schema[CivitaiCreator] = Schema.derived
  given Schema[CivitaiModelListStats] = Schema.derived
  given Schema[CivitaiModelListInfo] = Schema.derived
  given Schema[CivitaiSearchResult] = Schema.derived
  given JsonValueCodec[CivitaiSearchResult] = JsonCodecMaker.make
}

val searchCivitaiModels: PublicEndpoint[
  (
      Option[String],
      Option[Int],
      Option[String],
      Option[String],
      Option[String],
      List[String],
      Option[Boolean]
  ),
  Unit,
  CivitaiSearchResult,
  Any
] =
  // Under `/api` like every endpoint (bugs/16) — Vite's dev proxy forwards
  // nothing else.
  base
    .in("civitai-search")
    .in(query[Option[String]]("q"))
    .in(query[Option[Int]]("limit"))
    .in(query[Option[String]]("cursor"))
    .in(query[Option[String]]("sort"))
    .in(query[Option[String]]("types"))
    .in(query[List[String]]("baseModels"))
    .in(query[Option[Boolean]]("nsfw"))
    .out(jsonBody[CivitaiSearchResult])

val getCivitaiModelDetail
    : PublicEndpoint[Int, Unit, Option[CivitaiModelDetail], Any] =
  base
    .in("civitai-model" / path[Int])
    .out(jsonBody[Option[CivitaiModelDetail]])

/** One image or video posted on Civitai with a model (`specs/24`) — only what a
  * gallery tile and its viewer need: the post's `meta` (its recipe) came back
  * empty on every image probed on 2026-09-11. All but `id` and `url` is
  * optional, since one null count once failed a whole search page (bugs/20).
  */
case class CivitaiPostedImage(
    id: Long,
    url: String,
    `type`: Option[String] = None,
    username: Option[String] = None
) {
  def isVideo: Boolean = `type`.contains("video")
}

/** Civitai's `/images` answer: a page and the cursor to the next. */
case class CivitaiImagesResponse(
    items: List[CivitaiPostedImage] = List.empty,
    metadata: Option[CivitaiSearchMetadata] = None
)
object CivitaiImagesResponse {
  given JsonValueCodec[CivitaiImagesResponse] = JsonCodecMaker.make
}

/** What drift's gallery endpoint answers: one page plus the cursor that fetches
  * the next one (absent on the last page).
  */
case class CivitaiImagePage(
    items: List[CivitaiPostedImage] = List.empty,
    nextCursor: Option[String] = None
)
object CivitaiImagePage {
  given Schema[CivitaiPostedImage] = Schema.derived
  given Schema[CivitaiImagePage] = Schema.derived
  given JsonValueCodec[CivitaiImagePage] = JsonCodecMaker.make
}

val listCivitaiImages: PublicEndpoint[
  (Int, Option[Int], Option[String], Option[String], Option[Boolean]),
  String,
  CivitaiImagePage,
  Any
] =
  base.get
    .errorOut(upstreamFailure)
    .in("civitai-images")
    .in(query[Int]("modelId"))
    .in(query[Option[Int]]("modelVersionId"))
    .in(query[Option[String]]("sort"))
    .in(query[Option[String]]("cursor"))
    .in(query[Option[Boolean]]("nsfw"))
    .out(jsonBody[CivitaiImagePage])
