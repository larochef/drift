package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** The five states of `specs/05-model-cache-and-downloads.md`, mirroring the
  * job shape sd-cpp's own server uses so the frontend keeps one mental model
  * for both kinds of long-running work.
  */
enum DownloadState derives CanEqual {
  case Queued, Downloading, Completed, Failed, Cancelled

  def isActive: Boolean = this match {
    case Queued | Downloading => true
    case _                    => false
  }
}

/** One download, keyed by the model it fetches — a model has at most one
  * download in flight, so the model id is the job id.
  */
case class DownloadJob(
    modelId: String,
    state: DownloadState,
    downloadedBytes: Long = 0L,
    totalBytes: Option[Long] = None,
    error: Option[String] = None,
    startedAt: Option[Long] = None,
    completedAt: Option[Long] = None
)
object DownloadJob {
  // No discriminator, so `DownloadState` encodes as a plain string. The config
  // must be inline -- the macro needs a constant expression.
  given JsonValueCodec[DownloadJob] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[DownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** The sidecar written next to every Civitai download: what the source knew at
  * fetch time, so the cache is self-describing and the UI has something to
  * show. See "Metadata sidecars" in `specs/05-model-cache-and-downloads.md`.
  */
case class ModelMetadata(
    source: String,
    modelId: String,
    versionId: String,
    fileId: String,
    name: String,
    versionName: String,
    creator: Option[String] = None,
    description: Option[String] = None,
    tags: List[String] = List.empty,
    baseModel: Option[String] = None,
    trainedWords: List[String] = List.empty,
    publishedAt: Option[String] = None,
    nsfwLevel: Option[Int] = None,
    downloadCount: Option[Long] = None,
    thumbsUpCount: Option[Long] = None,
    sha256: Option[String] = None,
    sizeBytes: Option[Long] = None,
    previews: List[String] = List.empty,
    fetchedAt: String = ""
)
object ModelMetadata {
  given JsonValueCodec[ModelMetadata] = JsonCodecMaker.make
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val downloadsBase = endpoint.in("api").in("downloads")

val listDownloads: PublicEndpoint[Unit, Unit, List[DownloadJob], Any] =
  downloadsBase.get.out(jsonBody[List[DownloadJob]])

/** Starts a download for a registered model, or returns the job already in
  * flight for it. A model that cannot be downloaded (unknown, `Local`, or
  * already cached) answers with a terminal job saying why.
  */
val startDownload: PublicEndpoint[String, Unit, DownloadJob, Any] =
  downloadsBase.post.in(path[String]("modelId")).out(jsonBody[DownloadJob])

val cancelDownload: PublicEndpoint[String, Unit, DownloadJob, Any] =
  downloadsBase.post
    .in(path[String]("modelId") / "cancel")
    .out(jsonBody[DownloadJob])
