package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** One installed upscaler (`specs/10-generation-time-upscaling.md`): a single
  * ESRGAN weight file in the flat `~/.cache/drift/upscale/` directory, named by
  * its file stem, which doubles as `id`. The post-hoc upscale
  * (`specs/15-post-hoc-resize.md`) runs it; no launch is handed the directory.
  *
  * Upscalers are global on purpose: unlike LoRAs they are not tied to an
  * architecture — a RealESRGAN model upscales pixels, whatever produced them.
  */
case class Upscaler(
    /** The file stem — the name sd-server reports and requests use. */
    id: String,
    label: String,
    /** The on-disk name in the upscale root: `<stem><extension>`, and
      * `<fileId>-<name>` for Civitai installs (the same collision-proofing as
      * the LoRA store).
      */
    fileName: String,
    /** Where the file came from — kept so an interrupted transfer can resume
      * after a drift restart.
      */
    downloadUrl: String,
    sha256: Option[String],
    sizeBytes: Option[Long],
    /** Set when installed from Civitai; a plain URL install leaves it empty. */
    civitaiModelId: Option[String],
    createdAt: Long
)
object Upscaler {
  given JsonValueCodec[Upscaler] = JsonCodecMaker.make(storedRecord)
  given JsonValueCodec[List[Upscaler]] = JsonCodecMaker.make(storedRecord)
  given JsonValueCodec[Option[Upscaler]] = JsonCodecMaker.make(storedRecord)
  given Schema[Upscaler] = Schema.derived
}

/** Installs an upscaler from a direct URL — the curated RealESRGAN list and the
  * free-form URL field both land here. `fileName` overrides the name derived
  * from the URL's last path segment.
  */
case class InstallUpscalerRequest(
    url: String,
    label: Option[String] = None,
    fileName: Option[String] = None,
    sha256: Option[String] = None
)
object InstallUpscalerRequest {
  given JsonValueCodec[InstallUpscalerRequest] = JsonCodecMaker.make
}

/** Installs weight files of one Civitai model version (type `Upscaler`) — the
  * browser flow, mirroring LoRA installs. `fileIds` narrows which files; empty
  * means all of them.
  */
case class InstallUpscalerFromCivitaiRequest(
    civitaiModelId: String,
    versionId: String,
    fileIds: List[String] = List.empty
)
object InstallUpscalerFromCivitaiRequest {
  given JsonValueCodec[InstallUpscalerFromCivitaiRequest] = JsonCodecMaker.make
}

/** The refusal-friendly answer: the installed entities on success — a Civitai
  * version can carry several — `error` naming why not.
  */
case class InstallUpscalerResponse(
    upscalers: List[Upscaler] = List.empty,
    error: Option[String] = None
)
object InstallUpscalerResponse {
  given JsonValueCodec[InstallUpscalerResponse] = JsonCodecMaker.make
}

/** One upscaler transfer. Keyed by the upscaler id; a file already on disk
  * shows as an immediately `Completed` job, so the job list doubles as the
  * "which upscalers are actually fetched" answer.
  */
case class UpscalerDownloadJob(
    upscalerId: String,
    state: DownloadState,
    downloadedBytes: Long = 0L,
    totalBytes: Option[Long] = None,
    error: Option[String] = None
) extends TransferJob
object UpscalerDownloadJob {
  given JsonValueCodec[List[UpscalerDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[UpscalerDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

val listUpscalers: PublicEndpoint[Unit, Unit, List[Upscaler], Any] =
  base.get.in("upscalers").out(jsonBody[List[Upscaler]])

val installUpscaler: PublicEndpoint[
  InstallUpscalerRequest,
  Unit,
  InstallUpscalerResponse,
  Any
] =
  base.post
    .in("upscalers" / "install")
    .in(jsonBody[InstallUpscalerRequest])
    .out(jsonBody[InstallUpscalerResponse])

val installUpscalerFromCivitai: PublicEndpoint[
  InstallUpscalerFromCivitaiRequest,
  Unit,
  InstallUpscalerResponse,
  Any
] =
  base.post
    .in("upscalers" / "install-civitai")
    .in(jsonBody[InstallUpscalerFromCivitaiRequest])
    .out(jsonBody[InstallUpscalerResponse])

/** Deletes the entity and its weight file — a running session keeps its scan
  * until restart, so the name may linger in open capability lists.
  */
val deleteUpscaler: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("upscalers" / path[String]).out(jsonBody[Boolean])

val listUpscalerDownloads
    : PublicEndpoint[Unit, Unit, List[UpscalerDownloadJob], Any] =
  base.get
    .in("upscaler-downloads")
    .out(jsonBody[List[UpscalerDownloadJob]])

/** Stops an upscaler's download; the job as it stands, none if unknown. */
val cancelUpscalerDownload
    : PublicEndpoint[String, Unit, Option[UpscalerDownloadJob], Any] =
  base.post
    .in("upscaler-downloads" / path[String]("upscalerId") / "cancel")
    .out(jsonBody[Option[UpscalerDownloadJob]])
