package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** One installed highres upscaler (`specs/10-generation-time-upscaling.md`): a
  * single weight file in the flat `~/.cache/drift/upscale/` directory that
  * every session receives as `--hires-upscalers-dir`. sd-server scans that
  * directory's top level at launch — no subdirectories — and names each
  * upscaler by its file stem, so the stem doubles as `id` and is exactly what
  * `hires.upscaler` carries in a generation request.
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
    sha256: Option[String] = None,
    sizeBytes: Option[Long] = None,
    /** Set when installed from Civitai; a plain URL install leaves it empty. */
    civitaiModelId: Option[String] = None,
    createdAt: Long = 0
)
object Upscaler {
  given JsonValueCodec[Upscaler] = JsonCodecMaker.make
  given JsonValueCodec[List[Upscaler]] = JsonCodecMaker.make
  given JsonValueCodec[Option[Upscaler]] = JsonCodecMaker.make
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
)
object UpscalerDownloadJob {
  given JsonValueCodec[List[UpscalerDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[UpscalerDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val upscalersBase = endpoint.in("api")

val listUpscalers: PublicEndpoint[Unit, Unit, List[Upscaler], Any] =
  upscalersBase.get.in("upscalers").out(jsonBody[List[Upscaler]])

val installUpscaler: PublicEndpoint[
  InstallUpscalerRequest,
  Unit,
  InstallUpscalerResponse,
  Any
] =
  upscalersBase.post
    .in("upscalers" / "install")
    .in(jsonBody[InstallUpscalerRequest])
    .out(jsonBody[InstallUpscalerResponse])

val installUpscalerFromCivitai: PublicEndpoint[
  InstallUpscalerFromCivitaiRequest,
  Unit,
  InstallUpscalerResponse,
  Any
] =
  upscalersBase.post
    .in("upscalers" / "install-civitai")
    .in(jsonBody[InstallUpscalerFromCivitaiRequest])
    .out(jsonBody[InstallUpscalerResponse])

/** Deletes the entity and its weight file — a running session keeps its scan
  * until restart, so the name may linger in open capability lists.
  */
val deleteUpscaler: PublicEndpoint[String, Unit, Boolean, Any] =
  upscalersBase.delete.in("upscalers" / path[String]).out(jsonBody[Boolean])

val listUpscalerDownloads
    : PublicEndpoint[Unit, Unit, List[UpscalerDownloadJob], Any] =
  upscalersBase.get
    .in("upscaler-downloads")
    .out(jsonBody[List[UpscalerDownloadJob]])

/** Stops an upscaler's download; the job as it stands, none if unknown. */
val cancelUpscalerDownload
    : PublicEndpoint[String, Unit, Option[UpscalerDownloadJob], Any] =
  upscalersBase.post
    .in("upscaler-downloads" / path[String]("upscalerId") / "cancel")
    .out(jsonBody[Option[UpscalerDownloadJob]])
