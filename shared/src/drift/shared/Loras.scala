package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Which diffusion stage a LoRA file applies to
  * (`specs/09-lora-management.md`). `wan-2.2` splits its diffusion model into a
  * high-noise and a low-noise stage and its LoRAs ship as matching pairs;
  * everything else is `General`.
  */
enum LoraFileStage derives CanEqual {
  case General, LowNoise, HighNoise
}
object LoraFileStage {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[LoraFileStage] =
    Schema.derivedEnumeration[LoraFileStage].defaultStringBased
}

/** One weight file of a LoRA (`specs/33-lora-sources.md`).
  */
case class LoraFile(
    /** The file's name inside the LoRA's folder: what sd-cpp loads and what
      * keys its transfer. A Civitai file is stored as `<fileId>-<filename>`
      * (the model cache's collision-proofing), any other under its own name.
      */
    fileName: String,
    /** Where the file is fetched from: a Civitai version's file, a HuggingFace
      * repository's file, or a file on the drift host, copied in. Stored so an
      * interrupted transfer is re-queued after a restart without asking anyone.
      */
    source: ModelSource,
    stage: LoraFileStage = LoraFileStage.General,
    sizeBytes: Option[Long] = None,
    /** The content hash the source publishes (Civitai's, HuggingFace's LFS
      * hash), when it does — verified after every fetch, resumed ones included.
      */
    sha256: Option[String] = None
) {

  /** The source's own name for the file, without the Civitai prefix or the
    * repository's folders.
    */
  def displayName: String = source match {
    case Civitai(_, _, _, filename) => filename
    case HuggingFace(_, filename)   => filename.split('/').last
    case ModelScope(_, filename)    => filename.split('/').last
    case Local(path)                => path.split('/').last
  }

  /** Where the file comes from, in a few words. */
  def origin: String = source match {
    case Civitai(modelId, _, _, _)   => s"Civitai model $modelId"
    case HuggingFace(repo, filename) => s"HuggingFace $repo/$filename"
    case ModelScope(repo, filename)  => s"ModelScope $repo/$filename"
    case Local(path)                 => s"copied from $path"
  }
}
object LoraFile {
  given Schema[LoraFile] = Schema.derived
}

/** The LoRA files llama-server can load (`specs/35-assistant-loras.md`): it
  * takes GGUF adapters only, and names each on its command line.
  */
object LoraAdapters {
  def isAdapterFile(fileName: String): Boolean =
    fileName.toLowerCase.endsWith(".gguf")

  /** Every GGUF file of the architecture's installed LoRAs, relative to the
    * LoRA root.
    */
  def relativePaths(architectureId: String, loras: List[Lora]): List[String] =
    loras
      .filter(_.architectureId == architectureId)
      .flatMap(lora =>
        lora.files
          .filter(file => isAdapterFile(file.fileName))
          .map(lora.storagePathOf)
      )
}

/** A LoRA a run configuration applies by default wherever it runs
  * (`specs/28-configuration-loras.md`), at the configuration's own strength.
  */
case class ConfiguredLora(loraId: String, strength: Double)
object ConfiguredLora {
  given JsonValueCodec[ConfiguredLora] = JsonCodecMaker.make
  given Schema[ConfiguredLora] = Schema.derived
}

/** The sampling a LoRA was made for (`specs/49-lora-sampling-settings.md`): a
  * turbo LoRA's steps, CFG and flow shift, typed by the user from its author's
  * page. One more layer of defaults over the session's, each field overriding
  * only when set — defaults, never rules: the form shows them and the user may
  * change every one.
  */
case class LoraSampling(
    /** The count it was made for: more is fine, fewer break the image. */
    steps: Option[Int] = None,
    cfg: Option[Double] = None,
    flowShift: Option[Double] = None,
    /** Its exact noise levels, which replace the schedule and are their own
      * step count. Not for a two-expert model.
      */
    sigmas: List[Double] = List.empty,
    sampler: Option[String] = None,
    scheduler: Option[String] = None,
    distilledGuidance: Option[Double] = None,
    /** A pair's high-noise expert, on a two-expert model (wan 2.2). */
    highNoiseSteps: Option[Int] = None,
    highNoiseCfg: Option[Double] = None
) {
  def isEmpty: Boolean = this == LoraSampling()
  def nonEmpty: Boolean = !isEmpty

  /** The form fields it sets, by the names the generation form knows them. */
  def fields: Set[String] =
    List(
      steps.map(_ => "steps"),
      cfg.map(_ => "cfg"),
      flowShift.map(_ => "flowShift"),
      Option.when(sigmas.nonEmpty)("sigmas"),
      sampler.map(_ => "sampler"),
      scheduler.map(_ => "scheduler"),
      distilledGuidance.map(_ => "distilledGuidance"),
      highNoiseSteps.map(_ => "highNoiseSteps"),
      highNoiseCfg.map(_ => "highNoiseCfg")
    ).flatten.toSet

  /** What it sets, a phrase a setting: "6 steps", "CFG 1.0", "flow shift 3.0".
    */
  def summary: List[String] =
    List(
      steps.map(count => s"$count steps"),
      cfg.map(scale => s"CFG $scale"),
      flowShift.map(shift => s"flow shift $shift"),
      Option.when(sigmas.nonEmpty)(s"${sigmas.size} sigmas"),
      sampler,
      scheduler,
      distilledGuidance.map(scale => s"distilled guidance $scale"),
      highNoiseSteps.map(count => s"$count high-noise steps"),
      highNoiseCfg.map(scale => s"high-noise CFG $scale")
    ).flatten

  /** These settings over `base`, a later layer's over an earlier one's. */
  def over(base: LoraSampling): LoraSampling =
    LoraSampling(
      steps = steps.orElse(base.steps),
      cfg = cfg.orElse(base.cfg),
      flowShift = flowShift.orElse(base.flowShift),
      sigmas = if (sigmas.nonEmpty) sigmas else base.sigmas,
      sampler = sampler.orElse(base.sampler),
      scheduler = scheduler.orElse(base.scheduler),
      distilledGuidance = distilledGuidance.orElse(base.distilledGuidance),
      highNoiseSteps = highNoiseSteps.orElse(base.highNoiseSteps),
      highNoiseCfg = highNoiseCfg.orElse(base.highNoiseCfg)
    )

  /** A session's sampling defaults with these settings over them. */
  def over(base: SampleParameters): SampleParameters =
    base.copy(
      sampleSteps = steps.getOrElse(base.sampleSteps),
      flowShift = flowShift.orElse(base.flowShift),
      customSigmas = if (sigmas.nonEmpty) sigmas else base.customSigmas,
      sampleMethod = sampler.orElse(base.sampleMethod),
      scheduler = scheduler.orElse(base.scheduler),
      guidance = base.guidance.copy(
        txtCfg = cfg.getOrElse(base.guidance.txtCfg),
        distilledGuidance =
          distilledGuidance.getOrElse(base.guidance.distilledGuidance)
      )
    )

  /** The high-noise expert's defaults with these settings over them. */
  def overHighNoise(base: SampleParameters): SampleParameters =
    base.copy(
      sampleSteps = highNoiseSteps.getOrElse(base.sampleSteps),
      flowShift = flowShift.orElse(base.flowShift),
      guidance = base.guidance.copy(
        txtCfg = highNoiseCfg.getOrElse(base.guidance.txtCfg)
      )
    )
}
object LoraSampling {
  given JsonValueCodec[LoraSampling] = JsonCodecMaker.make
  given Schema[LoraSampling] = Schema.derived

  /** The settings of `loras` together, the last one's over the others'. */
  def of(loras: List[Lora]): LoraSampling =
    loras.foldLeft(LoraSampling())((merged, lora) => lora.sampling.over(merged))
}

/** One LoRA the user has installed (`specs/09-lora-management.md`), or one
  * drift offers to install (`specs/33-lora-sources.md`: the catalog is a list
  * of the same entity, never saved until installed).
  *
  * This entity lives in `~/.config/drift/loras/` and carries the user's tuning
  * — `defaultStrength`, the sfw/nsfw placement, corrected stages — precisely
  * because `~/.cache` must stay safe to delete. The Civitai metadata sidecar in
  * the cache folder is regenerable decoration; this is the authority.
  *
  * `id` doubles as the cache folder name (`<architectureId>-<slug>-<source>`,
  * the source part being the Civitai model id, the HuggingFace repository's
  * owner or `local`). The architecture is part of the identity: one Civitai
  * model often publishes versions for different base models (a wan version and
  * an ltx version, say), and installing each from its own architecture's
  * browser must yield two independent entities — own files, own tuning — not
  * one entity claiming both file sets.
  */
case class Lora(
    id: String,
    architectureId: String,
    label: String,
    nsfw: Boolean = false,
    /** Seeds the picker's strength input. Some LoRAs need values that are easy
      * to forget; this is where they are remembered.
      */
    defaultStrength: Double = 1.0,
    triggerWords: List[String] = List.empty,
    tags: List[String] = List.empty,
    description: Option[String] = None,
    files: List[LoraFile] = List.empty,
    createdAt: Long = 0,
    /** The sampling it was made for (`specs/49-lora-sampling-settings.md`);
      * empty for most LoRAs.
      */
    sampling: LoraSampling
) {

  /** The cache folder, relative to the LoRA root. */
  def folderRelativePath: String =
    s"$architectureId/${if (nsfw) "nsfw" else "sfw"}/$id"

  /** The file, relative to the LoRA root — where drift stores it. */
  def storagePathOf(file: LoraFile): String =
    s"$folderRelativePath/${file.fileName}"

  /** What `lora[].path` carries: the file, relative to `--lora-model-dir`,
    * which is the architecture's own folder (`specs/09`).
    */
  def requestPathOf(file: LoraFile): String =
    s"${if (nsfw) "nsfw" else "sfw"}/$id/${file.fileName}"

  /** The haystack the picker's search field matches against. */
  def searchText: String =
    (label :: triggerWords ::: tags ::: description.toList :::
      files.map(_.origin))
      .mkString(" ")
      .toLowerCase

  /** What applying this LoRA at `strength` sends as `lora[]`: every file, so a
    * Wan 2.2 pair sends both halves, the high-noise one flagged.
    */
  def selections(strength: Double): List[LoraSelection] =
    files.map(file =>
      LoraSelection(
        path = requestPathOf(file),
        multiplier = strength,
        isHighNoise = file.stage == LoraFileStage.HighNoise
      )
    )
}
object Lora {
  // No discriminator: `LoraFileStage` has only singleton cases, so it encodes
  // as a plain string. The config must be inline — the macro needs a constant
  // expression.
  given JsonValueCodec[Lora] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[Lora]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[Lora]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given Schema[Lora] = Schema.derived
}

/** One file of a Civitai model, told by the version that publishes it: the
  * download URL is built from both ids.
  */
case class CivitaiFileRef(versionId: String, fileId: String)
object CivitaiFileRef {
  given Schema[CivitaiFileRef] = Schema.derived
}

/** Where an install takes its files from (`specs/33-lora-sources.md`). Every
  * site's case carries a *list*: what is ticked in one model or repository is
  * installed in one go, which is what makes a wan 2.2 pair one LoRA.
  */
sealed trait LoraInstallSource
object LoraInstallSource {

  /** Files ticked in one Civitai model, from any of its versions. */
  case class CivitaiFiles(modelId: String, files: List[CivitaiFileRef])
      extends LoraInstallSource

  /** Files ticked in one HuggingFace repository. */
  case class HuggingFaceFiles(repo: String, filenames: List[String])
      extends LoraInstallSource

  /** Files ticked in one ModelScope repository (`specs/37-modelscope.md`). */
  case class ModelScopeFiles(repo: String, filenames: List[String])
      extends LoraInstallSource

  /** Weight files on the drift host, copied into the LoRA store: sd-cpp only
    * sees what is under its `--lora-model-dir`.
    */
  case class LocalFiles(paths: List[String]) extends LoraInstallSource

  /** One of the LoRAs drift offers, by id (`reference/loras.json`). */
  case class Catalog(loraId: String) extends LoraInstallSource

  given Schema[LoraInstallSource] = Schema.derived
}

/** Where an install's files land (`specs/33-lora-sources.md`), chosen beside
  * the install button: several files travel together by default, one file makes
  * a LoRA of its own, and either can join a LoRA already installed.
  */
sealed trait LoraGrouping derives CanEqual
object LoraGrouping {

  /** One LoRA holding all of them — a new one, or the one an identical install
    * already made, which is what makes a re-install fetch only what is missing.
    */
  case object Together extends LoraGrouping

  /** One LoRA per file, so two versions of the same LoRA can sit side by side
    * in the picker and be compared (François, 2026-09-18).
    */
  case object Separate extends LoraGrouping

  /** Beside the files of a LoRA already installed. */
  case class Into(loraId: String) extends LoraGrouping

  given Schema[LoraGrouping] = Schema.derived
}

/** Installs LoRAs for one architecture. Installing what is already installed
  * keeps the entity and its tuning and fetches whatever is missing.
  */
case class InstallLoraRequest(
    architectureId: String,
    source: LoraInstallSource,
    grouping: LoraGrouping = LoraGrouping.Together
)
object InstallLoraRequest {
  given JsonValueCodec[InstallLoraRequest] = JsonCodecMaker.make
  given Schema[InstallLoraRequest] = Schema.derived
}

/** Makes one LoRA of two (`specs/33-lora-sources.md`): a wan 2.2 pair whose
  * halves were published apart — two ModelScope repositories, one for the high
  * noise and one for the low (François, 2026-09-18) — installs as two LoRAs,
  * each holding one stage. Pairing moves `otherId`'s files beside `loraId`'s,
  * under the name the user gives the result, and the other entity goes.
  */
case class PairLoraRequest(loraId: String, otherId: String, label: String)
object PairLoraRequest {
  given JsonValueCodec[PairLoraRequest] = JsonCodecMaker.make
}

/** Adopts an on-disk LoRA folder that no entity references — orphans left
  * behind by schema breaks or copied in by hand — rebuilding the entity for the
  * given architecture from the folder's sidecar and weight files, and moving
  * the folder under that architecture. `path` is the folder, or any file inside
  * it, as the cache view lists it.
  */
case class AdoptLoraRequest(path: String, architectureId: String)
object AdoptLoraRequest {
  given JsonValueCodec[AdoptLoraRequest] = JsonCodecMaker.make
}

/** The refusal-friendly answer: `lora` on success, `error` naming why not. */
case class InstallLoraResponse(
    lora: Option[Lora] = None,
    error: Option[String] = None
)
object InstallLoraResponse {
  given JsonValueCodec[InstallLoraResponse] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** One LoRA file transfer. Same lifecycle as model downloads, keyed by (lora,
  * file) because a wan pair downloads as two transfers.
  */
case class LoraDownloadJob(
    loraId: String,
    fileName: String,
    state: DownloadState,
    downloadedBytes: Long = 0L,
    totalBytes: Option[Long] = None,
    error: Option[String] = None
)
object LoraDownloadJob {
  given JsonValueCodec[List[LoraDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[LoraDownloadJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val lorasBase = endpoint.in("api")

val listLoras: PublicEndpoint[Unit, Unit, List[Lora], Any] =
  lorasBase.get.in("loras").out(jsonBody[List[Lora]])

/** The LoRAs drift offers to install (`specs/33-lora-sources.md`), read from
  * `reference/loras.json` — installed ones included, the UI tells them apart by
  * id.
  */
val listLoraCatalog: PublicEndpoint[Unit, Unit, List[Lora], Any] =
  lorasBase.get.in("loras" / "catalog").out(jsonBody[List[Lora]])

val installLora
    : PublicEndpoint[InstallLoraRequest, Unit, InstallLoraResponse, Any] =
  lorasBase.post
    .in("loras" / "install")
    .in(jsonBody[InstallLoraRequest])
    .out(jsonBody[InstallLoraResponse])

val adoptLora
    : PublicEndpoint[AdoptLoraRequest, Unit, InstallLoraResponse, Any] =
  lorasBase.post
    .in("loras" / "adopt")
    .in(jsonBody[AdoptLoraRequest])
    .out(jsonBody[InstallLoraResponse])

/** Saves the user-editable fields. Flipping `nsfw` moves the cache folder, so
  * the relative paths a running session was launched with go stale — the
  * picker's visibility check surfaces that.
  */
val updateLora: PublicEndpoint[(String, Lora), Unit, Option[Lora], Any] =
  lorasBase.put
    .in("loras" / path[String])
    .in(jsonBody[Lora])
    .out(jsonBody[Option[Lora]])

/** Deletes the entity and its cache folder — the folder is regenerable, and
  * "remove this LoRA" means the gigabytes too.
  */
val deleteLora: PublicEndpoint[String, Unit, Boolean, Any] =
  lorasBase.delete.in("loras" / path[String]).out(jsonBody[Boolean])

/** The two LoRAs become one, holding both stages. */
val pairLoras: PublicEndpoint[PairLoraRequest, Unit, InstallLoraResponse, Any] =
  lorasBase.post
    .in("loras" / "pair")
    .in(jsonBody[PairLoraRequest])
    .out(jsonBody[InstallLoraResponse])

val listLoraDownloads: PublicEndpoint[Unit, Unit, List[LoraDownloadJob], Any] =
  lorasBase.get.in("lora-downloads").out(jsonBody[List[LoraDownloadJob]])

/** Stops one LoRA file's transfer; the job as it stands, none if unknown. */
val cancelLoraDownload
    : PublicEndpoint[(String, String), Unit, Option[LoraDownloadJob], Any] =
  lorasBase.post
    .in("lora-downloads" / path[String]("loraId") / path[String]("fileName"))
    .in("cancel")
    .out(jsonBody[Option[LoraDownloadJob]])
