package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.model.StatusCode
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Generation against a live session (`specs/08-inference-ui.md`).
  *
  * The parameter shapes here mirror the native sdcpp API of `sd-server`
  * (`examples/server/api.md` in stable-diffusion.cpp) field for field, which is
  * why every codec in this file is snake_case: the same bytes drift's frontend
  * sends to the backend are forwarded to `POST /sdcpp/v1/img_gen` and
  * `/vid_gen`, and the capabilities document decodes straight off the session
  * port. Fields sd-cpp treats as optional are `Option`s — jsoniter omits
  * `None`, and an omitted field means "use the backend default", which is the
  * server's own launch-flag-derived value.
  */

/** Self-attention-guidance parameters, nested under guidance. Drift never edits
  * these; they ride along from the session's defaults.
  */
case class SlgParameters(
    layers: List[Int] = List.empty,
    layerStart: Double = 0.01,
    layerEnd: Double = 0.2,
    scale: Double = 0.0
)
// tapir's auto-derivation does not reach through this file's nesting (maps of
// derived types, options of case classes), so every shape carries an explicit
// Schema. Schemas describe endpoints only; jsoniter alone decides the wire
// format.
object SlgParameters {
  given Schema[SlgParameters] = Schema.derived
}

case class GuidanceParameters(
    txtCfg: Double = 7.0,
    imgCfg: Option[Double] = None,
    distilledGuidance: Double = 3.5,
    slg: Option[SlgParameters] = None
)
object GuidanceParameters {
  given Schema[GuidanceParameters] = Schema.derived
}

/** One sampling stage. `wan-2.2`-style architectures have a second, high-noise
  * stage with its own copy of these.
  */
case class SampleParameters(
    /** None means the server's default; the capabilities document may also
      * report the literal string "default".
      */
    scheduler: Option[String] = None,
    sampleMethod: Option[String] = None,
    sampleSteps: Int = 20,
    eta: Option[Double] = None,
    shiftedTimestep: Int = 0,
    customSigmas: List[Double] = List.empty,
    flowShift: Option[Double] = None,
    guidance: GuidanceParameters = GuidanceParameters()
)
object SampleParameters {
  given Schema[SampleParameters] = Schema.derived

  /** Custom sigmas as typed, separated by commas or spaces: none when a piece
    * is no number, an empty list for an empty text.
    */
  def sigmasOf(text: String): Option[List[Double]] = {
    val values =
      text.split("[,\\s]+").toList.filter(_.nonEmpty).map(_.toDoubleOption)
    Option.when(values.forall(_.isDefined))(values.flatten)
  }

  def sigmasText(sigmas: List[Double]): String = sigmas.mkString(", ")

  /** The steps a list of sigmas is: one a level, the final 0 aside. */
  def sigmaSteps(sigmas: List[Double]): Int =
    if (sigmas.lastOption.contains(0.0)) sigmas.size - 1 else sigmas.size
}

/** The `hires` block of `POST /sdcpp/v1/img_gen` — highres-fix: generate at the
  * form's size, upscale, then refine with a second sampling pass
  * (`specs/10-generation-time-upscaling.md`). `upscaler` is a name from the
  * capabilities' `upscalers` list — built-ins plus the model files scanned from
  * `--hires-upscalers-dir`. `targetWidth`/`targetHeight` of 0 mean "use
  * `scale`", and `steps` 0 lets the server derive the second pass's count.
  */
case class HiresParameters(
    enabled: Boolean = false,
    upscaler: String = "Latent",
    scale: Double = 2.0,
    targetWidth: Int = 0,
    targetHeight: Int = 0,
    steps: Int = 0,
    denoisingStrength: Double = 0.7,
    customSigmas: List[Double] = List.empty,
    upscaleTileSize: Int = 128
)
object HiresParameters {
  given Schema[HiresParameters] = Schema.derived
}

/** The `vae_tiling_params` block — decode (and encode) the VAE in overlapping
  * tiles instead of one shot. The point on this hardware
  * (`specs/10-generation-time-upscaling.md`): a full-frame VAE pass at a large
  * resolution can exceed the ROCm kernel launch limits and abort the whole
  * `sd-server`; tiling keeps every kernel small.
  *
  * `tileSizeX`/`tileSizeY` are the tile size in *latent* units — the
  * latent-to-pixel ratio is the VAE's scale factor (8, 16 or 32 depending on
  * the model), so this is not a fixed pixel count — and 0 means the server
  * default (32). `relSizeX`/`relSizeY` override that when non-zero: a fraction
  * of the image dimension below 1, or a tile *count* per dimension at 1 or
  * above — model-agnostic, which is why drift's default sizes tiles this way.
  * `targetOverlap` is the overlap as a fraction of tile size, capped at 0.5.
  * `temporalTiling` and `extraTilingArgs` (e.g. `temporal_tile_frames=4`) apply
  * to video VAEs only.
  */
case class VaeTilingParameters(
    enabled: Boolean = false,
    temporalTiling: Boolean = false,
    tileSizeX: Int = 0,
    tileSizeY: Int = 0,
    targetOverlap: Double = 0.5,
    relSizeX: Double = 0.0,
    relSizeY: Double = 0.0,
    extraTilingArgs: String = ""
)
object VaeTilingParameters {
  given Schema[VaeTilingParameters] = Schema.derived
}

/** One LoRA applied to a generation — `path` is relative to the session's
  * `--lora-model-dir` (`specs/09-lora-management.md`).
  */
case class LoraSelection(
    path: String,
    multiplier: Double = 1.0,
    isHighNoise: Boolean = false
)
object LoraSelection {
  given Schema[LoraSelection] = Schema.derived
}

/** The body of `POST /sdcpp/v1/img_gen`. Image fields (`initImage`,
  * `refImages`, `maskImage`) carry base64 or data-URL images; absent means
  * plain txt2img.
  */
case class ImageGenerationParameters(
    prompt: String,
    negativePrompt: String = "",
    clipSkip: Int = -1,
    width: Int = 512,
    height: Int = 512,
    /** img2img denoising strength; ignored without `initImage`. */
    strength: Double = 0.75,
    /** -1 means the server picks a random seed. */
    seed: Long = -1,
    batchCount: Int = 1,
    initImage: Option[String] = None,
    refImages: List[String] = List.empty,
    /** `Some(false)` keeps every reference image its own size; None omits the
      * field, and the server scales each one to `width` × `height` while
      * decoding the request (sd-cpp master-892 and later honour the opt-out,
      * leejet/stable-diffusion.cpp#2011).
      */
    autoResizeRefImage: Option[Boolean] = None,
    maskImage: Option[String] = None,
    sampleParams: SampleParameters = SampleParameters(),
    lora: List[LoraSelection] = List.empty,
    /** None omits the field, which means the server's own default — hires off
      * unless the launch flags said otherwise.
      */
    hires: Option[HiresParameters] = None,
    /** None omits the field — the server's own VAE-tiling default. */
    vaeTilingParams: Option[VaeTilingParameters] = None,
    outputFormat: String = "png",
    outputCompression: Int = 100
)
object ImageGenerationParameters {
  given Schema[ImageGenerationParameters] = Schema.derived
  given JsonValueCodec[ImageGenerationParameters] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

/** An image, video or audio clip (base64 or a data URL) that the video holds at
  * `frameIndex` (a negative index counts from the end).
  */
case class VideoGuide(media: String, frameIndex: Int = 0)
object VideoGuide {
  given Schema[VideoGuide] = Schema.derived
}

/** The body of `POST /sdcpp/v1/vid_gen`. One video sequence per job — the
  * native schema has no batch count here. `initImage`/`endImage` make it
  * image-to-video where the model supports them.
  */
case class VideoGenerationParameters(
    prompt: String,
    negativePrompt: String = "",
    clipSkip: Int = -1,
    width: Int = 512,
    height: Int = 512,
    strength: Double = 0.75,
    seed: Long = -1,
    videoFrames: Int = 33,
    fps: Int = 16,
    moeBoundary: Option[Double] = None,
    vaceStrength: Option[Double] = None,
    initImage: Option[String] = None,
    endImage: Option[String] = None,
    controlFrames: List[String] = List.empty,
    /** Reference media in the order the model reads them (MiniMax H3's ref2va):
      * images, videos (with their soundtrack) and audio clips, as base64 or
      * data URLs.
      */
    references: List[String] = List.empty,
    /** Media anchored at a frame of the video (MiniMax H3's guides). */
    guides: List[VideoGuide] = List.empty,
    /** A control video (pose, depth, edges…) steering the motion through the
      * run configuration's ControlNet, with its strength and the fraction of
      * the steps it applies over.
      */
    controlVideo: Option[String] = None,
    controlStrength: Option[Double] = None,
    controlStart: Option[Double] = None,
    controlEnd: Option[Double] = None,
    /** With a control video: an image or video whose white marks what to
      * regenerate, over `sourceVideo`.
      */
    controlMask: Option[String] = None,
    sourceVideo: Option[String] = None,
    sampleParams: SampleParameters = SampleParameters(),
    highNoiseSampleParams: Option[SampleParameters] = None,
    lora: List[LoraSelection] = List.empty,
    vaeTilingParams: Option[VaeTilingParameters] = None,
    outputFormat: String = "webm",
    outputCompression: Int = 100
)
object VideoGenerationParameters {
  given Schema[VideoGenerationParameters] = Schema.derived
  given JsonValueCodec[VideoGenerationParameters] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

// ------------------------------------------------------------- capabilities

case class CapabilityModel(
    name: String = "",
    stem: String = "",
    path: String = ""
)
object CapabilityModel {
  given Schema[CapabilityModel] = Schema.derived
}

case class CapabilityLora(name: String = "", path: String = "")
object CapabilityLora {
  given Schema[CapabilityLora] = Schema.derived
}

case class CapabilityUpscaler(name: String = "")
object CapabilityUpscaler {
  given Schema[CapabilityUpscaler] = Schema.derived
}

case class CapabilityLimits(
    minWidth: Int = 64,
    maxWidth: Int = 4096,
    minHeight: Int = 64,
    maxHeight: Int = 4096,
    maxBatchCount: Int = 8,
    maxQueueSize: Int = 64
)
object CapabilityLimits {
  given Schema[CapabilityLimits] = Schema.derived
}

/** One mode's defaults from the capabilities document — the union of the
  * img_gen and vid_gen default sets, every field defaulted so a build that
  * omits some of them still decodes. These already reflect the launch flags:
  * `sd-server` seeds them from the same argv the run configuration assembled,
  * so seeding the form from here *is* seeding it from the configuration's
  * effective parameters.
  */
case class GenerationDefaults(
    prompt: String = "",
    negativePrompt: String = "",
    clipSkip: Int = -1,
    width: Int = 512,
    height: Int = 512,
    strength: Double = 0.75,
    seed: Long = -1,
    batchCount: Int = 1,
    videoFrames: Int = 33,
    fps: Int = 16,
    moeBoundary: Option[Double] = None,
    sampleParams: SampleParameters = SampleParameters(),
    highNoiseSampleParams: Option[SampleParameters] = None,
    /** img_gen only; vid_gen defaults simply omit it. */
    hires: Option[HiresParameters] = None,
    vaeTilingParams: Option[VaeTilingParameters] = None,
    outputFormat: String = "png",
    outputCompression: Int = 100
)
object GenerationDefaults {
  given Schema[GenerationDefaults] = Schema.derived
}

/** What the loaded model supports, from `GET /sdcpp/v1/capabilities` on the
  * session port. The form is built from this rather than from hardcoded lists
  * (`specs/08-inference-ui.md`): mode names are the native `img_gen` /
  * `vid_gen` strings, and `featuresByMode` carries flags such as `init_image`,
  * `ref_images`, `lora`, `cancel_queued` keyed exactly as the server names
  * them.
  */
case class SessionCapabilities(
    model: Option[CapabilityModel] = None,
    currentMode: String = "",
    supportedModes: List[String] = List.empty,
    defaultsByMode: Map[String, GenerationDefaults] = Map.empty,
    featuresByMode: Map[String, Map[String, Boolean]] = Map.empty,
    outputFormatsByMode: Map[String, List[String]] = Map.empty,
    samplers: List[String] = List.empty,
    schedulers: List[String] = List.empty,
    loras: List[CapabilityLora] = List.empty,
    upscalers: List[CapabilityUpscaler] = List.empty,
    limits: CapabilityLimits = CapabilityLimits()
)
object SessionCapabilities {
  given defaultsByModeSchema: Schema[Map[String, GenerationDefaults]] =
    Schema.schemaForMap
  given featuresByModeSchema: Schema[Map[String, Map[String, Boolean]]] =
    Schema.schemaForMap(using Schema.schemaForMap[Boolean])
  given outputFormatsByModeSchema: Schema[Map[String, List[String]]] =
    Schema.schemaForMap
  given Schema[SessionCapabilities] = Schema.derived
  given JsonValueCodec[SessionCapabilities] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
  given JsonValueCodec[Option[SessionCapabilities]] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

// -------------------------------------------------------------- generations

/** Mirrors the native job lifecycle: queued → generating → completed | failed
  * | cancelled.
  */
enum GenerationStatus derives CanEqual {
  case Queued, Generating, Completed, Failed, Cancelled

  def isActive: Boolean = this match {
    case Queued | Generating => true
    case _                   => false
  }
}
object GenerationStatus {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[GenerationStatus] =
    Schema.derivedEnumeration[GenerationStatus].defaultStringBased
}

/** One file a completed generation left on disk, under
  * `~/.local/share/drift/outputs/<date>/`. `url` is what the frontend loads;
  * `fileName` is relative to the date directory named by `date`.
  */
case class GenerationOutput(
    date: String,
    fileName: String,
    url: String,
    mimeType: String,
    format: String,
    index: Int = 0,
    fps: Option[Int] = None,
    frameCount: Option[Int] = None
)
object GenerationOutput {
  given Schema[GenerationOutput] = Schema.derived
}

/** Where a derived gallery entry came from (`specs/15-post-hoc-resize.md`): the
  * parent output it was made from and the operation. The parent is untouched;
  * the gallery shows the chain and offers original ↔ result.
  */
case class Derivation(
    parentId: String,
    parentDate: String,
    parentFileName: String,
    /** "upscale" (ESRGAN), "pid" (pixel diffusion decoder), "redraw" (tiled
      * img2img), "edit" (an instruction, `specs/39-seamless-edit.md`),
      * "seedvr2" (a SeedVR2 upscale of a picture or a video, `repeats` its
      * scale), or "resize" in entries made before resize was removed.
      */
    operation: String,
    upscalerId: Option[String] = None,
    repeats: Option[Int] = None,
    width: Option[Int] = None,
    height: Option[Int] = None,
    fit: Option[String] = None,
    /** The PiD or redraw run configuration and request that made the entry. */
    configurationId: Option[String] = None,
    prompt: Option[String] = None,
    steps: Option[Int] = None,
    seed: Option[Long] = None,
    /** A redraw's denoising strength and the instructions appended to its
      * prompt (`specs/27-redraw.md`); an edit's instruction (`specs/39`).
      */
    strength: Option[Double] = None,
    instructions: Option[String] = None,
    /** The part of the parent a redraw repainted or an edit changed
      * (`specs/27-redraw.md`, `specs/39`); none means the whole image.
      */
    region: Option[ImageRegion] = None,
    /** Whether a redraw's tiles had each their own prompt and strength, read
      * off the picture by the assistant (`specs/52-auto-redraw.md`).
      */
    planned: Option[Boolean] = None
)
object Derivation {
  given Schema[Derivation] = Schema.derived
}

/** An input that was picked from the gallery rather than from the disk
  * (`specs/50-inputs-from-the-gallery.md`): which slot of the request it filled
  * — "init", "end", "mask", "ref0", "reference1", "control"… the names the
  * persisted inputs carry — and the output it was. The generation keeps its own
  * copy of the input all the same; this is the way back to the entry.
  */
case class InputSource(
    slot: String,
    generationId: String,
    outputIndex: Int,
    date: String,
    fileName: String
)
object InputSource {
  given Schema[InputSource] = Schema.derived
}

/** One generation job, drift's view: the native job's lifecycle plus what drift
  * adds — which session and run configuration produced it, the full request,
  * and the files persisted on completion. Written as the sidecar JSON next to
  * the outputs, so history (`specs/08-inference-ui.md`) can reload it.
  *
  * A submission that is *refused* — session unknown, not ready, queue full —
  * answers with a `Generation` already `Failed`, the reason in `error`,
  * mirroring how launches report refusals.
  */
case class Generation(
    id: String,
    sessionId: String,
    runConfigurationId: String,
    /** The native job kind, "img_gen" or "vid_gen" — or, for an entry made from
      * another's output rather than generated, "upscale" or "resize" — or
      * "import", an image brought in from outside drift.
      */
    kind: String,
    status: GenerationStatus,
    queuePosition: Int = 0,
    submittedAt: Long,
    startedAt: Option[Long] = None,
    completedAt: Option[Long] = None,
    imageParameters: Option[ImageGenerationParameters] = None,
    videoParameters: Option[VideoGenerationParameters] = None,
    outputs: List[GenerationOutput] = List.empty,
    error: Option[String] = None,
    /** Set on entries derived from another's output post hoc. */
    derivation: Option[Derivation] = None,
    /** The name the file had where it came from, on an image imported into the
      * gallery (`specs/30-gallery-ergonomics-and-image-import.md`); none on
      * anything drift made.
      */
    importedFileName: Option[String],
    /** The gallery entries its inputs were picked from (`specs/50`); none for
      * inputs browsed from the disk.
      */
    inputSources: List[InputSource],
    /** The project and version this generation belongs to
      * (`specs/19-projects-and-prompt-versions.md`); derived entries inherit
      * their parent's.
      */
    projectId: Option[String] = None,
    promptVersionId: Option[String] = None,
    /** Free play (`specs/22-free-play-and-scratch-generations.md`): the outputs
      * live under `outputs/scratch/`, no sidecar was written, and nothing folds
      * it into the gallery or a project. Keeping it clears this.
      */
    scratch: Boolean = false
) {

  /** The seed output `index` of an image batch was made with: sd-cpp and the
    * drift runner both give image `b` of a batch `seed + b`, so the recorded
    * seed is only the first image's. None when the request left it random.
    */
  def seedOf(index: Int): Option[Long] =
    imageParameters.map(_.seed).filter(_ >= 0).map(_ + index)

  /** This generation narrowed to one output of its batch — that image's seed, a
    * batch of one — which is what reusing that image reproduces.
    */
  def ofOutput(index: Int): Generation =
    imageParameters match {
      case Some(p) if p.batchCount > 1 =>
        copy(imageParameters =
          Some(p.copy(seed = seedOf(index).getOrElse(p.seed), batchCount = 1))
        )
      case _ => this
    }
}
object Generation {
  given Schema[Generation] = Schema.derived
  given JsonValueCodec[Generation] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
  given JsonValueCodec[List[Generation]] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
  given JsonValueCodec[Option[Generation]] = JsonCodecMaker.make(
    CodecMakerConfig
      .withDiscriminatorFieldName(None)
      .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

// ---------------------------------------------------------------- endpoints

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val generationBase = endpoint.in("api")

/** The capabilities of a session's loaded model, proxied from the session port.
  * Empty when the session is unknown, not yet ready, or unreachable.
  */
val getSessionCapabilities
    : PublicEndpoint[String, Unit, Option[SessionCapabilities], Any] =
  generationBase.get
    .in("sessions" / path[String] / "capabilities")
    .out(jsonBody[Option[SessionCapabilities]])

/** The project context of a submission. It rides as query parameters, so the
  * body stays the native request sd-server takes. "Do not keep this" is
  * deliberately not part of it: that is not a project fact
  * (`specs/22-free-play-and-scratch-generations.md`).
  */
private val submitContextInput =
  query[Option[String]]("project")
    .and(query[Option[String]]("version"))
    .and(query[Option[String]]("origin"))
    .mapTo[SubmitContext]

/** Submits an image generation to the session's `sd-server`. Answers with the
  * queued generation, or a `Failed` one naming why it was refused.
  */
val submitImageGeneration: PublicEndpoint[
  (String, SubmitContext, Option[Boolean], ImageGenerationParameters),
  Unit,
  Generation,
  Any
] =
  generationBase.post
    .in("sessions" / path[String] / "generations" / "image")
    .in(submitContextInput)
    .in(query[Option[Boolean]]("scratch"))
    .in(LargeJsonBody[ImageGenerationParameters])
    .out(jsonBody[Generation])

/** Submits a video generation to the session's `sd-server`, versioned inside a
  * project like an image (`specs/31-project-kinds.md`).
  */
val submitVideoGeneration: PublicEndpoint[
  (String, SubmitContext, Option[Boolean], VideoGenerationParameters),
  Unit,
  Generation,
  Any
] =
  generationBase.post
    .in("sessions" / path[String] / "generations" / "video")
    .in(submitContextInput)
    .in(query[Option[Boolean]]("scratch"))
    .in(LargeJsonBody[VideoGenerationParameters])
    .out(jsonBody[Generation])

/** This session's generations, oldest first. The backend polls the native job
  * behind each active one, so polling this reflects queue position, progress
  * and completion without the frontend ever touching the session port.
  */
val listGenerations: PublicEndpoint[String, Unit, List[Generation], Any] =
  generationBase.get
    .in("sessions" / path[String] / "generations")
    .out(jsonBody[List[Generation]])

/** Cancels a queued or generating job. Answers with the generation's current
  * state — cancellation is asynchronous, so it may still show as active until
  * the next poll — or nothing for an unknown id.
  *
  * `force` is the way out when the build or the model cannot interrupt a
  * generation at all (`features_by_mode` says `cancel_generating: false`, or
  * the server answers 409): the generation is marked cancelled and its
  * sd-server is killed and launched again on the same configuration. The model
  * reloads, which costs what the launch cost — and beats killing drift itself,
  * which is the only other way out.
  */
val cancelGeneration
    : PublicEndpoint[(String, Boolean), Unit, Option[Generation], Any] =
  generationBase.post
    .in("generations" / path[String] / "cancel")
    .in(query[Boolean]("force").default(false))
    .out(jsonBody[Option[Generation]])

/** Serves one persisted output file. The date and file name come from a
  * `GenerationOutput`; anything else is a 404.
  */
/** A persisted output scaled down for the screen, its longest side at most
  * `side` px — what the gallery and the detail view show. An 8192² upscale is
  * 150 MB on disk and a quarter of a gigabyte decoded; handing that to an
  * `<img>` in a 700 px box froze the detail view for seconds at a time
  * (François, 2026-09-19). Scaled copies are cached, so the cost is paid once.
  *
  * An image already within `side` is served as it is. Videos are not scaled:
  * ask for the file itself.
  */
val getOutputPreview: PublicEndpoint[
  (String, String, Option[Int]),
  Unit,
  (Array[Byte], String),
  Any
] =
  generationBase.get
    .in("outputs" / path[String] / path[String] / "preview")
    .in(query[Option[Int]]("side"))
    .errorOut(statusCode(StatusCode.NotFound))
    .out(byteArrayBody)
    .out(header[String]("Content-Type"))

/** What a persisted image really measures. `Tiling` stays free of tapir and
  * jsoniter, so the answer has a shape of its own.
  */
case class OutputSize(width: Int, height: Int)
object OutputSize {
  given JsonValueCodec[OutputSize] = JsonCodecMaker.make
  given Schema[OutputSize] = Schema.derived
}

/** What a persisted image really measures, read from its header alone. The
  * preview the page shows is smaller, and a selection made on it has to be
  * recorded in the pixels of the file it will be repainted from
  * (`specs/27-redraw.md`).
  */
val getOutputSize: PublicEndpoint[(String, String), Unit, OutputSize, Any] =
  generationBase.get
    .in("outputs" / path[String] / path[String] / "size")
    .errorOut(statusCode(StatusCode.NotFound))
    .out(jsonBody[OutputSize])

/** A persisted output, streamed from disk. A `Range` header (one range of
  * bytes) gets `206` with just those bytes and their `Content-Range`, which a
  * video player needs to reach a webm's index at the end and a large image
  * (16k² is near 1 GB) to be fetched in parts; without one, the whole file. The
  * errors: `404`, or `416` with a `Content-Range` of the size alone for a range
  * past the end.
  */
val getOutputFile: PublicEndpoint[
  (String, String, Option[String]),
  (StatusCode, Option[String]),
  (StatusCode, String, Option[String], Long, FileRange),
  Any
] =
  generationBase.get
    .in("outputs" / path[String] / path[String])
    .in(header[Option[String]]("Range"))
    .errorOut(statusCode.and(header[Option[String]]("Content-Range")))
    .out(statusCode)
    .out(header[String]("Content-Type"))
    .out(header[Option[String]]("Content-Range"))
    .out(header("Accept-Ranges", "bytes"))
    // explicit: without it, a range that ends before the file does leaves the
    // client waiting for more
    .out(header[Long]("Content-Length"))
    .out(fileRangeBody)
