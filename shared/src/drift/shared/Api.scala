package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

// jsoniter derives a codec per concrete type rather than composing them, so a
// companion declares exactly the shapes -- `T`, `List[T]`, `Option[T]` -- that
// the endpoints and `StorageService` round-trip. A type that only ever appears
// inside another is inlined into its parent's codec and needs none of its own.

// `delete*` answers with a bare JSON boolean, and primitives have no codec in
// jsoniter's implicit scope either.
given JsonValueCodec[Boolean] = JsonCodecMaker.make
given JsonValueCodec[List[String]] = JsonCodecMaker.make

/** What a model does with a reference image (`specs/27-redraw.md`).
  *
  *   - `Unused`: sd-cpp has no reference preset for the family, so a reference
  *     is bytes and latency that change nothing.
  *   - `Context`: the reference informs the result — identity, colour, lighting
  *     — without dictating what is drawn where.
  *   - `Edit`: the reference *is* the subject; the preset passes it to the
  *     diffusion model, which renders that image's layout. Handing such a model
  *     the whole picture beside a tile paints the whole picture into every
  *     tile.
  */
enum ReferenceImageUse derives CanEqual {
  case Unused, Context, Edit
}
object ReferenceImageUse {
  given Schema[ReferenceImageUse] =
    Schema.derivedEnumeration[ReferenceImageUse].defaultStringBased
}

case class Architecture(
    id: String,
    label: String,
    /** Which runner this shape is for
      * (`specs/18-assistant-models-and-sessions.md`): sd-cpp for image and
      * video families, llama.cpp for chat families. The checkpoint slots,
      * parameters and run configurations mean the same thing for both; only the
      * launched executable, its listen flags and its readiness probe differ.
      */
    tool: RuntimeTool,
    checkpoints: List[CheckpointRef],
    defaultParameters: Map[String, String],
    civitaiBaseModels: List[String] = List.empty,
    /** What this architecture is for: `image`, `video`, `edit`, `audio`, `llm`,
      * `upscale` — as many as apply (François, 2026-09-10). Declared rather
      * than inferred, and open-ended: a model that does something drift has not
      * met yet gets a word for it without a schema change.
      */
    tags: List[String] = List.empty,
    builtIn: Boolean = false,
    /** A PiD pixel-diffusion decoder (`specs/15-post-hoc-resize.md`): its run
      * configurations upscale a reference image rather than generate from a
      * prompt, so the gallery offers them as diffusion upscalers.
      */
    pixelDiffusionDecoder: Boolean = false,
    /** Whether the model repaints an image it is *given*: img2img from an init
      * image at a denoising strength. True of every ordinary image model, and
      * what a redraw (`specs/27`) is made of.
      */
    initImage: Boolean = true,
    /** What this model does with a reference image, which is not a yes or no
      * (`ReferenceImageUse`). sd-cpp gives each family a preset
      * (`get_default_ref_image_preset`), and the presets differ in kind: some
      * let the reference inform the result, others hand it to the diffusion
      * model as the image being *edited* — and that one reproduces the
      * reference's own layout in every tile it is given (François, 2026-09-19,
      * redrawing with Krea2).
      *
      * It has to be declared here, and it cannot be asked for: sd-server's
      * `features_by_mode` answers `ref_images: true` for every model from a
      * hardcoded table (`make_img_gen_features_json`), and its API exposes no
      * `ref_image_args`, so neither the preset nor a way around it crosses the
      * wire.
      */
    referenceImages: ReferenceImageUse = ReferenceImageUse.Unused,
    /** How this model reads a prompt, in a sentence or two for the assistant
      * (`specs/20`): natural-language description or tag lists, what helps and
      * what does nothing. Seeded for the built-ins; `None` gets a generic note.
      */
    promptingNotes: Option[String] = None,
    /** The assistant system template a project targeting this architecture
      * starts with (`specs/32-prompt-library.md`): Ideogram names its caption
      * writer, `None` is drift's helper. A project's own pick wins over it.
      */
    assistantTemplateId: Option[String] = None,
    /** The HuggingFace repositories this architecture's LoRAs are trained on,
      * most relevant first (`specs/33-lora-sources.md`): the HuggingFace LoRA
      * browser lists their adapters (`base_model:adapter:<repo>`) by default,
      * as the Civitai browser filters by `civitaiBaseModels`.
      */
    huggingFaceBaseModels: List[String] = List.empty,
    /** The base model names this architecture's LoRAs declare on ModelScope
      * (`specs/37-modelscope.md`), revision suffixes included (`…@master`): the
      * ModelScope LoRA browser lists the LoRAs of all of them at once.
      */
    modelScopeBaseModels: List[String] = List.empty,
    /** The multiple every side of a generated image is aligned **up** to, which
      * is sd-cpp's `vae_scale_factor × diffusion_model_down_factor`
      * (`GenerationRequest::align_image_size`). It has to be declared, like
      * `referenceImages`: sd-cpp aligns the request silently and answers with
      * the bigger image, and a tiled job that asked for anything else fails on
      * the size it gets back (François, 2026-09-20, redrawing 1280×1264 tiles
      * that came back 1280×1280).
      *
      * No default: every architecture says its own, the built-ins from the
      * reference file and a user's from the form, so a family whose number is
      * not 16 cannot inherit one that is (François, 2026-09-20).
      */
    sizeMultiple: Int,
    /** What the model is, as the engines name it (`specs/43`): the GGUF
      * `general.architecture` (`qwen35moe`) or a diffusion family (`krea2`).
      * None when nobody said.
      */
    modelKind: Option[String],
    /** The engines that run it: its tool's upstream one, and the drift runner
      * where it has been run (`specs/43`).
      */
    runners: List[RuntimeEngine]
)
object Architecture {
  // No discriminator: `RuntimeTool` has only singleton cases, so it encodes as
  // a plain string ("SdCpp"), as the seed file and the UI write it. The config
  // must be inline -- the macro needs a constant expression.
  given JsonValueCodec[Architecture] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[Architecture]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[Architecture]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** The tag vocabulary the UI suggests. Nothing enforces it - `Architecture
  * .tags` is free text so an unforeseen kind of model needs no schema change -
  * but offering a fixed set keeps "llm" from also being written "LLM" and
  * "chat", which would split a filter in two.
  */
object ArchitectureTags {
  val Image = "image"
  val Video = "video"
  val Edit = "edit"
  val Audio = "audio"
  val Llm = "llm"
  val Upscale = "upscale"

  /** Offered by the form and ordered first in the filters. */
  val suggested: List[String] =
    List(Image, Video, Edit, Audio, Llm, Upscale)

  /** Tags for an architecture registered before tags existed, used once to fill
    * them in (`StorageService.init`) and never as a live rule - the whole point
    * of the field is that a model says what it is rather than being guessed at.
    *
    * The guesses it can make honestly: llama.cpp is an llm, a pixel diffusion
    * decoder is an upscaler, and anything defaulting a frame count makes video,
    * since only a video model has frames to set.
    */
  def derive(architecture: Architecture): List[String] =
    if (architecture.tool == RuntimeTool.LlamaCpp) List(Llm)
    else if (architecture.pixelDiffusionDecoder) List(Upscale)
    else if (
      architecture.defaultParameters.contains("--video-frames") ||
      architecture.defaultParameters.contains("--frames")
    ) List(Video)
    else List(Image)

  /** The tags in use, suggested ones first and in their canonical order, then
    * anything the user invented, alphabetically - so the filter row is stable
    * rather than shuffling as architectures load.
    */
  def ordered(inUse: Set[String]): List[String] =
    suggested.filter(inUse.contains) ++ (inUse -- suggested).toList.sorted
}

case class CheckpointRef(
    name: String,
    familyId: String,
    /** The command-line flag the assigned file is passed with, e.g.
      * `--diffusion-model` (sd-cpp) or `--mmproj` (llama.cpp).
      */
    flag: String,
    required: Boolean = true
) {

  /** The architecture's own model — what makes it this architecture — rather
    * than a component it shares with others (a text encoder, a VAE, a vision
    * projector…): read off the flag it is passed with.
    */
  def ownModel: Boolean = CheckpointRef.ownModelFlags.contains(flag)
}
object CheckpointRef {
  private val ownModelFlags = Set(
    "--diffusion-model",
    "--high-noise-diffusion-model",
    "--uncond-diffusion-model",
    "--model",
    "-m"
  )
}

enum ModelSourceType derives CanEqual {
  case HuggingFace, ModelScope, Civitai, Local
}

sealed trait ModelSource derives CanEqual {

  /** Which kind it is, for a form that shows the source it was given. */
  def kind: ModelSourceType = this match {
    case _: HuggingFace => ModelSourceType.HuggingFace
    case _: ModelScope  => ModelSourceType.ModelScope
    case _: Civitai     => ModelSourceType.Civitai
    case _: Local       => ModelSourceType.Local
  }

  /** Where the file comes from, as the model list and the model form show it:
    * the site and what it takes to find it again, a line each — a HuggingFace
    * or ModelScope file name is long enough to want its own.
    */
  def lines: List[String] = this match {
    case HuggingFace(repo, file) => List(s"HuggingFace $repo", file)
    case ModelScope(repo, file)  => List(s"ModelScope $repo", file)
    case Civitai(id, _, _, file) => List(s"Civitai model $id", file)
    case Local(path)             => List("On this machine", path)
  }

  /** The file's own name, wherever it comes from. */
  def fileName: String = this match {
    case HuggingFace(_, file) => file.split('/').last
    case ModelScope(_, file)  => file.split('/').last
    case Civitai(_, _, _, f)  => f
    case Local(path)          => path.split('/').last
  }
}
object ModelSource {

  /** Hierarchies are discriminated by a lowercase `"type"`, and the key may sit
    * anywhere in the object -- jsoniter otherwise demands it come first, which
    * a hand-edited config file has no reason to respect.
    */
  given JsonValueCodec[ModelSource] = JsonCodecMaker.make(
    CodecMakerConfig
      .withAdtLeafClassNameMapper(n =>
        JsonCodecMaker.simpleClassName(n).toLowerCase
      )
      .withRequireDiscriminatorFirst(false)
  )
}

case class HuggingFace(repo: String, filename: String) extends ModelSource

/** A file of a ModelScope repository (`specs/37-modelscope.md`), `repo` being
  * `owner/name` and `filename` its path in the repository. Fetched into drift's
  * own `modelscope/` tree, not a ModelScope SDK cache.
  */
case class ModelScope(repo: String, filename: String) extends ModelSource

/** A split safetensors model is named by its `*.safetensors.index.json`, which
  * maps every tensor to one of the shard files beside it. sd-cpp loads the
  * index; drift downloads and checks the shards with it
  * (`specs/34-sharded-safetensors.md`).
  */
object ShardedSafetensors {
  def isIndex(filename: String): Boolean =
    filename.toLowerCase.endsWith(".safetensors.index.json")
}

/** Civitai downloads are addressed by `modelId` + `fileId`; `filename` is what
  * the file is called on disk. `fileId` is required and undefaulted on purpose
  * -- a Civitai source without one cannot be fetched, so that should not
  * compile.
  */
case class Civitai(
    modelId: String,
    versionId: String,
    fileId: String,
    filename: String
) extends ModelSource
case class Local(path: String) extends ModelSource

case class Model(
    id: String,
    familyId: String,
    label: String,
    source: ModelSource,
    format: String,
    parameters: Map[String, String],
    /** Flags this checkpoint does not want at all, whatever the architecture
      * below it set (`specs/16-parameter-resolution.md`). A value cannot say
      * this: `""` is the convention for a valueless flag that *is* passed, so
      * an "empty" override would put `--steps ""` on the command line. A layer
      * above can set the flag again.
      */
    removedParameters: List[String] = List.empty,
    builtIn: Boolean = false
)
object Model {
  given JsonValueCodec[Model] = JsonCodecMaker.make
  given JsonValueCodec[List[Model]] = JsonCodecMaker.make
  given JsonValueCodec[Option[Model]] = JsonCodecMaker.make
}

case class RunConfiguration(
    id: String,
    label: String,
    architectureId: String,
    assignments: Map[String, String],
    overriddenParameters: Map[String, String],
    /** Flags this configuration does not want at all, whatever the
      * architecture, its models or the runtime set
      * (`specs/16-parameter-resolution.md`). The top layer removes as it
      * overrides: nothing above it can put the flag back.
      */
    removedParameters: List[String] = List.empty,
    createdAt: Long,
    /** When this configuration last launched a session — `createdAt` until the
      * first launch, so a fresh configuration sorts like a just-used one. The
      * inference page orders its grid by this, most recent first.
      */
    lastUsedAt: Long,
    /** LoRAs applied wherever this configuration runs — its generation form
      * starts with them, a tiled job sends them with every tile
      * (`specs/28-configuration-loras.md`). Only LoRAs of its architecture.
      */
    loras: List[ConfiguredLora] = List.empty,
    /** The assistant system template this configuration prefers over its
      * architecture's default (`specs/32-prompt-library.md`); none means the
      * architecture's. A project's assistant is offered a switch to it when the
      * configuration's model starts.
      */
    assistantTemplateId: Option[String] = None,
    /** The engine it runs on, one of its architecture's runners (`specs/43`):
      * every launch path resolves its runtime from it.
      */
    runner: RuntimeEngine
)
object RunConfiguration {
  // No discriminator: `RuntimeEngine` encodes as a plain string ("SdCpp").
  given JsonValueCodec[RunConfiguration] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[RunConfiguration]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[RunConfiguration]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** What a repository's model makes (`specs/36-huggingface-examples.md`): an
  * image or video of its card's gallery (the card's `widget`), with the prompt
  * the card gives for it, or a media file found in the repository.
  */
case class ModelExample(
    url: String,
    video: Boolean = false,
    prompt: Option[String] = None
)

case class HuggingFaceModelInfo(
    id: String,
    pipeline_tag: Option[String] = None,
    library_name: Option[String] = None,
    tags: List[String] = List.empty,
    downloads: Option[Long] = None,
    likes: Option[Long] = None,
    last_modified: Option[String] = None,
    author: Option[String] = None,
    /** Filled by drift, not HuggingFace: the first few examples, for the
      * result's tile — the next is tried when one does not load — and how many
      * there are.
      */
    previews: List[ModelExample] = Nil,
    exampleCount: Int = 0
)
object HuggingFaceModelInfo {
  given JsonValueCodec[List[HuggingFaceModelInfo]] = JsonCodecMaker.make
}

/** The `lfs` block HuggingFace attaches to large files. Only present when the
  * listing is requested with `?blobs=true`.
  */
case class HuggingFaceLfs(
    sha256: Option[String] = None,
    size: Option[Long] = None
)

case class HuggingFaceFileInfo(
    rfilename: String,
    size: Option[Long] = None,
    lfs: Option[HuggingFaceLfs] = None,
    /** The git blob id, which is the file's ETag — and so its blob name in the
      * HuggingFace cache — when the file is not in LFS.
      */
    blobId: Option[String] = None
) {

  /** The content SHA256, which is also the blob filename in HuggingFace's own
    * cache — simultaneously the verification hash and the cache key.
    */
  def sha256: Option[String] = lfs.flatMap(_.sha256).map(_.toLowerCase)
}

/** HuggingFace's `gated` flag: absent for an open repository, else the kind of
  * gate — "auto", where accepting the terms on the site is enough, or "manual",
  * where the author approves each request by hand. Either way a download needs
  * a token from an account that has been let in, so the browser says so before
  * a file is picked rather than leaving it to a 403.
  *
  * The field is a JSON union — `false`, or a string — which no derived codec
  * reads, hence this one; it writes the same shape back.
  */
case class Gated(kind: Option[String] = None) {
  def isGated: Boolean = kind.isDefined

  /** Whether the author lets people in one by one. */
  def isManual: Boolean = kind.contains("manual")
}
object Gated {
  given Schema[Gated] = Schema.derived

  given JsonValueCodec[Gated] = new JsonValueCodec[Gated] {
    def nullValue: Gated = Gated()

    def decodeValue(in: JsonReader, default: Gated): Gated = {
      val token = in.nextToken()
      in.rollbackToken()
      if (token == '"') Gated(Some(in.readString(null)))
      else if (token == 'n') in.readNullOrError(default, "expected null")
      else {
        in.readBoolean()
        Gated()
      }
    }

    def encodeValue(x: Gated, out: JsonWriter): Unit =
      x.kind match {
        case Some(kind) => out.writeVal(kind)
        case None       => out.writeVal(false)
      }
  }
}

case class HuggingFaceModelDetail(
    id: String,
    siblings: List[HuggingFaceFileInfo],
    /** The commit of `main`, which names the snapshot directory in the cache.
      */
    sha: Option[String] = None,
    /** Whether HuggingFace gates this repository, and how. */
    gated: Gated = Gated(),
    /** Filled by drift for the browser's Examples tab: the card's gallery, then
      * the repository's other media files.
      */
    examples: List[ModelExample] = Nil
)
object HuggingFaceModelDetail {
  given JsonValueCodec[HuggingFaceModelDetail] = JsonCodecMaker.make
  given JsonValueCodec[Option[HuggingFaceModelDetail]] = JsonCodecMaker.make
}

/** A repository's model card, rendered to HTML and cleaned by the backend
  * (`specs/24`), so the page inserts it as-is. Without `html` there is nothing
  * to show: `gated` when the repository wants its terms accepted and a token,
  * otherwise it has no README.
  */
case class HuggingFaceModelCard(
    html: Option[String] = None,
    gated: Boolean = false
)
object HuggingFaceModelCard {
  given JsonValueCodec[HuggingFaceModelCard] = JsonCodecMaker.make
}

/** Whether a registered model's weights are on disk.
  *
  * `Missing` and `Broken` are distinct on purpose: a missing file can be
  * downloaded, a broken `Local` path cannot — the user pointed at something
  * that is not there.
  */
enum CacheState derives CanEqual {
  case Cached, Missing, Broken
}

case class ModelCacheStatus(
    modelId: String,
    state: CacheState,
    path: Option[String] = None,
    bytes: Option[Long] = None
)
object ModelCacheStatus {
  // No discriminator: `CacheState` has only singleton cases, so jsoniter encodes
  // it as a plain string ("Cached") rather than an object ({"type":"Cached"}).
  // The config has to be inline -- the macro needs a constant expression.
  given JsonValueCodec[ModelCacheStatus] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[ModelCacheStatus]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Where a cached file came from — the on-disk view tabs by this. `Civitai`
  * means a drift-root file whose sidecar names Civitai as the source;
  * everything else in the drift models directory is `Local`. `Lora` is reserved
  * for `specs/09-lora-management.md`; nothing produces it yet.
  */
enum CachedFileKind derives CanEqual {
  case HuggingFace, ModelScope, Civitai, Local, Lora
}
object CachedFileKind {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[CachedFileKind] =
    Schema.derivedEnumeration[CachedFileKind].defaultStringBased
}

/** One file the cache actually holds, whether or not a model references it. The
  * on-disk view of the Model Cache page is built from these; orphans —
  * `referencedBy` empty — are exactly what that view exists to surface.
  */
case class CachedFileEntry(
    /** Real path; also the deletion key. */
    path: String,
    bytes: Long,
    /** Which store holds it: "drift" or "huggingface". */
    root: String,
    /** Family directory (drift) or repository name (HuggingFace). */
    group: String,
    /** Something readable: snapshot filename, or the sidecar's model name. */
    label: String,
    kind: CachedFileKind,
    referencedBy: List[String] = List.empty,
    /** A `.part` / `.incomplete` left by an interrupted download. */
    partial: Boolean = false,
    /** Where the file came from, as a model adopting it should reference it — a
      * HuggingFace repo file or a Civitai file — set only when that source
      * resolves back to this very file. None means referencing it in place, by
      * its path.
      */
    source: Option[ModelSource] = None
)
object CachedFileEntry {
  // No discriminator, so `CachedFileKind` encodes as a plain string. The
  // config must be inline -- the macro needs a constant expression.
  given JsonValueCodec[CachedFileEntry] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[CachedFileEntry]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

case class FileEntry(
    name: String,
    path: String,
    isDirectory: Boolean,
    size: Long,
    extension: String
)
object FileEntry {
  given JsonValueCodec[List[FileEntry]] = JsonCodecMaker.make
}

/** Every endpoint's `/api` prefix. Package-visible, so `Civitai.scala` builds
  * on it too: file-private, it once left the Civitai endpoints at the server
  * root (bugs/16).
  */
private[shared] val base = endpoint.in("api")

// The endpoints below are `lazy` for one reason: a plain top-level `val`
// initializes in this file's static initializer, and tapir's auto-derived
// schemas are enough bytecode that all of them together overran the JVM's
// 64 KB limit on one method ("Method too large:
// drift/shared/Api$package$.<clinit>") the day `Model` and `RunConfiguration`
// each gained a field (`specs/16-parameter-resolution.md`). A lazy val gets
// its own initializer, so the ceiling is per endpoint rather than per file.

lazy val listArchitectures
    : PublicEndpoint[Unit, Unit, List[Architecture], Any] =
  base.get.in("architectures").out(jsonBody[List[Architecture]])

lazy val getArchitecture
    : PublicEndpoint[String, Unit, Option[Architecture], Any] =
  base.get
    .in("architectures" / path[String])
    .out(jsonBody[Option[Architecture]])

lazy val createArchitecture
    : PublicEndpoint[Architecture, Unit, Architecture, Any] =
  base.post
    .in("architectures")
    .in(jsonBody[Architecture])
    .out(jsonBody[Architecture])

lazy val updateArchitecture
    : PublicEndpoint[(String, Architecture), Unit, Option[Architecture], Any] =
  base.put
    .in("architectures" / path[String])
    .in(jsonBody[Architecture])
    .out(jsonBody[Option[Architecture]])

lazy val deleteArchitecture: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("architectures" / path[String]).out(jsonBody[Boolean])

lazy val listModels: PublicEndpoint[Unit, Unit, List[Model], Any] =
  base.get.in("models").out(jsonBody[List[Model]])

lazy val getModel: PublicEndpoint[String, Unit, Option[Model], Any] =
  base.get.in("models" / path[String]).out(jsonBody[Option[Model]])

lazy val createModel: PublicEndpoint[Model, Unit, Model, Any] =
  base.post.in("models").in(jsonBody[Model]).out(jsonBody[Model])

lazy val updateModel
    : PublicEndpoint[(String, Model), Unit, Option[Model], Any] =
  base.put
    .in("models" / path[String])
    .in(jsonBody[Model])
    .out(jsonBody[Option[Model]])

lazy val deleteModel: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("models" / path[String]).out(jsonBody[Boolean])

lazy val listRunConfigurations
    : PublicEndpoint[Unit, Unit, List[RunConfiguration], Any] =
  base.get.in("run-configurations").out(jsonBody[List[RunConfiguration]])

lazy val getRunConfiguration
    : PublicEndpoint[String, Unit, Option[RunConfiguration], Any] =
  base.get
    .in("run-configurations" / path[String])
    .out(jsonBody[Option[RunConfiguration]])

lazy val createRunConfiguration
    : PublicEndpoint[RunConfiguration, Unit, RunConfiguration, Any] =
  base.post
    .in("run-configurations")
    .in(jsonBody[RunConfiguration])
    .out(jsonBody[RunConfiguration])

lazy val updateRunConfiguration
    : PublicEndpoint[(String, RunConfiguration), Unit, Option[
      RunConfiguration
    ], Any] =
  base.put
    .in("run-configurations" / path[String])
    .in(jsonBody[RunConfiguration])
    .out(jsonBody[Option[RunConfiguration]])

lazy val deleteRunConfiguration: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete.in("run-configurations" / path[String]).out(jsonBody[Boolean])

/** Cache state for every registered model. Cheap enough to return wholesale: it
  * is a stat per model, and the UI wants the whole picture on every page it
  * appears on.
  */
lazy val cacheStatus: PublicEndpoint[Unit, Unit, List[ModelCacheStatus], Any] =
  base.get.in("cache" / "status").out(jsonBody[List[ModelCacheStatus]])

lazy val listCachedFiles
    : PublicEndpoint[Unit, Unit, List[CachedFileEntry], Any] =
  base.get.in("cache" / "files").out(jsonBody[List[CachedFileEntry]])

/** Deletes one cached file by its listed path. The backend refuses anything
  * outside its roots; whether deleting is *wise* — a ready run configuration
  * may depend on the file — is the frontend's warning to give.
  */
lazy val deleteCachedFile: PublicEndpoint[String, Unit, Boolean, Any] =
  base.delete
    .in("cache" / "files")
    .in(query[String]("path"))
    .out(jsonBody[Boolean])

/** The Civitai file ids of one model already present in the cache — what the
  * browser's "downloaded" markers are built from. Read off the on-disk
  * `<fileId>-<filename>` naming, so it needs no registry entry to answer.
  */
lazy val listCivitaiCachedFileIds
    : PublicEndpoint[String, Unit, List[String], Any] =
  base.get
    .in("cache" / "civitai-files")
    .in(query[String]("modelId"))
    .out(jsonBody[List[String]])

lazy val getHomeDirectory: PublicEndpoint[Unit, Unit, String, Any] =
  base.get.in("files" / "home").out(stringBody)

lazy val listDirectory: PublicEndpoint[String, Unit, List[FileEntry], Any] =
  base.get
    .in("files" / "list")
    .in(query[String]("path"))
    .out(jsonBody[List[FileEntry]])

/** `baseModel` narrows the search to that repository's adapters — LoRAs trained
  * on it (`specs/33-lora-sources.md`).
  */
lazy val searchHuggingFaceModels: PublicEndpoint[
  (
      String,
      Option[Int],
      Option[Int],
      Option[String],
      Option[Int],
      Option[
        String
      ]
  ),
  Unit,
  List[HuggingFaceModelInfo],
  Any
] =
  base.get
    .in("hf-search")
    .in(query[String]("q"))
    .in(query[Option[Int]]("limit"))
    .in(query[Option[Int]]("offset"))
    .in(query[Option[String]]("sort"))
    .in(query[Option[Int]]("direction"))
    .in(query[Option[String]]("baseModel"))
    .out(jsonBody[List[HuggingFaceModelInfo]])

lazy val getHuggingFaceModelDetail
    : PublicEndpoint[(String, String), Unit, Option[
      HuggingFaceModelDetail
    ], Any] =
  base.get
    .in("hf-model" / path[String] / path[String])
    .out(jsonBody[Option[HuggingFaceModelDetail]])

/** The error output of an endpoint relaying another site (`specs/24`): 502, and
  * the site's reason in words, which the page shows as it is — not the client's
  * "Endpoint … returned error: undefined, inputs: …".
  */
private[shared] val upstreamFailure: EndpointOutput[String] =
  statusCode(sttp.model.StatusCode.BadGateway).and(stringBody)

/** The repository's README, rendered and cleaned (`specs/24`). */
lazy val getHuggingFaceModelCard
    : PublicEndpoint[(String, String), String, HuggingFaceModelCard, Any] =
  base.get
    .in("hf-card" / path[String] / path[String])
    .errorOut(upstreamFailure)
    .out(jsonBody[HuggingFaceModelCard])
