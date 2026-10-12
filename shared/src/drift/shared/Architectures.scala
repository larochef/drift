package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

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
    civitaiBaseModels: List[String],
    /** What this architecture is for: `image`, `video`, `edit`, `audio`, `llm`,
      * `upscale` — as many as apply. Declared rather than inferred, and
      * open-ended: a model that does something drift has not met yet gets a
      * word for it without a schema change.
      */
    tags: List[String],
    builtIn: Boolean,
    /** A PiD pixel-diffusion decoder (`specs/15-post-hoc-resize.md`): its run
      * configurations upscale a reference image rather than generate from a
      * prompt, so the gallery offers them as diffusion upscalers.
      */
    pixelDiffusionDecoder: Boolean,
    /** Whether the model repaints an image it is *given*: img2img from an init
      * image at a denoising strength. True of every ordinary image model, and
      * what a redraw (`specs/27`) is made of.
      */
    initImage: Boolean,
    /** What this model does with a reference image, which is not a yes or no
      * (`ReferenceImageUse`). sd-cpp gives each family a preset
      * (`get_default_ref_image_preset`), and the presets differ in kind: some
      * let the reference inform the result, others hand it to the diffusion
      * model as the image being *edited* — and that one reproduces the
      * reference's own layout in every tile it is given (seen redrawing with
      * Krea 2).
      *
      * It has to be declared here, and it cannot be asked for: sd-server's
      * `features_by_mode` answers `ref_images: true` for every model from a
      * hardcoded table (`make_img_gen_features_json`), and its API exposes no
      * `ref_image_args`, so neither the preset nor a way around it crosses the
      * wire.
      */
    referenceImages: ReferenceImageUse,
    /** How this model reads a prompt, in a sentence or two for the assistant
      * (`specs/20`): natural-language description or tag lists, what helps and
      * what does nothing. Seeded for the built-ins; `None` gets a generic note.
      */
    promptingNotes: Option[String],
    /** The assistant system template a project targeting this architecture
      * starts with (`specs/32-prompt-library.md`): Ideogram names its caption
      * writer, `None` is drift's helper. A project's own pick wins over it.
      */
    assistantTemplateId: Option[String],
    /** The HuggingFace repositories this architecture's LoRAs are trained on,
      * most relevant first (`specs/33-lora-sources.md`): the HuggingFace LoRA
      * browser lists their adapters (`base_model:adapter:<repo>`) by default,
      * as the Civitai browser filters by `civitaiBaseModels`.
      */
    huggingFaceBaseModels: List[String],
    /** The base model names this architecture's LoRAs declare on ModelScope
      * (`specs/37-modelscope.md`), revision suffixes included (`…@master`): the
      * ModelScope LoRA browser lists the LoRAs of all of them at once.
      */
    modelScopeBaseModels: List[String],
    /** The multiple every side of a generated image is aligned **up** to, which
      * is sd-cpp's `vae_scale_factor × diffusion_model_down_factor`
      * (`GenerationRequest::align_image_size`). It has to be declared, like
      * `referenceImages`: sd-cpp aligns the request silently and answers with
      * the bigger image, and a tiled job that asked for anything else fails on
      * the size it gets back (1280×1264 tiles came back 1280×1280).
      *
      * No default: every architecture says its own, the built-ins from the
      * reference file and a user's from the form, so a family whose number is
      * not 16 cannot inherit one that is.
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
) {

  /** Whether `configuration`, one of this architecture's, is a chat model
    * launched with a vision projector — what a live session's `/props` says
    * once it serves (`AssistantProperties.vision`), known before any launch.
    */
  def readsImages(configuration: RunConfiguration): Boolean =
    tool == RuntimeTool.LlamaCpp &&
      // a flag the configuration removes is not passed, whatever is assigned
      !configuration.removedParameters
        .contains(CheckpointRef.VisionProjectorFlag) &&
      checkpoints.exists(slot =>
        slot.flag == CheckpointRef.VisionProjectorFlag &&
          configuration.assignments.get(slot.name).exists(_.nonEmpty)
      )
}
object Architecture {
  // No discriminator: `RuntimeTool` has only singleton cases, so it encodes as
  // a plain string ("SdCpp"), as the seed file and the UI write it. The config
  // must be inline -- the macro needs a constant expression.
  given JsonValueCodec[Architecture] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[Architecture]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[Architecture]] =
    JsonCodecMaker.make(storedRecord.withDiscriminatorFieldName(None))
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
    required: Boolean,
    /** The runners the slot is for; empty means every runner of the
      * architecture. On another runner the slot does not exist: nothing is
      * asked for it and nothing assigned to it is passed — a tokenizer file the
      * drift runner cannot do without and sd-cpp does not read.
      */
    runners: List[RuntimeEngine]
) {

  def appliesTo(runner: RuntimeEngine): Boolean =
    runners.isEmpty || runners.contains(runner)

  /** The architecture's own model — what makes it this architecture — rather
    * than a component it shares with others (a text encoder, a VAE, a vision
    * projector…): read off the flag it is passed with.
    */
  def ownModel: Boolean = CheckpointRef.ownModelFlags.contains(flag)
}
object CheckpointRef {

  /** The flag a chat model's vision projector is passed with. */
  val VisionProjectorFlag = "--mmproj"

  private val ownModelFlags = Set(
    "--diffusion-model",
    "--high-noise-diffusion-model",
    "--uncond-diffusion-model",
    "--model",
    "-m"
  )
}

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
