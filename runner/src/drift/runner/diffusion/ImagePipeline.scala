package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.weights.WeightSource

import java.awt.image.BufferedImage
import java.nio.file.Path

/** One image request, in sd-server's terms: `cfgScale` 1 runs the prompt alone;
  * above, each step also runs `negativePrompt` and moves by `uncond + cfgScale
  * × (cond − uncond)`. `guidance` is the distilled guidance scale, read by the
  * models that embed one (FLUX.2 [dev]). `shift` is the schedule's μ (sd-cpp's
  * flow shift) where the model takes one.
  */
final case class ImageRequest(
    prompt: String,
    negativePrompt: String,
    width: Int,
    height: Int,
    steps: Int,
    cfgScale: Float,
    guidance: Float,
    seed: Long,
    /** The flow shift (sd-cpp's `--flow-shift`, `flow_shift`): the request's,
      * else the launch's, else `FlowSchedule.DefaultShift` — or none of them
      * for a family with its own (`ownShift`), which then runs that.
      */
    shift: Option[Double],
    /** LoRA files, each with its multiplier. */
    loras: Seq[(Path, Float)],
    /** img2img's starting image, already `width × height`. */
    initImage: Option[BufferedImage],
    /** How much of the schedule img2img runs, in (0, 1]. */
    strength: Float,
    /** With an init image, the part of it to repaint: white repainted, black
      * kept as it is (`Inpainting`), already `width × height`.
      */
    mask: Option[BufferedImage],
    /** Reference images, each its own size. */
    references: Seq[BufferedImage],
    /** The noise levels to step through in place of the model's schedule, 0
      * last (`FlowSchedule.custom`): `steps` is one less than their count.
      */
    sigmas: Option[IndexedSeq[Float]],
    /** The sampler asked for (sd-server's `sample_method`), one of the family's
      * `samplers`; none runs the family's own.
      */
    sampler: Option[String]
)

/** A model family end to end, from a prompt to an image. */
trait ImagePipeline extends AutoCloseable {

  /** The family's name, as messages give it. */
  def family: String

  /** Whether requests may carry an init image (img2img). */
  def takesInitImage: Boolean

  /** Whether an init image may come with a mask of the part to repaint. */
  def takesMask: Boolean = false

  /** Whether requests may carry reference images. */
  def takesReferences: Boolean

  /** Whether requests may carry LoRAs. */
  def takesLoras: Boolean

  /** Whether the model reads the distilled guidance scale. */
  def takesGuidance: Boolean

  /** Whether the family derives its flow shift itself (from the image's size)
    * and only reads one that a request or the launch gives.
    */
  def ownShift: Boolean = false

  /** Whether requests may carry their own noise levels (custom sigmas). */
  def takesSigmas: Boolean = false

  /** The samplers a request may ask for, by sd-server's names. */
  def samplers: Seq[String] = Seq("euler")

  /** The sampler a request without one runs, as the capabilities name it:
    * `default` where the family picks one by the request.
    */
  def defaultSampler: String = "euler"

  /** Handed each step's estimate of the finished picture, small, before the
    * step is counted (`specs/54-step-preview.md`): `preview(step, steps,
    * picture)`, `picture` drawn only if called, there and then. The server's to
    * set around `generate`; a family that cannot draw one never calls it.
    */
  var preview: (Int, Int, () => BufferedImage) => Unit = (_, _, _) => ()

  /** The image of `request`; `progress(0, steps)` before the first step,
    * `progress(step, steps)` after each.
    */
  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage
}

object ImagePipeline {

  /** The files a launch gives an image model, by what each is. */
  final case class ModelFiles(
      diffusionModel: Path,
      vae: Option[Path],
      textEncoder: Option[Path],
      tokenizer: Option[Path],
      textEncoderVision: Option[Path],
      connectors: Option[Path],
      lladaParts: Map[String, Path]
  ) {

    /** A file the family cannot do without, or the launch's error naming the
      * flag it comes by.
      */
    def needed(file: Option[Path], flag: String, what: String): Path =
      file.getOrElse(
        throw new IllegalArgumentException(
          s"${diffusionModel.getFileName} needs $what: pass $flag <file>"
        )
      )

    def neededVae: Path = needed(vae, "--vae", "a VAE")
    def neededTextEncoder: Path = needed(textEncoder, "--llm", "a text encoder")
  }

  /** A family of image models: how its diffusion model is recognised by its
    * weights, and how its pipeline is opened from the launch's files.
    */
  final case class Family(
      name: String,
      holds: WeightSource => Boolean,
      open: (Ops, ModelFiles) => ImagePipeline
  )

  /** Every family the runner draws with, in the order they are tried: the first
    * whose weights the file holds is the file's. All but HiDream O1 take a VAE
    * and a text encoder.
    */
  val Families: Seq[Family] = Seq(
    // one file; reads references with Qwen3-VL-8B's vision tower
    Family(
      "HiDream O1",
      HiDreamO1.holds,
      (ops, files) =>
        new HiDreamO1Pipeline(
          ops,
          files.diffusionModel,
          files
            .needed(
              files.tokenizer,
              "--tokenizer",
              "HiDream O1's tokenizer.json"
            ),
          files.textEncoderVision
        )
    ),
    Family(
      "FLUX.2",
      Flux2Config.holds,
      (ops, files) =>
        new Flux2Pipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder
        )
    ),
    // edits with the text encoder's vision tower
    Family(
      "Qwen Image 2.1",
      QwenImage21Config.holds,
      (ops, files) =>
        new QwenImage21Pipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder,
          files.textEncoderVision
        )
    ),
    Family(
      "PiD",
      _.has("lq_proj.pit_head.weight"),
      (ops, files) =>
        new PidPipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder,
          files
            .needed(files.tokenizer, "--tokenizer", "Gemma 2's tokenizer.json")
        )
    ),
    Family(
      "Nucleus-Image",
      NucleusConfig.holds,
      (ops, files) =>
        new NucleusPipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder
        )
    ),
    Family(
      "GRN",
      GrnConfig.holds,
      (ops, files) =>
        new GrnPipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.needed(files.textEncoder, "--llm", "umT5"),
          files.needed(files.tokenizer, "--tokenizer", "umT5's tokenizer.json")
        )
    ),
    // takes LLaDA2's `tokenizer.json` and its connectors, or finds both in
    // the released folders
    Family(
      "LLaDA-Image",
      LladaImageConfig.holds,
      (ops, files) =>
        new LladaImagePipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder,
          files.tokenizer,
          files.connectors,
          files.lladaParts
        )
    ),
    // The first Qwen Image has the same blocks, on 64 packed channels: it is
    // not Mage-Flow, and not a model the runner draws with. Edits with the
    // vision tower too.
    Family(
      "Mage-Flow",
      source =>
        MageFlowConfig.holds(source) && source("img_in.weight").shape.last ==
          MageFlowConfig.LatentChannels,
      (ops, files) =>
        new MageFlowPipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder,
          files.textEncoderVision
        )
    ),
    Family(
      "Krea 2",
      Krea2Config.holds,
      (ops, files) =>
        new Krea2Pipeline(
          ops,
          files.diffusionModel,
          files.neededVae,
          files.neededTextEncoder
        )
    )
  )

  /** The pipeline of the family the diffusion model is of. A file no family
    * recognises is refused, with the families the runner does draw with.
    */
  def open(ops: Ops, files: ModelFiles): ImagePipeline = {
    val source = WeightSource.open(ops, files.diffusionModel)
    val family =
      try Families.find(_.holds(source))
      finally source.close()
    family
      .getOrElse(
        throw new IllegalArgumentException(
          s"${files.diffusionModel.getFileName} is none of the image models " +
            s"the runner draws with (${Families.map(_.name).mkString(", ")})"
        )
      )
      .open(ops, files)
  }

  def open(
      ops: Ops,
      diffusionModel: Path,
      vaeFile: Option[Path],
      textEncoderFile: Option[Path],
      tokenizer: Option[Path],
      textEncoderVision: Option[Path],
      connectors: Option[Path] = None,
      lladaParts: Map[String, Path] = Map.empty
  ): ImagePipeline =
    open(
      ops,
      ModelFiles(
        diffusionModel,
        vaeFile,
        textEncoderFile,
        tokenizer,
        textEncoderVision,
        connectors,
        lladaParts
      )
    )
}
