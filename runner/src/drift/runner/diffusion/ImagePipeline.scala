package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops

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
    sigmas: Option[IndexedSeq[Float]]
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

  /** The image of `request`; `progress(0, steps)` before the first step,
    * `progress(step, steps)` after each.
    */
  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage
}

object ImagePipeline {

  /** The pipeline of `diffusionModel`'s family: HiDream O1 (one file, which
    * takes a `tokenizer.json`), FLUX.2, Qwen Image 2.1 (which edits with the
    * text encoder's vision tower), PiD (which takes a `tokenizer.json`), else
    * Krea 2. All but HiDream O1 take a VAE and a text encoder.
    */
  def open(
      ops: Ops,
      diffusionModel: Path,
      vaeFile: Option[Path],
      textEncoderFile: Option[Path],
      tokenizer: Option[Path],
      textEncoderVision: Option[Path]
  ): ImagePipeline = {
    val (hiDreamO1, flux2, qwenImage21, pid) = {
      val source = WeightSource.open(ops, diffusionModel)
      try
        (
          HiDreamO1.holds(source),
          Flux2Config.holds(source),
          QwenImage21Config.holds(source),
          source.has("lq_proj.pit_head.weight")
        )
      finally source.close()
    }
    def needed(file: Option[Path], flag: String, what: String) =
      file.getOrElse(
        throw new IllegalArgumentException(
          s"${diffusionModel.getFileName} needs $what: pass $flag <file>"
        )
      )
    lazy val vae = needed(vaeFile, "--vae", "a VAE")
    lazy val textEncoder = needed(textEncoderFile, "--llm", "a text encoder")
    if (hiDreamO1)
      new HiDreamO1Pipeline(
        ops,
        diffusionModel,
        needed(tokenizer, "--tokenizer", "HiDream O1's tokenizer.json")
      )
    else if (flux2) new Flux2Pipeline(ops, diffusionModel, vae, textEncoder)
    else if (qwenImage21)
      new QwenImage21Pipeline(
        ops,
        diffusionModel,
        vae,
        textEncoder,
        textEncoderVision
      )
    else if (pid)
      new PidPipeline(
        ops,
        diffusionModel,
        vae,
        textEncoder,
        tokenizer.getOrElse(
          throw new IllegalArgumentException(
            "PiD needs Gemma 2's tokenizer.json: pass --tokenizer <file>"
          )
        )
      )
    else new Krea2Pipeline(ops, diffusionModel, vae, textEncoder)
  }
}
