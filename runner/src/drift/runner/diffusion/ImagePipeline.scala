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
    shift: Double,
    /** LoRA files, each with its multiplier. */
    loras: Seq[(Path, Float)],
    /** img2img's starting image, already `width × height`. */
    initImage: Option[BufferedImage],
    /** How much of the schedule img2img runs, in (0, 1]. */
    strength: Float,
    /** Reference images, each its own size. */
    references: Seq[BufferedImage]
)

/** A model family end to end, from a prompt to an image. */
trait ImagePipeline extends AutoCloseable {

  /** The family's name, as messages give it. */
  def family: String

  /** Whether requests may carry an init image (img2img). */
  def takesInitImage: Boolean

  /** Whether requests may carry reference images. */
  def takesReferences: Boolean

  /** Whether requests may carry LoRAs. */
  def takesLoras: Boolean

  /** Whether the model reads the distilled guidance scale. */
  def takesGuidance: Boolean

  /** The image of `request`; `progress(step, steps)` after each step. */
  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage
}

object ImagePipeline {

  /** The pipeline of `diffusionModel`'s family: FLUX.2, Qwen Image 2.1 (which
    * edits with the text encoder's vision tower), PiD (which takes a
    * `tokenizer.json`), else Krea 2.
    */
  def open(
      ops: Ops,
      diffusionModel: Path,
      vae: Path,
      textEncoder: Path,
      tokenizer: Option[Path],
      textEncoderVision: Option[Path]
  ): ImagePipeline = {
    val (flux2, qwenImage21, pid) = {
      val source = WeightSource.open(ops, diffusionModel)
      try
        (
          Flux2Config.holds(source),
          QwenImage21Config.holds(source),
          source.has("lq_proj.pit_head.weight")
        )
      finally source.close()
    }
    if (flux2) new Flux2Pipeline(ops, diffusionModel, vae, textEncoder)
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
