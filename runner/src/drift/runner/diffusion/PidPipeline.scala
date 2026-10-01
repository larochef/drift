package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** PiD 1.5 end to end, as NVIDIA's distilled inference runs it
  * (`pid_distill_model_infer.py`): the request's first reference image is the
  * source, a quarter of the output's size; its latent conditions a pixel-space
  * decode to the full size in 4 steps. The checkpoint tells the latent
  * (`PidConfig.latentUnpatchify`), the VAE's names its encoder: FLUX.2's or
  * FLUX.1's (LDM names, `FluxVae`), or Wan 2.1's for Qwen Image (`WanVae`).
  *   - Text: Gemma 2 2B's last hidden state (normed) over the official
  *     instruction prompt followed by the caption, right-padded with id 0 to
  *     the prompt's length + 298 (the padding masked as keys), keeping the BOS
  *     row and the last 299 (sd-cpp encodes the caption without the prompt).
  *   - Sampling: pure Gaussian RGB, σ = 0.999, 0.866, 0.634, 0.342 (the
  *     student's list, no shift); each step `x0 = x − σ v`, then `x = (1 − σ')
  *     x0 + σ' ε` with fresh noise, the last step's `x0` the image.
  */
final class PidPipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Path
) extends ImagePipeline {

  def family: String = "PiD"
  def takesInitImage: Boolean = true
  def takesReferences: Boolean = true
  def takesLoras: Boolean = false
  def takesGuidance: Boolean = false

  private val encoder = Gemma2.open(ops, textEncoder)
  private val tokenizer: Tokenizer = TokenizerJson.load(tokenizerFile)
  private val decoder = Pid.open(ops, diffusionModel)
  private val autoencoder: Either[FluxVae, WanVae] = {
    val source = WeightSource.open(ops, vae)
    val wan =
      try source.has("encoder.conv1.weight")
      finally source.close()
    if (wan) Right(WanVae.open(ops, vae)) else Left(FluxVae.open(ops, vae))
  }

  /** Image pixels per latent row along each side, and features per row. */
  private val (latentDown, latentFeatures) =
    autoencoder.fold(
      vae => (vae.patch, vae.features.toInt),
      vae => (vae.scale, vae.LatentMean.length)
    )
  require(
    latentDown == decoder.config.latentDown &&
      latentFeatures == decoder.config.latentFeatures,
    s"$vae gives $latentFeatures features at 1/$latentDown, this PiD takes ${decoder.config.latentFeatures} at 1/${decoder.config.latentDown}"
  )

  /** `pixels`' latent as the decoder takes it, `[h × w, latentFeatures]`. */
  private def encode(pixels: Tensor): Tensor =
    autoencoder.fold(
      _.encode(pixels),
      vae => {
        val latents = vae.encode(pixels)
        val Seq(height, width, channels) = latents.shape.dimensions
        latents.view(height * width, channels)
      }
    )

  /** The official prompt the caption follows (`_CHI_PROMPT`). */
  private val Instruction = Seq(
    "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: "
  ).mkString("\n")

  /** The distilled student's noise levels. */
  private val Sigmas = Seq(0.999f, 0.866f, 0.634f, 0.342f, 0f)

  /** The caption's text features, `[textLength, hidden]`. */
  private def text(caption: String): Tensor = {
    val length = decoder.config.textLength
    val instruction = tokenizer.encode(Instruction, addSpecial = true).length
    val kept = instruction + length - 2
    val ids =
      tokenizer.encode(Instruction + caption, addSpecial = true).take(kept)
    val hidden = encoder.config.hidden.toLong
    val all = ops.allocate(DType.F32, Shape.of(kept.toLong, hidden))
    try {
      encoder.lastHiddenState(ids, all, kept - ids.length, 0)
      // the BOS row, then the last length − 1
      val out = ops.allocate(DType.F32, Shape.of(length.toLong, hidden))
      ops.copy(all.rows(0, 1), out.rows(0, 1))
      ops.copy(
        all.rows(kept - length + 1L, length - 1L),
        out.rows(1, length - 1L)
      )
      out
    } finally ops.release(all)
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val (width, height) = (request.width, request.height)
    val multiple = 4 * latentDown
    require(
      width % multiple == 0 && height % multiple == 0,
      s"$width × $height: this PiD decodes to multiples of $multiple (a quarter in multiples of $latentDown)"
    )
    require(
      request.steps == Sigmas.size - 1,
      s"PiD 1.5 is distilled for ${Sigmas.size - 1} steps, not ${request.steps}"
    )
    val source = request.references.headOption.getOrElse(
      throw new IllegalArgumentException(
        "PiD decodes a reference image: send the source as the first one"
      )
    )
    val patch = decoder.config.patch
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    try {
      val features = hold(text(request.prompt))
      val quarter = Images.resized(source, width / 4, height / 4)
      val pixels = hold(
        ops.fromFloats(
          Shape.of(height / 4L, width / 4L, 3),
          Images.pixels(quarter)
        )
      )
      val latent = hold(encode(pixels))
      val conditioning = decoder.conditionOn(
        features,
        latent,
        height / 4 / latentDown,
        height / patch,
        width / patch,
        0f
      )
      try {
        val random = new SplittableRandom(request.seed)
        def noise() =
          Array.fill(width * height * 3)(Images.gaussian(random))
        val shape = Shape.of(height.toLong, width.toLong, 3)
        val x = hold(ops.fromFloats(shape, noise()))
        val velocity = hold(ops.allocate(DType.F32, shape))
        val steps = Sigmas.size - 1
        progress(0, steps)
        (0 until steps).foreach { i =>
          val (sigma, next) = (Sigmas(i), Sigmas(i + 1))
          decoder.velocity(x, sigma, conditioning, velocity)
          // x0 = x − σ v, then back to σ' with fresh noise
          ops.scale(velocity, -sigma, velocity)
          ops.add(x, velocity, x)
          if (next > 0f) {
            val fresh = ops.fromFloats(shape, noise())
            try {
              ops.scale(x, 1 - next, x)
              ops.scale(fresh, next, fresh)
              ops.add(x, fresh, x)
            } finally ops.release(fresh)
          }
          progress(i + 1, steps)
        }
        Images.toImage(ops.toFloats(x), width, height)
      } finally conditioning.release()
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    autoencoder.fold(_.close(), _.close())
    decoder.close()
    encoder.close()
  }
}
