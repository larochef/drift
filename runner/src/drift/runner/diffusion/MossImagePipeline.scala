package drift.runner.diffusion

import drift.runner.lora.{LoraFiles, LoraReport}
import drift.runner.models.*
import drift.runner.models.Flux2.Grid
import drift.runner.ops.{Ops, TensorScope}
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** MOSS-Image end to end (OpenMOSS's `MOSSImagePipeline`), the SFT or the Base
  * checkpoint: the prompt in Qwen3.5's chat template, thinking on, through
  * Qwen3.5-9B to its last hidden state, then the transformer's text refiner
  * once; noise in packed 2 × 2 patches of the FLUX.2 VAE's 32 channels,
  * denoised by Euler steps on the scheduler's snapped levels (shift 5 unless
  * the request gives another); with CFG, the negative prompt (empty by default)
  * runs without block 22; the latents decoded by the FLUX.2 VAE. img2img: the
  * init image's latents mixed with the noise at the schedule's step
  * `steps − ⌊steps × strength⌋`, the steps from there run, under a mask when
  * one comes.
  */
final class MossImagePipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Option[Path]
) extends ImagePipeline {

  // the products' activations in BF16, as the reference runs them
  ops.wideProducts = true
  private val transformer = MossImage.open(ops, diffusionModel)

  def family: String = "MOSS-Image"
  override def defaultSteps: Int = 50
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = false
  def takesLoras: Boolean = true
  override def ownShift: Boolean = true
  override def takesSigmas: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen35.open(ops, textEncoder, None)
  // a GGUF's own vocabulary, else OpenMOSS's tokenizer.json (sd-cpp reads
  // that one in every case)
  private val tokenizer: Tokenizer =
    if (textEncoder.getFileName.toString.endsWith(".gguf"))
      GgufTokenizer.of(textEncoder)
    else
      TokenizerJson.load(
        tokenizerFile.getOrElse(
          throw new IllegalArgumentException(
            "--tokenizer: Qwen3.5's tokenizer.json, for a text encoder not in GGUF"
          )
        )
      )
  private val autoencoder = FluxVae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)
  private val projection = LatentProjection.named(
    LatentProjection.Flux2,
    transformer.config.latentChannels
  )

  private val Prefix = "<|im_start|>user\n"
  private val Suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
  private val MaximumTextTokens = 4096
  private val Shift = 5.0

  /** The block the negative branch runs without. */
  private val UnconditionalSkip = Set(22)

  require(
    encoder.hiddenSize == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not Qwen3.5's ${encoder.hiddenSize}: is the text encoder Qwen3.5-9B?"
  )

  /** The prompt's text for every step, `[L, hidden]`. The template trims the
    * prompt, as Qwen3.5's own does.
    */
  private def text(prompt: String): Tensor = {
    val ids = tokenizer
      .encode(Prefix + prompt.trim + Suffix, addSpecial = false)
      .take(MaximumTextTokens)
    val features =
      ops.allocate(DType.F32, Shape.of(ids.length, encoder.hiddenSize))
    val out =
      ops.allocate(DType.F32, Shape.of(ids.length, transformer.config.hidden))
    try {
      encoder.lastHiddenState(ids, features)
      transformer.embedText(features, out)
      out
    } catch {
      case error: Throwable =>
        ops.release(out)
        throw error
    } finally ops.release(features)
  }

  /** `image`'s packed latents. */
  private def latents(image: BufferedImage): Tensor = {
    val pixels = ops.fromFloats(
      Shape.of(image.getHeight, image.getWidth, 3),
      Images.pixels(image)
    )
    try autoencoder.encode(pixels)
    finally ops.release(pixels)
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    require(
      request.width % 16 == 0 && request.height % 16 == 0,
      s"${request.width} × ${request.height}: MOSS-Image takes multiples of 16"
    )
    val (loras, problems) = loraFiles.open(request.loras)
    LoraReport.print(loras, problems, transformer.useLoras(loras))
    val grid = Grid(request.height / 16, request.width / 16)
    val features = transformer.config.latentChannels
    val guided = request.cfgScale != 1f
    val held = new TensorScope(ops)
    try {
      val conditional = held.keep(text(request.prompt))
      val unconditional =
        Option.when(guided)(held.keep(text(request.negativePrompt)))
      // a request's own levels (a turbo LoRA's) in place of the schedule
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule.snapped(request.steps, request.shift.getOrElse(Shift))
      )
      val steps = sigmas.size - 1
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(grid.tokens * features)(Noise.gaussian(random))
      val x = held.fromFloats(Shape.of(grid.tokens, features), noise)
      val inpainting =
        request.mask.filter(_ => request.initImage.isDefined).map { mask =>
          new Inpainting(
            ops,
            x.shape,
            Inpainting.weights(mask, grid.height, grid.width, features),
            noise
          )
        }
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(steps, request.strength)
        val encoded =
          latents(Images.resized(init, request.width, request.height))
        try {
          FlowSampler.started(ops, x, encoded, sigmas(start), inpainting)
        } finally ops.release(encoded)
        start
      }
      FlowSampler.run(
        ops,
        x,
        sigmas,
        first,
        steps,
        velocityAt = (i, into) =>
          transformer
            .velocity(x, grid, conditional, sigmas(i), Set.empty, into),
        guidance = unconditional.map(uncond =>
          FlowSampler.Guidance(
            request.cfgScale,
            (i, into) =>
              transformer.velocity(
                x,
                grid,
                uncond,
                sigmas(i),
                UnconditionalSkip,
                into
              )
          )
        ),
        inpainting = inpainting,
        onVelocity = (step, steps, velocity, sigma) =>
          projection.foreach(drawn =>
            preview(
              step,
              steps,
              () =>
                drawn.estimate(
                  ops.toFloats(x),
                  ops.toFloats(velocity),
                  sigma,
                  grid.height,
                  grid.width
                )
            )
          ),
        progress = progress
      )
      inpainting.foreach(_.release())
      Images.decoding("the image")
      val rgb = autoencoder.decode(x, grid.height)
      try Images.toImage(ops.toFloats(rgb), request.width, request.height)
      finally ops.release(rgb)
    } finally held.close()
  }

  def close(): Unit = {
    loraFiles.close()
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}
