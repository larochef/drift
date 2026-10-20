package drift.runner.diffusion

import drift.runner.lora.{LoraFiles, LoraReport}
import drift.runner.models.*
import drift.runner.models.Flux2.Grid
import drift.runner.ops.{Ops, TensorScope}
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** FLUX 3 Image Klein end to end (BFL's `flux3.pipeline`), the step-distilled
  * model or the base one, told apart by the transformer's flow-map embedding:
  * the prompt in Qwen3-VL's chat template, the residual stream after every
  * fourth of its layers (4 to 32) side by side, padded to a multiple of 80
  * rows, through the transformer's `txt_in` once; noise in packed 2 × 2 patches
  * of the FLUX.2 VAE's 32 channels, denoised by Euler steps on `linspace(1, 0,
  * steps + 1)` shifted by 16 at every size (μ = ln 16); the flow map is told
  * each step's next level, the base model runs CFG; the latents decoded by the
  * FLUX.2 VAE.
  *   - img2img: the init image's latents mixed with the noise at the schedule's
  *     step `steps − ⌊steps × strength⌋`, the steps from there run;
  *   - references: one at the output's size is taken as it is (its tokens
  *     overlay the target's), another fitted under 4 megapixels; several each
  *     fitted under a megapixel and cropped to the 16-pixel grid. Encoded once
  *     and given to every step.
  */
final class Flux3Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path
) extends ImagePipeline {

  // the products' activations in BF16, as the reference runs: Qwen3-VL's
  // residual stream reaches 10⁴ and the shared blocks' activations pass F16's
  // range
  ops.wideProducts = true
  private val transformer = Flux3.open(ops, diffusionModel)

  def family: String =
    if (transformer.config.flowMap) "FLUX 3 Image Klein"
    else "FLUX 3 Image Klein Base"
  override def defaultSteps: Int = if (transformer.config.flowMap) 16 else 50
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = true
  def takesLoras: Boolean = true
  override def ownShift: Boolean = true
  override def takesSigmas: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen3.open(ops, textEncoder)
  private val tokenizer: Tokenizer = GgufTokenizer.of(textEncoder)
  private val autoencoder = FluxVae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)
  private val projection = LatentProjection.named(
    LatentProjection.Flux2,
    transformer.config.latentChannels
  )

  private val Prefix = "<|im_start|>user\n"
  private val Suffix = "<|im_end|>\n<|im_start|>assistant\n"
  private val Taps = Seq(4, 8, 12, 16, 20, 24, 28, 32)
  private val PadMultiple = 80
  private val MaximumTextTokens = 4096

  /** `<|endoftext|>`, at position 1 as transformers places masked tokens. */
  private val PadId = 151643
  private val PadPosition = 1

  /** The shift at every size, as μ: BFL's `alpha` of 16. */
  private val Shift = math.log(16)

  private val MaximumReferences = 10
  private val SingleReferencePixels = 4.0 * 1024 * 1024
  private val ReferencePixels = 1024.0 * 1024
  private val MinimumSide = 64
  private val MaximumSide = 4096

  require(
    Taps.size * encoder.config.hidden == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not ${Taps.size} taps of ${encoder.config.hidden}"
  )

  /** The prompt's text for every step, `[L, hidden]`, `L` the template's tokens
    * rounded up to a multiple of 80: the pad tokens are text the transformer
    * reads.
    */
  private def text(prompt: String): Tensor = {
    val ids = tokenizer
      .encode(Prefix + prompt + Suffix, addSpecial = false)
      .take(MaximumTextTokens)
    val tokens = math.min(
      (ids.length + PadMultiple - 1) / PadMultiple * PadMultiple,
      MaximumTextTokens
    )
    val width = encoder.config.hidden.toLong
    val tapped =
      ops.allocate(DType.F32, Shape.of(Taps.size.toLong * tokens, width))
    val features = ops.allocate(DType.F32, Shape.of(tokens, Taps.size * width))
    val out =
      ops.allocate(DType.F32, Shape.of(tokens, transformer.config.hidden))
    try {
      encoder.encode(
        ids,
        Taps,
        tapped,
        tokens - ids.length,
        PadId,
        padPosition = Some(PadPosition)
      )
      ops.concatColumns(
        Taps.indices.map(i => tapped.rows(i.toLong * tokens, tokens)),
        features
      )
      transformer.embedText(features, out)
      out
    } catch {
      case error: Throwable =>
        ops.release(out)
        throw error
    } finally {
      ops.release(tapped)
      ops.release(features)
    }
  }

  /** `image`'s packed latents and their grid. */
  private def latents(image: BufferedImage): (Tensor, Grid) = {
    val pixels = ops.fromFloats(
      Shape.of(image.getHeight, image.getWidth, 3),
      Images.pixels(image)
    )
    try
      (
        autoencoder.encode(pixels),
        Grid(
          image.getHeight / autoencoder.patch,
          image.getWidth / autoencoder.patch
        )
      )
    finally ops.release(pixels)
  }

  /** The references as the transformer reads them (BFL's `prepare_reference`
    * and `prepare_multi_reference`).
    */
  private def prepared(
      references: Seq[BufferedImage],
      width: Int,
      height: Int
  ): Seq[BufferedImage] = references match {
    case Seq(only) if only.getWidth == width && only.getHeight == height =>
      Seq(only)
    case Seq(only) =>
      val (w, h) = (only.getWidth.toDouble, only.getHeight.toDouble)
      val scale = Seq(
        1.0,
        math.sqrt(SingleReferencePixels / (w * h)),
        MaximumSide / math.max(w, h)
      ).min
      def side(value: Double) =
        math.min(
          MaximumSide,
          math.max(MinimumSide, math.round(value * scale / 16).toInt * 16)
        )
      Seq(Images.resized(only, side(w), side(h)))
    case several =>
      several.map { reference =>
        val (w, h) = (reference.getWidth, reference.getHeight)
        val scale =
          math.min(1.0, math.sqrt(ReferencePixels / (w.toDouble * h)))
        val (fittedWidth, fittedHeight) =
          if (scale < 1) ((w * scale).toInt, (h * scale).toInt) else (w, h)
        val fitted = Images.resized(reference, fittedWidth, fittedHeight)
        val (croppedWidth, croppedHeight) =
          (fittedWidth / 16 * 16, fittedHeight / 16 * 16)
        fitted.getSubimage(
          (fittedWidth - croppedWidth) / 2,
          (fittedHeight - croppedHeight) / 2,
          croppedWidth,
          croppedHeight
        )
      }
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    require(
      request.width % 16 == 0 && request.height % 16 == 0,
      s"${request.width} × ${request.height}: FLUX 3 takes multiples of 16"
    )
    require(
      request.references.size <= MaximumReferences,
      s"${request.references.size} references: FLUX 3 takes at most $MaximumReferences"
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
      val references =
        prepared(request.references, request.width, request.height).map {
          reference =>
            val (tensor, g) = latents(reference)
            (held.keep(tensor), g)
        }
      // a request's own levels (a turbo LoRA's) in place of the schedule
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule.shifted(
          (0 to request.steps)
            .map(i => 1f - i.toFloat / request.steps),
          request.shift.getOrElse(Shift)
        )
      )
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
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val (encoded, _) =
          latents(Images.resized(init, request.width, request.height))
        try {
          FlowSampler.started(ops, x, encoded, sigmas(start), inpainting)
        } finally ops.release(encoded)
        start
      }
      // the flow map is told where each step lands
      def next(i: Int) =
        Option.when(transformer.config.flowMap)(sigmas(i + 1))
      FlowSampler.run(
        ops,
        x,
        sigmas,
        first,
        request.steps,
        velocityAt = (i, into) =>
          transformer.velocity(
            x,
            grid,
            references,
            conditional,
            sigmas(i),
            next(i),
            into
          ),
        guidance = unconditional.map(uncond =>
          FlowSampler.Guidance(
            request.cfgScale,
            (i, into) =>
              transformer.velocity(
                x,
                grid,
                references,
                uncond,
                sigmas(i),
                next(i),
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
