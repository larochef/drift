package drift.runner.diffusion

import drift.runner.formats.Gguf
import drift.runner.models.*
import drift.runner.models.Flux2.Grid
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** FLUX.2 end to end, [klein] (diffusers' `Flux2KleinPipeline`) or [dev]
  * (`Flux2Pipeline`), told apart by the transformer's guidance embedding
  * (`Flux2Text`): the prompt in the text encoder's template, the residual
  * stream after three of its layers side by side, 512 rows, through the
  * transformer's `txt_in` once; noise in packed 2 × 2 patches of the VAE's 32
  * channels, denoised by Euler steps on the flow-matching schedule shifted by
  * FLUX.2's μ for the image's size, dev's distilled guidance scale given to
  * every step; the latents decoded by the FLUX.2 VAE.
  *   - img2img: the init image's latents mixed with the noise at the schedule's
  *     step `steps − ⌊steps × strength⌋` (diffusers' `get_timesteps`), the
  *     steps from there run;
  *   - references: each scaled as sd-cpp does before its VAE (about a
  *     megapixel, its shape kept, sides multiples of 16), encoded once, and
  *     given to every step after the target's tokens.
  */
final class Flux2Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path
) extends ImagePipeline {

  private val transformer = Flux2.open(ops, diffusionModel)
  private val reading =
    if (transformer.config.guidance) Flux2Text.Dev else Flux2Text.Klein

  def family: String = reading.family
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = true
  def takesLoras: Boolean = true
  override def takesSigmas: Boolean = true
  def takesGuidance: Boolean = transformer.config.guidance

  private val encoder = reading.open(ops, textEncoder)
  private val tokenizer: Tokenizer = {
    val (file, mapped) = Gguf.open(textEncoder)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
  private val autoencoder = FluxVae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)

  private val TextTokens = 512
  private val taps = reading.taps
  require(
    taps.size * encoder.config.hidden == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not ${taps.size} taps of ${encoder.config.hidden}"
  )

  /** The prompt's text for every step, `[512, hidden]`. */
  private def text(prompt: String): Tensor = {
    val ids = tokenizer
      .encode(
        reading.prefix + prompt + reading.suffix,
        addSpecial = reading.beginning
      )
      .take(TextTokens)
    // pad tokens masked as keys (Klein), or zero rows after the prompt (dev)
    val encoded = if (reading.padId.isDefined) TextTokens else ids.length
    val width = encoder.config.hidden.toLong
    val tapped =
      ops.allocate(DType.F32, Shape.of(taps.size.toLong * encoded, width))
    val features =
      ops.allocate(DType.F32, Shape.of(encoded, taps.size * width))
    val out =
      ops.allocate(DType.F32, Shape.of(TextTokens, transformer.config.hidden))
    try {
      encoder.encode(
        ids,
        taps,
        tapped,
        encoded - ids.length,
        reading.padId.getOrElse(0)
      )
      ops.concatColumns(
        taps.indices.map(i => tapped.rows(i.toLong * encoded, encoded)),
        features
      )
      if (encoded < TextTokens) ops.zero(out)
      transformer.embedText(features, out.rows(0, encoded))
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

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    require(
      request.width % 16 == 0 && request.height % 16 == 0,
      s"${request.width} × ${request.height}: FLUX.2 takes multiples of 16"
    )
    val (loras, problems) = loraFiles.open(request.loras)
    (problems ++ transformer
      .useLoras(loras)
      .map(target => s"a target of no weight: $target"))
      .foreach(problem => println(s"[WARN] LoRA left unapplied: $problem"))
    val grid = Grid(request.height / 16, request.width / 16)
    val features = transformer.config.latentChannels
    val guided = request.cfgScale != 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    try {
      val conditional = hold(text(request.prompt))
      val unconditional =
        Option.when(guided)(hold(text(request.negativePrompt)))
      val references = request.references.map { reference =>
        val (width, height) = Images.referenceSize(
          reference.getWidth,
          reference.getHeight,
          request.width,
          request.height,
          16
        )
        val (tensor, g) = latents(Images.resized(reference, width, height))
        (hold(tensor), g)
      }
      // a request's own levels (a turbo LoRA's) in place of the schedule
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule.sigmas(
          request.steps,
          FlowSchedule.flux2Shift(grid.tokens, request.steps)
        )
      )
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(grid.tokens * features)(Images.gaussian(random))
      val x = hold(ops.fromFloats(Shape.of(grid.tokens, features), noise))
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
          inpainting.foreach(_.remember(encoded))
          ops.scale(x, sigmas(start), x)
          ops.scale(encoded, 1 - sigmas(start), encoded)
          ops.add(x, encoded, x)
        } finally ops.release(encoded)
        start
      }
      val velocity = hold(ops.allocate(DType.F32, x.shape))
      val other =
        unconditional.map(_ => hold(ops.allocate(DType.F32, x.shape)))
      val steps = request.steps - first
      progress(0, steps)
      (first until request.steps).foreach { i =>
        transformer.velocity(
          x,
          grid,
          references,
          conditional,
          sigmas(i),
          request.guidance,
          velocity
        )
        for {
          uncond <- unconditional
          v <- other
        } {
          transformer.velocity(
            x,
            grid,
            references,
            uncond,
            sigmas(i),
            request.guidance,
            v
          )
          // v = uncond + scale × (cond − uncond)
          ops.scale(v, 1 - request.cfgScale, v)
          ops.scale(velocity, request.cfgScale, velocity)
          ops.add(velocity, v, velocity)
        }
        ops.scale(velocity, sigmas(i + 1) - sigmas(i), velocity)
        ops.add(x, velocity, x)
        inpainting.foreach(_.restore(x, sigmas(i + 1)))
        progress(i + 1 - first, steps)
      }
      inpainting.foreach(_.release())
      Images.decoding("the image")
      val rgb = autoencoder.decode(x, grid.height)
      try Images.toImage(ops.toFloats(rgb), request.width, request.height)
      finally ops.release(rgb)
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    loraFiles.close()
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}

/** How a FLUX.2 model reads its prompt: the text encoder, the template around
  * the prompt (after the beginning token when `beginning`), the layers whose
  * output it takes, and how the 512 rows are filled: pad tokens masked as keys
  * (`padId`), else zero rows after the prompt's.
  */
final case class Flux2Text(
    family: String,
    open: (Ops, Path) => DenseDecoder,
    prefix: String,
    suffix: String,
    beginning: Boolean,
    taps: Seq[Int],
    padId: Option[Int]
)

object Flux2Text {

  /** Qwen3 in its chat template, thinking closed; `<|endoftext|>` pads. */
  val Klein: Flux2Text = Flux2Text(
    "FLUX.2 [klein]",
    Qwen3.open,
    "<|im_start|>user\n",
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    beginning = false,
    Seq(9, 18, 27),
    Some(151643)
  )

  /** Mistral Small 3.x in its instruction template with BFL's system prompt,
    * the prompt's rows then zeros (sd-cpp's reading; diffusers pads with masked
    * tokens instead).
    */
  val Dev: Flux2Text = Flux2Text(
    "FLUX.2 [dev]",
    Mistral.open,
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]",
    "[/INST]",
    beginning = true,
    Seq(10, 20, 30),
    None
  )
}
