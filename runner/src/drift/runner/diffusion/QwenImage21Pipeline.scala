package drift.runner.diffusion

import drift.runner.decode.Prompt
import drift.runner.formats.Gguf
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}
import drift.runner.vision.{ImageSizing, PreparedImage}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** Qwen Image 2.1 end to end (diffusers' `QwenImage21Pipeline`): the prompt in
  * its template through Qwen3-VL's text model (the last layer's residual
  * stream, the system turn dropped), run once through the transformer as a
  * prefix; noise of the VAE's 64 channels, one token per latent pixel, denoised
  * by Euler steps on the flow-matching schedule shifted for the image's size
  * and stretched to end at 0.02; the latents decoded to RGBA by its VAE.
  *   - Guidance: `--cfg-scale` above 1 runs the negative prompt too (the
  *     official pipeline's `true_cfg_scale`), empty prompts encoded as a space.
  *   - img2img: the init image's latents mixed with the noise at the schedule's
  *     step `steps − ⌊steps × strength⌋`, as for FLUX.2 (the official pipeline
  *     has no img2img).
  *   - References (editing), with Qwen3-VL's vision tower (`vision`, an
  *     mmproj): each scaled to about a megapixel (sides multiples of 32), read
  *     by the text encoder as `<imageN>` and its image tokens in the user turn
  *     (with deepstack), and its latents taking those tokens' places in the
  *     transformer's prefix.
  */
final class QwenImage21Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    vision: Option[Path]
) extends ImagePipeline {

  def family: String = "Qwen Image 2.1"
  def takesInitImage: Boolean = true
  def takesReferences: Boolean = tower.isDefined
  def takesLoras: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen3.open(ops, textEncoder)
  private val tokenizer: Tokenizer = {
    val (file, mapped) = Gguf.open(textEncoder)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
  private val transformer = QwenImage21.open(ops, diffusionModel)
  private val autoencoder = QwenImage21Vae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)
  private val tower = vision.map(QwenVision.open(ops, _))
  private val sizing =
    tower.map(t => ImageSizing.qwen(t.config.patch, t.config.merge))

  require(
    encoder.config.hidden == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not the encoder's ${encoder.config.hidden}"
  )

  private val System =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
  private val Template =
    ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
  private val SystemTokens = tokenizer.encode(System, addSpecial = false).length
  // the residual stream after the last layer (`encode` taps after k layers)
  private val LastLayer = encoder.config.layers

  /** A reference as both halves of the model read it: the vision tower's tokens
    * and its latents.
    */
  final private case class Reference(
      image: PreparedImage,
      seen: VisionOutput,
      latents: Tensor,
      gridHeight: Int,
      gridWidth: Int
  ) {
    def release(): Unit = {
      ops.release(seen.tokens)
      seen.deepstack.foreach(ops.release)
      ops.release(latents)
    }
  }

  /** `picture` at about a megapixel, sides multiples of 32, as diffusers'
    * `calculate_dimensions` sizes a condition image.
    */
  private def reference(picture: BufferedImage, index: Int): Reference = {
    val (tower, sizing) = (this.tower.get, this.sizing.get)
    val (width, height) = Images.referenceSize(
      picture.getWidth,
      picture.getHeight,
      1024,
      1024,
      2 * autoencoder.scale
    )
    val resized = Images.resized(picture, width, height)
    val image = sizing.prepare(resized, s"reference $index")
    val seen = tower.encode(image)
    val pixels = ops.fromFloats(
      Shape.of(height, width, autoencoder.imageChannels),
      Images.pixels(resized, alpha = true)
    )
    val (gridHeight, gridWidth) =
      (height / autoencoder.scale, width / autoencoder.scale)
    val latents =
      try autoencoder.encode(pixels)
      finally ops.release(pixels)
    Reference(
      image,
      seen,
      latents.view(gridHeight * gridWidth, transformer.config.latentChannels),
      gridHeight,
      gridWidth
    )
  }

  /** The prompt's prefix for images of `imageTokens`, reading `references`. */
  private def prefix(
      prompt: String,
      imageTokens: Int,
      references: Seq[Reference]
  ): QwenImage21Prefix = {
    val text = if (prompt.isEmpty) " " else prompt
    val tags = references.indices
      .map(i => s"<image${i + 1}><|vision_start|><|image_pad|><|vision_end|>")
      .mkString(" ")
    val rendered = tokenizer.encode(
      System + Template._1 + tags + text + Template._2,
      addSpecial = false
    )
    val tokens =
      if (references.isEmpty) Prompt.text(rendered)
      else
        Prompt.withImages(
          rendered,
          tokenizer
            .id("<|image_pad|>")
            .getOrElse(
              throw new IllegalStateException("no <|image_pad|> token")
            ),
          references.map(_.image)
        )
    val ids = tokens.ids
    val placed = tokens.images.zip(references)
    val seen = Option.when(references.nonEmpty)(
      SeenImages(
        tokens.positions,
        placed.map((p, r) => GivenRows(p.at, r.seen.tokens)),
        references.head.seen.deepstack.indices.map(k =>
          placed.map((p, r) => GivenRows(p.at, r.seen.deepstack(k)))
        )
      )
    )
    val width = encoder.config.hidden.toLong
    val length = ids.length - SystemTokens
    val all = ops.allocate(DType.F32, Shape.of(ids.length, width))
    val kept = ops.allocate(DType.F32, Shape.of(length, width))
    try {
      encoder.encode(ids, Seq(LastLayer), all, images = seen)
      ops.copy(all.rows(SystemTokens, length), kept)
      transformer.prefix(
        kept,
        imageTokens,
        placed.map((p, r) =>
          PrefixReference(
            p.at - SystemTokens,
            p.image.tokens,
            r.latents,
            r.gridHeight,
            r.gridWidth
          )
        )
      )
    } finally {
      ops.release(all)
      ops.release(kept)
    }
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val scale = autoencoder.scale
    require(
      request.width % scale == 0 && request.height % scale == 0,
      s"${request.width} × ${request.height}: Qwen Image 2.1 takes multiples of $scale"
    )
    val (loras, problems) = loraFiles.open(request.loras)
    (problems ++ transformer
      .useLoras(loras)
      .map(target => s"a target of no weight: $target"))
      .foreach(problem => println(s"[WARN] LoRA left unapplied: $problem"))
    val (gridHeight, gridWidth) =
      (request.height / scale, request.width / scale)
    val tokens = gridHeight * gridWidth
    val channels = transformer.config.latentChannels
    val guided = request.cfgScale != 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    val prefixes = mutable.ArrayBuffer.empty[QwenImage21Prefix]
    def keep(prefix: QwenImage21Prefix) = {
      prefixes += prefix
      prefix
    }
    val references = mutable.ArrayBuffer.empty[Reference]
    try {
      if (request.references.nonEmpty && tower.isEmpty)
        throw new IllegalArgumentException(
          "Qwen Image 2.1 edits with its text encoder's vision tower: pass --llm_vision <mmproj>"
        )
      request.references.zipWithIndex.foreach((picture, i) =>
        references += reference(picture, i)
      )
      val conditional = keep(prefix(request.prompt, tokens, references.toSeq))
      val unconditional = Option.when(guided)(
        keep(prefix(request.negativePrompt, tokens, references.toSeq))
      )
      val sigmas = FlowSchedule.stretched(
        FlowSchedule.sigmas(
          request.steps,
          FlowSchedule.qwenImage21Shift(tokens)
        ),
        0.02
      )
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(tokens * channels)(Images.gaussian(random))
      val x = hold(ops.fromFloats(Shape.of(tokens, channels), noise))
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val pixels = ops.fromFloats(
          Shape.of(request.height, request.width, autoencoder.imageChannels),
          Images.pixels(
            Images.resized(init, request.width, request.height),
            alpha = true
          )
        )
        val encoded =
          try autoencoder.encode(pixels)
          finally ops.release(pixels)
        try {
          val rows = encoded.view(tokens, channels)
          ops.scale(x, sigmas(start), x)
          ops.scale(rows, 1 - sigmas(start), rows)
          ops.add(x, rows, x)
        } finally ops.release(encoded)
        start
      }
      val velocity = hold(ops.allocate(DType.F32, x.shape))
      val other =
        unconditional.map(_ => hold(ops.allocate(DType.F32, x.shape)))
      val steps = request.steps - first
      (first until request.steps).foreach { i =>
        transformer.velocity(
          x,
          conditional,
          sigmas(i),
          gridHeight,
          gridWidth,
          velocity
        )
        for {
          uncond <- unconditional
          v <- other
        } {
          transformer.velocity(x, uncond, sigmas(i), gridHeight, gridWidth, v)
          // v = uncond + scale × (cond − uncond)
          ops.scale(v, 1 - request.cfgScale, v)
          ops.scale(velocity, request.cfgScale, velocity)
          ops.add(velocity, v, velocity)
        }
        ops.scale(velocity, sigmas(i + 1) - sigmas(i), velocity)
        ops.add(x, velocity, x)
        progress(i + 1 - first, steps)
      }
      val rgba =
        autoencoder.decode(x.view(gridHeight, gridWidth, channels))
      try Images.toImage(ops.toFloats(rgba), request.width, request.height)
      finally ops.release(rgba)
    } finally {
      prefixes.foreach(_.close())
      references.foreach(_.release())
      held.foreach(ops.release)
    }
  }

  def close(): Unit = {
    loraFiles.close()
    tower.foreach(_.close())
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}
