package drift.runner.diffusion

import drift.runner.decode.Prompt
import drift.runner.formats.Gguf
import drift.runner.models.*
import drift.runner.models.Flux2.Grid
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}
import drift.runner.vision.{ImageSizing, PreparedImage}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** Mage-Flow end to end (microsoft/Mage's `MageFlowPipeline`): the prompt in
  * its template through Qwen3-VL's text model (the last hidden state, the
  * system turn dropped); noise of the VAE's 128 channels, one token per latent,
  * denoised by Euler steps on the flow-matching schedule at a fixed shift of 6;
  * the latents decoded by the Mage VAE.
  *   - Guidance: `--cfg-scale` above 1 runs the negative prompt too, an empty
  *     prompt encoded as a space. Turbo runs 4 steps at 1.
  *   - References (the Edit models), with Qwen3-VL's vision tower (`vision`, an
  *     mmproj): each read by the text encoder in the edit template as `Image
  *     N:` and its image tokens, its longest side held to 384; and each scaled
  *     to the output's size, encoded by the VAE (drawn from its posterior, as
  *     the released pipeline encodes) and run after the target's tokens at
  *     every step, kept clean. The text encoder counts its positions straight
  *     through the image tokens, as the released one is called.
  *   - img2img: the init image's latents (the posterior's mean, so that what a
  *     mask keeps comes back as it went in) mixed with the noise at the
  *     schedule's step `steps − ⌊steps × strength⌋`, as for FLUX.2 (the
  *     released pipeline has none), with a mask of the part to repaint.
  * The released pipeline also marks its starting noise with a watermark and
  * screens prompts with the text encoder; neither is part of the model, and
  * neither is done here.
  */
final class MageFlowPipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    vision: Option[Path]
) extends ImagePipeline {

  def family: String = "Mage-Flow"
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = tower.isDefined
  def takesLoras: Boolean = true
  override def takesSigmas: Boolean = true
  override def ownShift: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen3.open(ops, textEncoder)
  private val tokenizer: Tokenizer = {
    val (file, mapped) = Gguf.open(textEncoder)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
  private val transformer = MageFlow.open(ops, diffusionModel)
  private val autoencoder = MageVae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)
  private val tower = vision.map(QwenVision.open(ops, _))
  private val sizing =
    tower.map(t => ImageSizing.qwen(t.config.patch, t.config.merge))

  require(
    encoder.config.hidden == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not the encoder's ${encoder.config.hidden}"
  )
  require(
    autoencoder.channels == transformer.config.latentChannels,
    s"the transformer takes ${transformer.config.latentChannels} latent channels, " +
      s"not the VAE's ${autoencoder.channels}"
  )

  /** A template's system turn and the user turn's opening, and what closes the
    * user turn; the system turn's tokens are dropped from what the transformer
    * reads.
    */
  final private case class Template(opening: String, dropped: Int) {
    require(
      tokenizer.encode(opening, addSpecial = false).length == dropped,
      s"a Mage-Flow template opens with $dropped tokens, not " +
        tokenizer.encode(opening, addSpecial = false).length
    )
  }
  private val Closing = "<|im_end|>\n<|im_start|>assistant\n"
  private val Describe = Template(
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
      "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n",
    34
  )
  private val Edit = Template(
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, " +
      "background), then explain how the user's text instruction should alter or modify the image. Generate a new " +
      "image that meets the user's requirements while maintaining consistency with the original input where " +
      "appropriate.<|im_end|>\n<|im_start|>user\n",
    64
  )

  /** The text encoder's copy of a reference: its longest side at most this. */
  private val SeenSide = 384

  /** The flow shift (the released `static_shift`). */
  private val Shift = 6.0

  /** A reference as both halves of the model read it: the vision tower's tokens
    * and its latents on the target's grid.
    */
  final private case class Reference(
      image: PreparedImage,
      seen: VisionOutput,
      latents: Tensor
  ) {
    def release(): Unit = {
      ops.release(seen.tokens)
      seen.deepstack.foreach(ops.release)
      ops.release(latents)
    }
  }

  /** `picture`'s latents at `width × height`: drawn from the posterior with
    * `random`, else its mean.
    */
  private def latentsOf(
      picture: BufferedImage,
      width: Int,
      height: Int,
      random: Option[SplittableRandom]
  ): Tensor = {
    val pixels = ops.fromFloats(
      Shape.of(height, width, 3),
      Images.pixels(Images.resized(picture, width, height))
    )
    try autoencoder.encode(pixels, random)
    finally ops.release(pixels)
  }

  private def reference(
      picture: BufferedImage,
      index: Int,
      width: Int,
      height: Int,
      random: SplittableRandom
  ): Reference = {
    val (tower, sizing) = (this.tower.get, this.sizing.get)
    val longest = math.max(picture.getWidth, picture.getHeight)
    val small =
      if (longest <= SeenSide) picture
      else
        Images.resized(
          picture,
          math
            .max(1, math.round(picture.getWidth * SeenSide.toFloat / longest)),
          math
            .max(1, math.round(picture.getHeight * SeenSide.toFloat / longest))
        )
    val image = sizing.prepare(small, s"reference $index")
    Reference(
      image,
      tower.encode(image),
      latentsOf(picture, width, height, Some(random))
    )
  }

  /** The prompt's text for every step, `[L, hidden]` (L past the system turn),
    * reading `references`.
    */
  private def text(prompt: String, references: Seq[Reference]): Tensor = {
    val instruction = if (prompt.isEmpty) " " else prompt
    val template = if (references.isEmpty) Describe else Edit
    val tags = references.indices
      .map(i => s"Image ${i + 1}: <|vision_start|><|image_pad|><|vision_end|>")
      .mkString
    val rendered = tokenizer.encode(
      template.opening + tags + instruction + Closing,
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
        // every token at its own index on the three axes
        Array.tabulate(3 * ids.length)(_ % ids.length),
        placed.map((p, r) => GivenRows(p.at, r.seen.tokens)),
        references.head.seen.deepstack.indices.map(k =>
          placed.map((p, r) => GivenRows(p.at, r.seen.deepstack(k)))
        )
      )
    )
    val width = encoder.config.hidden.toLong
    val length = ids.length - template.dropped
    val all = ops.allocate(DType.F32, Shape.of(ids.length, width))
    try {
      encoder.lastHiddenState(ids, all, images = seen)
      val out =
        ops.allocate(DType.F32, Shape.of(length, transformer.config.hidden))
      transformer.embedText(all.rows(template.dropped, length), out)
      out
    } finally ops.release(all)
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val scale = autoencoder.scale
    require(
      request.width % scale == 0 && request.height % scale == 0,
      s"${request.width} × ${request.height}: Mage-Flow takes multiples of $scale"
    )
    val (loras, problems) = loraFiles.open(request.loras)
    (problems ++ transformer
      .useLoras(loras)
      .map(target => s"a target of no weight: $target"))
      .foreach(problem => println(s"[WARN] LoRA left unapplied: $problem"))
    val grid = Grid(request.height / scale, request.width / scale)
    val channels = transformer.config.latentChannels
    val guided = request.cfgScale != 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    val references = mutable.ArrayBuffer.empty[Reference]
    try {
      if (request.references.nonEmpty && tower.isEmpty)
        throw new IllegalArgumentException(
          "Mage-Flow edits with its text encoder's vision tower: pass --llm_vision <mmproj>"
        )
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(grid.tokens * channels)(Images.gaussian(random))
      // the posteriors' draws, apart from the noise's
      val posterior = new SplittableRandom(request.seed ^ 0x6d616765L)
      request.references.zipWithIndex.foreach((picture, i) =>
        references += reference(
          picture,
          i,
          request.width,
          request.height,
          posterior
        )
      )
      val conditional = hold(text(request.prompt, references.toSeq))
      val unconditional = Option.when(guided)(
        hold(text(request.negativePrompt, references.toSeq))
      )
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule
          .sigmas(request.steps, math.log(request.shift.getOrElse(Shift)))
      )
      val x = hold(ops.fromFloats(Shape.of(grid.tokens, channels), noise))
      val inpainting =
        request.mask.filter(_ => request.initImage.isDefined).map { mask =>
          new Inpainting(
            ops,
            x.shape,
            Inpainting.weights(mask, grid.height, grid.width, channels),
            noise
          )
        }
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val encoded =
          latentsOf(init, request.width, request.height, None)
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
      val clean = references.toSeq.map(r => (r.latents, grid))
      val steps = request.steps - first
      progress(0, steps)
      (first until request.steps).foreach { i =>
        transformer.velocity(x, grid, clean, conditional, sigmas(i), velocity)
        for {
          uncond <- unconditional
          v <- other
        } {
          transformer.velocity(x, grid, clean, uncond, sigmas(i), v)
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
      val rgb = autoencoder.decode(x.view(grid.height, grid.width, channels))
      try Images.toImage(ops.toFloats(rgb), request.width, request.height)
      finally ops.release(rgb)
    } finally {
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
