package drift.runner.diffusion

import drift.runner.formats.Gguf
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** Nucleus-Image end to end (diffusers' `NucleusMoEImagePipeline`): the prompt
  * as a chat with the pipeline's system turn through Qwen3-VL-8B's text model
  * (the residual stream eight hidden states from the end, nothing dropped),
  * made once into every block's keys and values; noise in packed 2 × 2 patches
  * of 16 latent channels, denoised by Euler steps on the flow-matching
  * schedule, which the released scheduler leaves unshifted; the latents
  * unpacked and decoded by the Wan 2.1 VAE (Qwen Image's).
  *   - Guidance: `--cfg-scale` above 1 runs the negative prompt too (empty by
  *     default), and the guided velocity is brought back, token by token, to
  *     the length of the prompt's own. A base model: 50 steps at 4.
  *   - img2img: the init image's latents mixed with the noise at the schedule's
  *     step `steps − ⌊steps × strength⌋`, as for FLUX.2 (the released pipeline
  *     has none), with a mask of the part to repaint.
  */
final class NucleusPipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path
) extends ImagePipeline {

  def family: String = "Nucleus-Image"
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = false
  def takesLoras: Boolean = false
  override def takesSigmas: Boolean = true
  override def ownShift: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen3.open(ops, textEncoder)
  private val tokenizer: Tokenizer = {
    val (file, mapped) = Gguf.open(textEncoder)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
  private val transformer = Nucleus.open(ops, diffusionModel)
  private val autoencoder = WanVae.open(ops, vae)

  require(
    encoder.config.hidden == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, " +
      s"not the encoder's ${encoder.config.hidden}"
  )

  private val System =
    "You are an image generation assistant. Follow the user's prompt literally. Pay careful attention to " +
      "spatial layout: objects described as on the left must appear on the left, on the right on the right. " +
      "Match exact object counts and assign colors to the correct objects."

  /** The hidden state the transformer reads: transformers' `hidden_states[−8]`,
    * the residual stream seven layers before the last.
    */
  private val Tap = encoder.config.layers - 7

  /** The flow shift: none, as the released scheduler is configured. */
  private val Shift = 1.0

  /** The prompt as every block's keys and values, for a grid. */
  private def text(
      prompt: String,
      gridHeight: Int,
      gridWidth: Int
  ): NucleusText = {
    val ids = tokenizer.encode(
      s"<|im_start|>system\n$System<|im_end|>\n<|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n",
      addSpecial = false
    )
    val features =
      ops.allocate(DType.F32, Shape.of(ids.length, encoder.config.hidden))
    try {
      encoder.encode(ids, Seq(Tap), features)
      transformer.text(features, gridHeight, gridWidth)
    } finally ops.release(features)
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    require(
      request.width % 16 == 0 && request.height % 16 == 0,
      s"${request.width} × ${request.height}: Nucleus-Image takes multiples of 16"
    )
    val (gridHeight, gridWidth) = (request.height / 16, request.width / 16)
    val channels = transformer.config.latentChannels
    val tokens = gridHeight * gridWidth
    val guided = request.cfgScale != 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    val texts = mutable.ArrayBuffer.empty[NucleusText]
    def keep(text: NucleusText) = {
      texts += text
      text
    }
    try {
      val conditional = keep(text(request.prompt, gridHeight, gridWidth))
      val unconditional = Option.when(guided)(
        keep(text(request.negativePrompt, gridHeight, gridWidth))
      )
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(tokens * channels)(Images.gaussian(random))
      val x = hold(ops.fromFloats(Shape.of(tokens, channels), noise))
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule
          .sigmas(request.steps, math.log(request.shift.getOrElse(Shift)))
      )
      val inpainting =
        request.mask.filter(_ => request.initImage.isDefined).map { mask =>
          new Inpainting(
            ops,
            x.shape,
            Inpainting.weights(mask, gridHeight, gridWidth, channels),
            noise
          )
        }
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val encoded = initLatents(init, request.width, request.height)
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
          conditional,
          sigmas(i),
          gridHeight,
          gridWidth,
          velocity
        )
        val step = sigmas(i + 1) - sigmas(i)
        (unconditional, other) match {
          case (Some(uncond), Some(v)) =>
            transformer.velocity(x, uncond, sigmas(i), gridHeight, gridWidth, v)
            val guidedStep = ops.fromFloats(
              x.shape,
              NucleusPipeline.guided(
                ops.toFloats(velocity),
                ops.toFloats(v),
                request.cfgScale,
                channels,
                step
              )
            )
            try ops.add(x, guidedStep, x)
            finally ops.release(guidedStep)
          case _ =>
            ops.scale(velocity, step, velocity)
            ops.add(x, velocity, x)
        }
        inpainting.foreach(_.restore(x, sigmas(i + 1)))
        progress(i + 1 - first, steps)
      }
      inpainting.foreach(_.release())
      val image = ops.allocate(
        DType.F32,
        Shape.of(2L * gridHeight, 2L * gridWidth, channels / 4)
      )
      try {
        ops.unpackPatches(x, gridHeight, 2, image)
        val rgb = autoencoder.decode(image)
        try Images.toImage(ops.toFloats(rgb), request.width, request.height)
        finally ops.release(rgb)
      } finally ops.release(image)
    } finally {
      texts.foreach(_.close())
      held.foreach(ops.release)
    }
  }

  /** `image`'s latents as the transformer takes them, packed 2 × 2. */
  private def initLatents(image: BufferedImage, width: Int, height: Int) = {
    val pixels = ops.fromFloats(
      Shape.of(height, width, 3),
      Images.pixels(Images.resized(image, width, height))
    )
    val encoded =
      try autoencoder.encode(pixels)
      finally ops.release(pixels)
    try {
      val packed = ops.allocate(
        DType.F32,
        Shape.of(
          (height / 16).toLong * (width / 16),
          transformer.config.latentChannels
        )
      )
      ops.packPatches(encoded, 2, packed)
      packed
    } finally ops.release(encoded)
  }

  def close(): Unit = {
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}

object NucleusPipeline {

  /** The guided velocity times `step`: `uncond + scale × (cond − uncond)`, each
    * token's (a row of `channels`) brought back to the length of its `cond`.
    */
  def guided(
      cond: Array[Float],
      uncond: Array[Float],
      scale: Float,
      channels: Int,
      step: Float
  ): Array[Float] = {
    val out = new Array[Float](cond.length)
    (0 until cond.length / channels).foreach { token =>
      val at = token * channels
      var (own, combined) = (0.0, 0.0)
      (at until at + channels).foreach { i =>
        val value = uncond(i) + scale * (cond(i) - uncond(i))
        out(i) = value
        own += cond(i).toDouble * cond(i)
        combined += value.toDouble * value
      }
      val factor = (math.sqrt(own) / math.sqrt(combined) * step).toFloat
      (at until at + channels).foreach(i => out(i) *= factor)
    }
    out
  }
}
