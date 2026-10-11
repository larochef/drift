package drift.runner.diffusion

import drift.runner.formats.Gguf
import drift.runner.lora.{LoraFiles, LoraReport}
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{GgufTokenizer, Tokenizer}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** Krea 2 end to end (diffusers' `Krea2Pipeline`): the prompt through
  * Qwen3-VL's text model in Krea's template (the residual stream after every
  * third layer, the template's 34 prefix tokens dropped), fused once by the
  * transformer; noise in packed 2 × 2 patches of 16 latent channels, denoised
  * by Euler steps on a flow-matching schedule; the latents unpacked and decoded
  * by the Wan 2.1 VAE. img2img: the init image encoded by the same VAE, packed,
  * and mixed with the noise at the schedule's step `steps − ⌊steps ×
  * strength⌋`, the steps from there run (as for FLUX.2).
  */
final class Krea2Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path
) extends ImagePipeline {

  def family: String = "Krea 2"
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = false
  def takesLoras: Boolean = true
  override def takesSigmas: Boolean = true
  def takesGuidance: Boolean = false

  private val encoder = Qwen3.open(ops, textEncoder)
  private val tokenizer: Tokenizer = {
    val (file, mapped) = Gguf.open(textEncoder)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
  private val transformer = Krea2.open(ops, diffusionModel)
  private val autoencoder = WanVae.open(ops, vae)

  private val loraFiles = new LoraFiles(ops)
  private val projection = LatentProjection.named(
    LatentProjection.Wan21,
    transformer.config.latentChannels
  )

  /** Makes `loras` the transformer's active set; returns what is left
    * unapplied.
    */
  private def useLoras(loras: Seq[(Path, Float)]): Unit = {
    val (opened, problems) = loraFiles.open(loras)
    LoraReport.print(opened, problems, transformer.useLoras(opened))
  }

  private val Prefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
      "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
  private val Suffix = "<|im_end|>\n<|im_start|>assistant\n"
  private val PrefixTokens = tokenizer.encode(Prefix, addSpecial = false).length
  require(
    PrefixTokens == 34,
    s"Krea 2's template prefix is 34 tokens, not $PrefixTokens"
  )

  /** The encoder layers whose output the transformer fuses: 2, 5, …, 35. */
  private val Taps = (0 until transformer.config.textLayers).map(3 * _ + 2)

  /** The prompt's text for every step, `[L, hidden]` (L past the prefix). */
  private def text(prompt: String): Tensor = {
    val ids = tokenizer.encode(Prefix + prompt + Suffix, addSpecial = false)
    val width = encoder.config.hidden.toLong
    val all =
      ops.allocate(DType.F32, Shape.of(Taps.size.toLong * ids.length, width))
    val length = ids.length - PrefixTokens
    val taps =
      ops.allocate(DType.F32, Shape.of(Taps.size.toLong * length, width))
    try {
      encoder.encode(ids, Taps, all)
      Taps.indices.foreach(i =>
        ops.copy(
          all.rows(i.toLong * ids.length + PrefixTokens, length),
          taps.rows(i.toLong * length, length)
        )
      )
      val out =
        ops.allocate(DType.F32, Shape.of(length, transformer.config.hidden))
      transformer.encodeText(taps, out)
      out
    } finally {
      ops.release(all)
      ops.release(taps)
    }
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    require(
      request.width % 16 == 0 && request.height % 16 == 0,
      s"${request.width} × ${request.height}: Krea 2 takes multiples of 16"
    )
    val (gridHeight, gridWidth) = (request.height / 16, request.width / 16)
    val channels = transformer.config.latentChannels
    val tokens = gridHeight * gridWidth
    val guided = request.cfgScale != 1f
    useLoras(request.loras)
    val conditional = text(request.prompt)
    val unconditional = Option.when(guided)(text(request.negativePrompt))
    val random = new SplittableRandom(request.seed)
    val noise = Array.fill(tokens * channels)(Noise.gaussian(random))
    val latents = ops.fromFloats(Shape.of(tokens, channels), noise)
    val inpainting =
      request.mask.filter(_ => request.initImage.isDefined).map { mask =>
        new Inpainting(
          ops,
          latents.shape,
          Inpainting.weights(mask, gridHeight, gridWidth, channels),
          noise
        )
      }
    try {
      // a request's own levels (a turbo LoRA's) in place of the schedule
      val sigmas = request.sigmas.getOrElse(
        FlowSchedule.sigmas(
          request.steps,
          request.shift.getOrElse(FlowSchedule.DefaultShift)
        )
      )
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val encoded = initLatents(init, request.width, request.height)
        try {
          FlowSampler.started(ops, latents, encoded, sigmas(start), inpainting)
        } finally ops.release(encoded)
        start
      }
      FlowSampler.run(
        ops,
        latents,
        sigmas,
        first,
        request.steps,
        velocityAt = (i, into) =>
          transformer.velocity(
            latents,
            conditional,
            sigmas(i),
            gridHeight,
            gridWidth,
            into
          ),
        guidance = unconditional.map(uncond =>
          FlowSampler.Guidance(
            request.cfgScale,
            (i, into) =>
              transformer.velocity(
                latents,
                uncond,
                sigmas(i),
                gridHeight,
                gridWidth,
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
                  ops.toFloats(latents),
                  ops.toFloats(velocity),
                  sigma,
                  gridHeight,
                  gridWidth
                )
            )
          ),
        progress = progress
      )
      val image = ops.allocate(
        DType.F32,
        Shape.of(2L * gridHeight, 2L * gridWidth, channels / 4)
      )
      try {
        ops.unpackPatches(latents, gridHeight, 2, image)
        Images.decoding("the image")
        val rgb = autoencoder.decode(image)
        try Images.toImage(ops.toFloats(rgb), request.width, request.height)
        finally ops.release(rgb)
      } finally ops.release(image)
    } finally {
      inpainting.foreach(_.release())
      (Seq(conditional, latents) ++ unconditional).foreach(ops.release)
    }
  }

  /** `image`'s latents as the transformer takes them, packed 2 × 2: `[H/16 ×
    * W/16, 64]`.
    */
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
    loraFiles.close()
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}
