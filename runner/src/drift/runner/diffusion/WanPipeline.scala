package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.Unigram

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** Wan 2.2 A14B end to end (sd-cpp's `vid_gen` on Wan, diffusers' `WanPipeline`
  * and `WanImageToVideoPipeline` for the conditioning): the prompt through UMT5
  * (its tokens, `</s>` after, the rows past them zeros up to 512), noise in 2 ×
  * 2 patch rows of 16 latent channels per latent frame (4 frames each but the
  * first), denoised by Euler steps on one flow schedule (`linspace(1, 0)`
  * through the flow shift) shared by two experts: the high-noise model for its
  * steps (`--high-noise-steps`, else while σ ≥ the MoE boundary) at its CFG
  * scale, the low-noise one for the rest at its own; then the Wan 2.1 VAE,
  * frame by frame. The I2V checkpoints (36 input channels) read a mask (the
  * first latent frame's when an init image is given) and the VAE's latents of a
  * video that is the init image then zeros, or zeros alone: those rows are made
  * once per size and kept.
  */
final class WanPipeline(
    ops: Ops,
    lowNoiseModel: Path,
    highNoiseModel: Option[Path],
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Path
) extends VideoPipeline {

  def family: String = "Wan 2.2"
  def fps: Int = 16
  def sizeMultiple: Int = 16

  def alignedFrames(frames: Int): Int = (math.max(frames, 1) + 2) / 4 * 4 + 1

  // the products' activations in BF16, as the reference runs
  ops.wideProducts = true
  private val tokenizer = Unigram.load(tokenizerFile)
  private val encoder = Umt5.open(ops, textEncoder)
  private val low = Wan.open(ops, lowNoiseModel)
  private val high = highNoiseModel.map(Wan.open(ops, _))
  private val decoder = WanVideoVae.open(ops, vae)
  private val TextTokens = 512

  require(
    high.forall(_.config == low.config),
    s"the high-noise model is not the low-noise one's shape: ${high.map(_.config)} against ${low.config}"
  )

  override def takesInitImage: Boolean =
    low.config.inChannels != low.config.outChannels

  /** UMT5's rows of `prompt`, zeros past its tokens up to 512. */
  private def encoded(prompt: String): Tensor = {
    val cleaned = prompt.trim.replaceAll("\\s+", " ")
    val ids = tokenizer.encode(cleaned, addSpecial = true).take(TextTokens)
    val rows = encoder.encode(ids)
    try {
      val padded =
        ops.allocate(DType.F32, Shape.of(TextTokens, encoder.width.toLong))
      ops.zero(padded)
      ops.copy(rows, padded.rows(0, ids.length))
      padded
    } finally ops.release(rows)
  }

  /** The I2V rows for a size and an init image, kept for the last size when
    * there is no image.
    */
  private var conditionFor = Option.empty[((Int, Int, Int), Tensor)]

  private def condition(
      width: Int,
      height: Int,
      frames: Int,
      init: Option[BufferedImage]
  ): Tensor = {
    val key = (width, height, frames)
    conditionFor.filter(c => init.isEmpty && c._1 == key).map(_._2).getOrElse {
      val (h, w) = (height / 8, width / 8)
      val latentFrames = (frames - 1) / 4 + 1
      val video = (0 until frames).map { i =>
        val pixels = ops.allocate(DType.F32, Shape.of(height, width, 3))
        init.filter(_ => i == 0) match {
          case Some(image) =>
            val uploaded =
              ops.fromFloats(
                pixels.shape,
                Images.pixels(Images.resized(image, width, height))
              )
            ops.copy(uploaded, pixels)
            ops.release(uploaded)
          case None => ops.zero(pixels)
        }
        pixels
      }
      val latents =
        try decoder.encode(video)
        finally video.foreach(ops.release)
      val tokens = (h / 2) * (w / 2)
      val rows =
        ops.allocate(DType.F32, Shape.of(latentFrames.toLong * tokens, 80))
      try
        latents.zipWithIndex.foreach { (latent, t) =>
          // [mask of 4 ; latents of 16] per pixel, packed 2 × 2
          val mask = ops.fromFloats(
            Shape.of(h, w, 4),
            Array.fill(h * w * 4)(if (init.isDefined && t == 0) 1f else 0f)
          )
          val joined = ops.allocate(DType.F32, Shape.of(h, w, 20))
          ops.concatColumns(
            Seq(mask.view(h.toLong * w, 4), latent.view(h.toLong * w, 16)),
            joined.view(h.toLong * w, 20)
          )
          ops.packPatches(joined, 2, rows.rows(t.toLong * tokens, tokens))
          Seq(mask, joined).foreach(ops.release)
        }
      finally latents.foreach(ops.release)
      if (init.isEmpty) {
        conditionFor.foreach((_, old) => ops.release(old))
        conditionFor = Some(key -> rows)
      }
      rows
    }
  }

  /** `linspace(1, 0, steps + 1)` through the flow shift `s`: `s σ / (1 + (s −
    * 1) σ)`.
    */
  private def sigmas(steps: Int, shift: Double): Array[Float] =
    Array.tabulate(steps + 1) { i =>
      val base = 1.0 - i.toDouble / steps
      (shift * base / (1 + (shift - 1) * base)).toFloat
    }

  def generate(request: VideoRequest, progress: (Int, Int) => Unit): Video = {
    def rounded(side: Int) =
      (side + sizeMultiple - 1) / sizeMultiple * sizeMultiple
    val (width, height) = (rounded(request.width), rounded(request.height))
    val frames = alignedFrames(request.frames)
    val (latentFrames, gridHeight, gridWidth) =
      ((frames - 1) / 4 + 1, height / 16, width / 16)
    val tokens = latentFrames * gridHeight * gridWidth
    val c = low.config
    val held = mutable.ArrayBuffer.empty[Tensor]
    val texts = mutable.ArrayBuffer.empty[WanText]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val highSteps = high.fold(0)(_ => request.highNoiseSteps.getOrElse(-1))
      val total = request.steps + math.max(highSteps, 0)
      val schedule = sigmas(total, request.shift)
      // the high-noise expert's steps: as asked, else while σ ≥ the boundary
      val switch =
        if (high.isEmpty) 0
        else if (highSteps >= 0) highSteps
        else schedule.indexWhere(_ < request.moeBoundary).max(0)
      val condition = Option.when(takesInitImage)(
        this.condition(width, height, frames, request.initImage)
      )
      if (request.initImage.isDefined) condition.foreach(keep)
      val (prompt, negative) =
        (encoded(request.prompt), encoded(request.negativePrompt))
      held ++= Seq(prompt, negative)
      def textsOf(model: Wan, guided: Boolean) = {
        val conditional = model.text(prompt)
        texts += conditional
        val unconditional = Option.when(guided) {
          val t = model.text(negative)
          texts += t
          t
        }
        (conditional, unconditional)
      }
      val highScale = request.highNoiseCfgScale.getOrElse(request.cfgScale)
      val highTexts =
        high.filter(_ => switch > 0).map(textsOf(_, highScale != 1f))
      val lowTexts =
        Option.when(switch < total)(textsOf(low, request.cfgScale != 1f))
      val random = new SplittableRandom(request.seed)
      val latents = keep(
        ops.fromFloats(
          Shape.of(tokens, 4L * c.outChannels),
          Array.fill(tokens * 4 * c.outChannels)(Images.gaussian(random))
        )
      )
      val (velocity, other) =
        (
          keep(ops.allocate(DType.F32, latents.shape)),
          keep(ops.allocate(DType.F32, latents.shape))
        )
      (0 until total).foreach { i =>
        val highNoise = i < switch
        val model = if (highNoise) high.get else low
        val ((conditional, unconditional), scale) =
          if (highNoise) (highTexts.get, highScale)
          else (lowTexts.get, request.cfgScale)
        val timestep = schedule(i) * 1000
        model.velocity(
          latents,
          condition,
          conditional,
          timestep,
          latentFrames,
          gridHeight,
          gridWidth,
          velocity
        )
        unconditional.foreach { uncond =>
          model.velocity(
            latents,
            condition,
            uncond,
            timestep,
            latentFrames,
            gridHeight,
            gridWidth,
            other
          )
          // v = uncond + scale × (cond − uncond)
          ops.scale(other, 1 - scale, other)
          ops.scale(velocity, scale, velocity)
          ops.add(velocity, other, velocity)
        }
        ops.scale(velocity, schedule(i + 1) - schedule(i), velocity)
        ops.add(latents, velocity, latents)
        progress(i + 1, total)
      }
      // patch rows → latent frames [h, w, 16], then the VAE
      val perFrame = gridHeight * gridWidth
      val latentImages = (0 until latentFrames).map { t =>
        val image = keep(
          ops.allocate(
            DType.F32,
            Shape.of(2L * gridHeight, 2L * gridWidth, c.outChannels)
          )
        )
        ops.unpackPatches(
          latents.rows(t.toLong * perFrame, perFrame),
          gridHeight,
          2,
          image
        )
        image
      }
      val started = System.nanoTime()
      val images = mutable.ArrayBuffer.empty[BufferedImage]
      decoder.decode(
        latentImages,
        rgb => images += Images.toImage(ops.toFloats(rgb), width, height)
      )
      println(
        f"decoded ${images.size} frames in ${(System.nanoTime() - started) / 1e9}%.1f s"
      )
      Video(images.toSeq, fps, None)
    } finally {
      texts.foreach(_.release())
      held.foreach(ops.release)
    }
  }

  def close(): Unit = {
    conditionFor.foreach((_, rows) => ops.release(rows))
    decoder.close()
    high.foreach(_.close())
    low.close()
    encoder.close()
  }
}
