package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.SplittableRandom
import javax.imageio.ImageIO
import scala.collection.mutable

/** LTX 2.5 text to video (diffusers' `LTX25AutoBlocks` `text2video`, with the
  * convolutional VAE): the prompt (Gemma's tokenizer, `<bos>` first, read out
  * of the text encoder's own file) through Gemma 4 to every hidden state, the
  * text features and the transformer file's connectors (1024 rows, the
  * registers after the prompt), noise for the video (128 latent channels per 32
  * × 32 pixels × 8 frames) and the audio (128 per latent, 25 a second),
  * denoised together by Euler steps on the distilled checkpoint's σ list (8
  * steps) or `linspace(1, 1/N, N)` through the resolution's shift; the video
  * latents through the conv VAE; the audio latents through the audio VAE and
  * its vocoders into 48 kHz stereo, when it is given (else the video is
  * silent). The frame rate is a model input (the RoPE's time); frames round
  * down to `8n + 1`, sides up to multiples of 32.
  *
  * Conditions (diffusers' `LTX2Condition*` blocks at strength 1): the init
  * image, the end image and the guides, VAE-encoded (stills first through H.264
  * at CRF 18, as LTX 2.5 learned them). One at frame 0 (a still, or a clip the
  * video continues) replaces the first latent frames; the others are appended
  * as keyframe tokens with their own positions (the end image at the last
  * frame; a clip from a multiple of 8 plus 1, encoded after a throwaway first
  * frame, as ComfyUI's `LTXVAddGuide`). Their tokens stay clean, at timestep 0,
  * and the appended ones are dropped before decoding. The audio is denoised
  * from noise as in text to video.
  *
  * LoRAs apply on every linear of the transformer and its text connectors.
  */
final class LtxPipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    defaultFps: Int,
    audioVae: Option[Path]
) extends VideoPipeline {

  def family: String = "LTX 2.5"
  def fps: Int = defaultFps
  def sizeMultiple: Int = 32
  override def takesInitImage: Boolean = true
  override def takesEndImage: Boolean = true
  override def takesLoras: Boolean = true
  override def takesGuides: Boolean = true
  override def makesSoundtrack: Boolean = true
  override def decodesSoundtrack: Boolean = audioDecoder.isDefined

  def alignedFrames(frames: Int): Int = math.max(frames - 1, 8) / 8 * 8 + 1

  private val tokenizer: Tokenizer = TokenizerJson.read(
    ujson.read(LtxPipeline.rawTensor(textEncoder, "tokenizer_json")),
    s"$textEncoder (tokenizer_json)"
  )
  private val textSource = WeightSource.open(ops, textEncoder)
  private val gemma = Gemma4Text(ops, textSource, "model.")
  private val features =
    new LtxTextFeatures(ops, textSource, gemma.layers + 1, gemma.hidden)
  private val transformer = Ltx2.open(ops, diffusionModel)
  private val (videoConnector, audioConnector) =
    transformer.connectors.getOrElse(
      throw new IllegalArgumentException(
        s"${diffusionModel.getFileName} has no text connectors"
      )
    )
  private val decoder = LtxVideoVae.open(ops, vae)
  private val loraFiles = new LoraFiles(ops)
  private val audioDecoder = audioVae.map(LtxAudio.open(ops, _))
  private val TextRows = 1024
  private val DistilledSigmas =
    Array(1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f,
      0.421875f, 0f)

  /** The prompt's connector rows: the video's and the audio's. */
  private def text(prompt: String): (Tensor, Tensor) = {
    val ids = tokenizer.encode(prompt.trim, addSpecial = true).take(TextRows)
    val states = gemma.encode(ids)
    val (video, audio) =
      try features(states)
      finally ops.release(states)
    try (videoConnector(video, TextRows), audioConnector(audio, TextRows))
    finally Seq(video, audio).foreach(ops.release)
  }

  /** The σ schedule of `steps`: the distilled list for 8, else `linspace(1,
    * 1/N, N)` through the exponential shift μ of `tokens`, then 0.
    */
  private def sigmas(steps: Int, tokens: Int): Array[Float] =
    if (steps == DistilledSigmas.length - 1) DistilledSigmas
    else {
      val mu = {
        val (baseLength, maxLength, baseShift, maxShift) =
          (1024.0, 4096.0, 0.95, 2.05)
        val slope = (maxShift - baseShift) / (maxLength - baseLength)
        tokens * slope + baseShift - slope * baseLength
      }
      Array.tabulate(steps + 1) { i =>
        if (i == steps) 0f
        else {
          val sigma = 1.0 - i * (1.0 - 1.0 / steps) / math.max(steps - 1, 1)
          (math.exp(mu) / (math.exp(mu) + (1 / sigma - 1))).toFloat
        }
      }
    }

  def generate(request: VideoRequest, progress: (Int, Int) => Unit): Video = {
    def rounded(side: Int) =
      (side + sizeMultiple - 1) / sizeMultiple * sizeMultiple
    val (width, height) = (rounded(request.width), rounded(request.height))
    val frames = alignedFrames(request.frames)
    val rate = request.fps.getOrElse(defaultFps)
    val c = transformer.config
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val (opened, problems) =
        loraFiles.open(request.loras.map(lora => lora.path -> lora.multiplier))
      (problems ++ transformer
        .useLoras(opened)
        .map(target => s"a target of no weight: $target")).foreach(problem =>
        println(s"[WARN] LoRA left unapplied: $problem")
      )
      val base = Ltx2Layout(
        frames = (frames - 1) / 8 + 1,
        height = height / 32,
        width = width / 32,
        audioFrames = math.round(frames.toDouble / rate * 25).toInt,
        fps = rate
      )
      val conditions = conditionsOf(request, width, height, frames, base)
      conditions.foreach(condition => held ++= condition.latents)
      val layout =
        base.copy(appended =
          conditions.flatMap(_.positions.toSeq.flatten).toVector
        )
      // each condition's first token: from the start in place, else after the
      // video's and the conditions appended before it
      val perFrame = layout.height.toLong * layout.width
      val starts = conditions
        .scanLeft(layout.generatedTokens.toLong) { (next, condition) =>
          if (condition.positions.isEmpty) next
          else next + condition.latents.size * perFrame
        }
        .zip(conditions)
        .map((next, condition) => if (condition.positions.isEmpty) 0L else next)
      val heldRanges = {
        val inPlace = conditions.filter(_.positions.isEmpty).map(_.latents.size)
        inPlace.maxOption.map(count => (0L, count * perFrame)).toSeq ++
          conditions.zip(starts).collect {
            case (condition, start) if condition.positions.isDefined =>
              (start, condition.latents.size * perFrame)
          }
      }
      val (videoText, audioText) = text(request.prompt)
      held ++= Seq(videoText, audioText)
      val audioCfgScale = request.audioCfgScale.getOrElse(request.cfgScale)
      val guided = request.cfgScale != 1f || audioCfgScale != 1f
      val (modalityScale, audioModalityScale) =
        (request.modalityScale, request.audioModalityScale)
      val negative = Option.when(guided) {
        val (v, a) = text(request.negativePrompt)
        held ++= Seq(v, a)
        (v, a)
      }
      val random = new SplittableRandom(request.seed)
      val video = keep(
        ops.fromFloats(
          Shape.of(layout.videoTokens, c.videoChannels),
          Array.fill(layout.videoTokens * c.videoChannels)(
            Images.gaussian(random)
          )
        )
      )
      // the conditions' clean latents in their tokens, again after each step
      def restore(): Unit =
        conditions.zip(starts).foreach { (condition, start) =>
          condition.latents.zipWithIndex.foreach { (latent, f) =>
            ops.copy(
              latent.view(perFrame, c.videoChannels),
              video.rows(start + f * perFrame, perFrame)
            )
          }
        }
      restore()
      val audio = keep(
        ops.fromFloats(
          Shape.of(layout.audioFrames, c.audioChannels),
          Array.fill(layout.audioFrames * c.audioChannels)(
            Images.gaussian(random)
          )
        )
      )
      val (videoVelocity, audioVelocity) =
        (
          keep(ops.allocate(DType.F32, video.shape)),
          keep(ops.allocate(DType.F32, audio.shape))
        )
      val others = negative.map(_ =>
        (
          keep(ops.allocate(DType.F32, video.shape)),
          keep(ops.allocate(DType.F32, audio.shape))
        )
      )
      val isolatedOut =
        Option.when(modalityScale != 1f || audioModalityScale != 1f)(
          (
            keep(ops.allocate(DType.F32, video.shape)),
            keep(ops.allocate(DType.F32, audio.shape))
          )
        )
      val schedule = sigmas(request.steps, layout.videoTokens)
      val steps = schedule.length - 1
      (0 until steps).foreach { i =>
        val timestep = schedule(i) * 1000
        transformer.velocity(
          layout,
          video,
          audio,
          videoText,
          audioText,
          timestep,
          videoVelocity,
          audioVelocity,
          conditioned = heldRanges
        )
        for {
          (v, a) <- negative
          (ov, oa) <- others
        } transformer.velocity(
          layout,
          video,
          audio,
          v,
          a,
          timestep,
          ov,
          oa,
          conditioned = heldRanges
        )
        isolatedOut.foreach((iv, ia) =>
          transformer.velocity(
            layout,
            video,
            audio,
            videoText,
            audioText,
            timestep,
            iv,
            ia,
            isolated = true,
            conditioned = heldRanges
          )
        )
        // v = cond + (cfg − 1)(cond − uncond) + (m − 1)(cond − isolated),
        // per stream (diffusers' LTX 2 guider, in velocities: its x0 is affine
        // in them at one σ)
        Seq(
          (
            videoVelocity,
            others.map(_._1),
            isolatedOut.map(_._1),
            request.cfgScale,
            modalityScale
          ),
          (
            audioVelocity,
            others.map(_._2),
            isolatedOut.map(_._2),
            audioCfgScale,
            audioModalityScale
          )
        ).foreach { (velocity, uncond, isolated, cfg, modality) =>
          ops.scale(velocity, cfg + modality - 1, velocity)
          uncond.foreach { other =>
            ops.scale(other, 1 - cfg, other)
            ops.add(velocity, other, velocity)
          }
          isolated.foreach { other =>
            ops.scale(other, 1 - modality, other)
            ops.add(velocity, other, velocity)
          }
        }
        val (sigma, next) = (schedule(i), schedule(i + 1))
        if (!request.ancestral || next == 0f)
          Seq(video -> videoVelocity, audio -> audioVelocity).foreach {
            (x, velocity) =>
              ops.scale(velocity, next - sigma, velocity)
              ops.add(x, velocity, x)
          }
        else {
          // ComfyUI's Euler ancestral for flows (η 1): Euler down to σ↓ =
          // σ'² / σ, rescaled to σ' and renoised
          val down = next * next / sigma
          val renoise = math
            .sqrt(
              next * next - down * down * (1 - next) * (1 - next) /
                ((1 - down) * (1 - down))
            )
            .toFloat
          Seq(video -> videoVelocity, audio -> audioVelocity).foreach {
            (x, velocity) =>
              ops.scale(velocity, down - sigma, velocity)
              ops.add(x, velocity, x)
              ops.scale(x, (1 - next) / (1 - down), x)
              val noise = ops.fromFloats(
                x.shape,
                Array.fill(x.shape.elementCount.toInt)(
                  Images.gaussian(random) * renoise
                )
              )
              ops.add(x, noise, x)
              ops.release(noise)
          }
        }
        restore()
        progress(i + 1, steps)
      }
      val latentFrames = (0 until layout.frames).map(t =>
        video
          .rows(t.toLong * perFrame, perFrame)
          .view(layout.height, layout.width, c.videoChannels)
      )
      val started = System.nanoTime()
      val images = mutable.ArrayBuffer.empty[BufferedImage]
      decoder.decode(
        latentFrames,
        rgb => images += Images.toImage(ops.toFloats(rgb), width, height)
      )
      println(
        f"decoded ${images.size} frames in ${(System.nanoTime() - started) / 1e9}%.1f s"
      )
      val soundtrack = audioDecoder.map { audioDecoder =>
        val started = System.nanoTime()
        val samples = audioDecoder.decode(ops.toFloats(audio))
        println(
          f"decoded ${samples.length.toDouble / audioDecoder.channels / audioDecoder.sampleRate}%.2f s of sound in ${(System.nanoTime() - started) / 1e9}%.1f s"
        )
        Soundtrack.fitted(
          samples,
          audioDecoder.channels,
          audioDecoder.sampleRate
        )
      }
      Video(images.toSeq, rate, soundtrack)
    } finally held.foreach(ops.release)
  }

  /** A condition: its normalized latents (`[h, w, channels]` per latent frame,
    * released with the request), in place from the first latent frame or, with
    * `positions`, appended as keyframe tokens there.
    */
  final private case class Condition(
      latents: Seq[Tensor],
      positions: Option[IndexedSeq[Array[Double]]]
  )

  /** The request's conditions, in order: the init image (frame 0), the end
    * image (the last frame), the guides (a negative index from the end).
    */
  private def conditionsOf(
      request: VideoRequest,
      width: Int,
      height: Int,
      frames: Int,
      layout: Ltx2Layout
  ): Seq[Condition] = {
    val placed =
      request.initImage.map(Seq(_) -> 0).toSeq ++
        request.endImage.map(Seq(_) -> (frames - 1)) ++
        request.guides.map { guide =>
          val index =
            if (guide.frameIndex < 0) frames + guide.frameIndex
            else guide.frameIndex
          require(
            index >= 0 && index < frames,
            s"a guide at frame ${guide.frameIndex} of $frames"
          )
          guide.media match {
            case Media.Still(image)     => Seq(image) -> index
            case Media.Clip(clip, _, _) => clip -> index
            case Media.Sound(_)         =>
              throw new IllegalArgumentException(
                s"$family takes stills and clips as guides, not sounds"
              )
          }
        }
    def encoded(images: Seq[BufferedImage]): Seq[Tensor] = {
      val pixels = images.map(image =>
        ops.fromFloats(
          Shape.of(height, width, 3),
          Images.pixels(LtxPipeline.covered(image, width, height))
        )
      )
      try decoder.encode(pixels)
      finally pixels.foreach(ops.release)
    }
    def still(image: BufferedImage) =
      encoded(Seq(LtxPipeline.compressed(image)))
    placed.map { (images, index) =>
      if (index == 0) {
        // in place: a still, or a clip trimmed to 8n + 1 frames
        val kept = (math.min(images.size, frames) - 1) / 8 * 8 + 1
        Condition(
          if (kept == 1) still(images.head) else encoded(images.take(kept)),
          None
        )
      } else if (images.size < 9) {
        // one frame, at [index, index + 1)
        Condition(
          still(images.head),
          Some(
            Ltx2Layout.keyframe(
              1,
              layout.height,
              layout.width,
              index,
              true,
              layout.fps
            )
          )
        )
      } else {
        // a clip from the multiple of 8 plus 1 at or before its index, as many
        // whole latent frames as fit, encoded after a throwaway first frame
        val start = (index - 1) / 8 * 8 + 1
        val count = math.min(images.size, frames - start) / 8
        if (count == 0)
          Condition(
            still(images.head),
            Some(
              Ltx2Layout.keyframe(
                1,
                layout.height,
                layout.width,
                index,
                true,
                layout.fps
              )
            )
          )
        else {
          val latents = encoded(images.head +: images.take(count * 8))
          ops.release(latents.head)
          Condition(
            latents.tail,
            Some(
              Ltx2Layout.keyframe(
                count,
                layout.height,
                layout.width,
                start,
                false,
                layout.fps
              )
            )
          )
        }
      }
    }
  }

  def close(): Unit = {
    loraFiles.close()
    audioDecoder.foreach(_.close())
    decoder.close()
    transformer.close()
    features.release()
    gemma.close()
  }
}

object LtxPipeline {

  /** `image` scaled to cover `width × height` (bilinear) and centre-cropped to
    * it, as diffusers' and ComfyUI's conditions are.
    */
  def covered(image: BufferedImage, width: Int, height: Int): BufferedImage =
    if (image.getWidth == width && image.getHeight == height) image
    else {
      val scale =
        math.max(
          height.toDouble / image.getHeight,
          width.toDouble / image.getWidth
        )
      val (scaledWidth, scaledHeight) = (
        math.ceil(image.getWidth * scale).toInt,
        math.ceil(image.getHeight * scale).toInt
      )
      val out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      val graphics = out.createGraphics()
      try {
        graphics.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION,
          RenderingHints.VALUE_INTERPOLATION_BILINEAR
        )
        graphics.drawImage(
          image,
          -(scaledWidth - width) / 2,
          -(scaledHeight - height) / 2,
          scaledWidth,
          scaledHeight,
          null
        )
      } finally graphics.dispose()
      out
    }

  /** H.264's constant rate factor of LTX 2.5's image conditions. */
  private val ConditionCrf = 18

  /** `image` through H.264 at CRF 18 (libx264, `veryfast`, 4:2:0) and back, by
    * ffmpeg (diffusers' `apply_image_conditioning_crf`, ltx-pipelines'
    * `preprocess`): LTX 2.5 learned its image conditions with those artifacts,
    * and clean stills tend to freeze the video. Left as it is, with a warning,
    * when ffmpeg fails. Guides are compressed at their own size, as the
    * references do; the init and end images arrive at the video's.
    */
  def compressed(image: BufferedImage): BufferedImage = {
    val folder = Files.createTempDirectory("drift-ltx-condition")
    val (still, video, back, log) = (
      folder.resolve("still.png"),
      folder.resolve("still.mp4"),
      folder.resolve("back.png"),
      folder.resolve("ffmpeg.log")
    )
    def ffmpeg(arguments: String*): Unit = {
      val status = new ProcessBuilder(
        (Seq(
          "ffmpeg",
          "-hide_banner",
          "-loglevel",
          "error",
          "-y"
        ) ++ arguments)*
      ).redirectErrorStream(true)
        .redirectOutput(log.toFile)
        .start()
        .waitFor()
      if (status != 0)
        throw new IllegalStateException(
          s"ffmpeg failed ($status): ${Files.readString(log).trim}"
        )
    }
    try {
      val even = new BufferedImage(
        image.getWidth / 2 * 2,
        image.getHeight / 2 * 2,
        BufferedImage.TYPE_INT_RGB
      )
      val graphics = even.createGraphics()
      try graphics.drawImage(image, 0, 0, null)
      finally graphics.dispose()
      ImageIO.write(even, "png", still.toFile)
      ffmpeg(
        "-i",
        still.toString,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        ConditionCrf.toString,
        "-pix_fmt",
        "yuv420p",
        video.toString
      )
      ffmpeg("-i", video.toString, "-frames:v", "1", back.toString)
      ImageIO.read(back.toFile)
    } catch {
      case error: Exception =>
        println(
          s"[WARN] the image condition goes in without its H.264 compression: ${error.getMessage}"
        )
        image
    } finally Seq(still, video, back, log, folder).foreach(Files.deleteIfExists)
  }

  /** A tensor's raw bytes out of a safetensors file, whatever its type (the
    * text encoder's `tokenizer_json`, U8, which the loaders skip).
    */
  def rawTensor(path: Path, name: String): String = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      val length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
      channel.read(length, 0)
      length.flip()
      val headerLength = length.getLong.toInt
      val header = ByteBuffer.allocate(headerLength)
      channel.read(header, 8)
      val entry =
        ujson.read(new String(header.array(), StandardCharsets.UTF_8))(name)
      val Seq(from, to) = entry("data_offsets").arr.map(_.num.toLong).toSeq
      val data = ByteBuffer.allocate((to - from).toInt)
      channel.read(data, 8L + headerLength + from)
      new String(data.array(), StandardCharsets.UTF_8)
    } finally channel.close()
  }
}
