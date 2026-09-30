package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** MiniMax H3 text to video (diffusers' `t2va` modular pipeline): the prompt,
  * tokenized bare, through Qwen3-VL-32B's language model to the residual stream
  * after its 50th layer, projected and refined once by the transformer; noise
  * for the video (2 × 2 patches of 24 latent channels) and the audio (40
  * latents a second on two stereo channels) packed after the text and denoised
  * together, one forward per step (the model is guidance-distilled), each
  * modality down its own schedule (`linspace(1, 0)` through a flow shift of 12
  * for the video, 3 for the audio); the video latents decoded by the ViT VAE in
  * its released tiling; the audio latents, one stereo channel after the other,
  * each decoded by the audio VAE (BigVGAN) into 32 kHz, when it is given (else
  * the video is silent). A request's frames round up to the `17n + 5` the VAE
  * decodes, its sides up to multiples of 32; the model runs at 24 fps.
  */
final class MiniMaxH3Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Path,
    audioVae: Option[Path]
) extends VideoPipeline {

  def family: String = "MiniMax H3"
  def fps: Int = 24
  def sizeMultiple: Int = 32
  override def makesSoundtrack: Boolean = true
  override def decodesSoundtrack: Boolean = audioDecoder.isDefined

  def alignedFrames(frames: Int): Int = {
    var aligned = math.max(frames, 5)
    while (aligned % 17 != 5) aligned += 1
    aligned
  }

  // the quantized products' activations pass F16's range
  ops.wideProducts = true
  private val tokenizer: Tokenizer = TokenizerJson.load(tokenizerFile)
  private val encoder = Qwen3.open(ops, textEncoder)
  private val transformer = MiniMaxH3.open(ops, diffusionModel)
  private val decoder = MiniMaxH3Vae.open(ops, vae)
  private val audioDecoder = audioVae.map(MiniMaxH3Audio.open(ops, _))
  private val TextLayer = 50
  private val AudioShift = 3.0
  private val AudioLatentsPerSecond = 40

  /** The prompt's text for every step, `[L, hidden]`. */
  private def text(prompt: String): Tensor = {
    val ids = tokenizer.encode(prompt, addSpecial = false)
    require(ids.nonEmpty, "MiniMax H3 needs a prompt")
    val encoded = ops.allocate(
      DType.F32,
      Shape.of(ids.length.toLong, encoder.config.hidden.toLong)
    )
    try {
      encoder.encode(ids, Seq(TextLayer), encoded)
      val out = ops.allocate(
        DType.F32,
        Shape.of(ids.length.toLong, transformer.config.hidden.toLong)
      )
      transformer.encodeText(encoded, out)
      out
    } finally ops.release(encoded)
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
    val layout = MiniMaxH3Layout(
      textTokens = 0,
      frames = (frames - 5) / 17 * 5 + 2,
      latentHeight = height / 16,
      latentWidth = width / 16,
      audioLatents =
        math.round(frames.toDouble / fps * AudioLatentsPerSecond).toInt
    )
    val c = transformer.config
    val guided = request.cfgScale != 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val conditional = keep(text(request.prompt))
      val unconditional =
        Option.when(guided)(keep(text(request.negativePrompt)))
      def laid(prompt: Tensor) =
        layout.copy(textTokens = prompt.shape.dimensions.head.toInt)
      val random = new SplittableRandom(request.seed)
      val video = keep(
        ops.fromFloats(
          Shape.of(layout.videoRows, c.videoWidth),
          Array.fill(layout.videoRows * c.videoWidth)(Images.gaussian(random))
        )
      )
      val audio = keep(
        ops.fromFloats(
          Shape.of(layout.audioRows, c.audioWidth),
          Array.fill(layout.audioRows * c.audioWidth)(Images.gaussian(random))
        )
      )
      val (videoVelocity, audioVelocity) = (
        keep(ops.allocate(DType.F32, video.shape)),
        keep(ops.allocate(DType.F32, audio.shape))
      )
      val others = unconditional.map(_ =>
        (
          keep(ops.allocate(DType.F32, video.shape)),
          keep(ops.allocate(DType.F32, audio.shape))
        )
      )
      val (videoSigmas, audioSigmas) =
        (
          sigmas(request.steps, request.shift),
          sigmas(request.steps, AudioShift)
        )
      (0 until request.steps).foreach { i =>
        val (videoTime, audioTime) = (1 - videoSigmas(i), 1 - audioSigmas(i))
        transformer.velocity(
          laid(conditional),
          conditional,
          video,
          audio,
          videoTime,
          audioTime,
          videoVelocity,
          audioVelocity
        )
        for {
          uncond <- unconditional
          (v, a) <- others
        } {
          transformer.velocity(
            laid(uncond),
            uncond,
            video,
            audio,
            videoTime,
            audioTime,
            v,
            a
          )
          // v = uncond + scale × (cond − uncond)
          Seq(v -> videoVelocity, a -> audioVelocity).foreach {
            (other, velocity) =>
              ops.scale(other, 1 - request.cfgScale, other)
              ops.scale(velocity, request.cfgScale, velocity)
              ops.add(velocity, other, velocity)
          }
        }
        // the velocity points towards the data: x += (σ − σ') v
        ops.scale(
          videoVelocity,
          videoSigmas(i) - videoSigmas(i + 1),
          videoVelocity
        )
        ops.add(video, videoVelocity, video)
        ops.scale(
          audioVelocity,
          audioSigmas(i) - audioSigmas(i + 1),
          audioVelocity
        )
        ops.add(audio, audioVelocity, audio)
        progress(i + 1, request.steps)
      }
      // rows (t, y, x) of features (c, py, px) → latents [T, h, w, 24]
      val rows = ops.toFloats(video)
      val channels = c.videoWidth / 4
      val (latentHeight, latentWidth) =
        (layout.latentHeight, layout.latentWidth)
      val latents = new Array[Float](rows.length)
      rows.indices.foreach { i =>
        val (row, feature) = (i / c.videoWidth, i % c.videoWidth)
        val (channel, py, px) = (feature / 4, feature / 2 % 2, feature % 2)
        val perFrame = (latentHeight / 2) * (latentWidth / 2)
        val (t, y, x) =
          (
            row / perFrame,
            row % perFrame / (latentWidth / 2),
            row % (latentWidth / 2)
          )
        latents(
          ((t * latentHeight + 2 * y + py) * latentWidth + 2 * x + px) * channels + channel
        ) = rows(i)
      }
      val started = System.nanoTime()
      val images = mutable.ArrayBuffer.empty[BufferedImage]
      decoder.decode(
        latents,
        layout.frames,
        latentHeight,
        latentWidth,
        rgb => images += Images.toImage(rgb.map(_ * 2 - 1), width, height)
      )
      println(
        f"decoded ${images.size} frames in ${(System.nanoTime() - started) / 1e9}%.1f s"
      )
      Video(images.toSeq.take(frames), fps, audioDecoder.map(soundtrack(audio, _)))
    } finally held.foreach(ops.release)
  }

  /** The stereo soundtrack of the audio rows (`[2L, width]`, channel-major):
    * each channel decoded alone, interleaved.
    */
  private def soundtrack(audio: Tensor, audioDecoder: MiniMaxH3Audio) = {
    val started = System.nanoTime()
    val rows = ops.toFloats(audio)
    val channels = MiniMaxH3Layout.AudioChannels
    val perChannel = rows.length / channels
    require(
      audio.shape.dimensions.last == audioDecoder.channels,
      s"audio rows of ${audio.shape.dimensions.last} for a decoder of ${audioDecoder.channels} channels"
    )
    val tracks = (0 until channels).map(c =>
      audioDecoder.decode(rows.slice(c * perChannel, (c + 1) * perChannel))
    )
    val samples = Array.tabulate(tracks.head.length * channels)(i =>
      tracks(i % channels)(i / channels)
    )
    println(
      f"decoded ${tracks.head.length.toDouble / audioDecoder.sampleRate}%.2f s of sound in ${(System.nanoTime() - started) / 1e9}%.1f s"
    )
    Soundtrack.fitted(samples, channels, audioDecoder.sampleRate)
  }

  def close(): Unit = {
    audioDecoder.foreach(_.close())
    decoder.close()
    transformer.close()
    encoder.close()
  }
}
