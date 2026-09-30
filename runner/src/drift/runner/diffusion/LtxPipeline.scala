package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Path, StandardOpenOption}
import java.util.SplittableRandom
import scala.collection.mutable

/** LTX 2.5 text to video (diffusers' `LTX25AutoBlocks` `text2video`, with the
  * convolutional VAE): the prompt (Gemma's tokenizer, `<bos>` first, read out
  * of the text encoder's own file) through Gemma 4 to every hidden state, the
  * text features and the transformer file's connectors (1024 rows, the
  * registers after the prompt), noise for the video (128 latent channels per 32
  * × 32 pixels × 8 frames) and the audio (128 per latent, 25 a second),
  * denoised together by Euler steps on the distilled checkpoint's σ list (8
  * steps) or `linspace(1, 1/N, N)` through the resolution's shift; the video
  * latents through the conv VAE. The audio is denoised, not decoded (a silent
  * video). The frame rate is a model input (the RoPE's time); frames round down
  * to `8n + 1`, sides up to multiples of 32.
  */
final class LtxPipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    defaultFps: Int
) extends VideoPipeline {

  def family: String = "LTX 2.5"
  def fps: Int = defaultFps
  def sizeMultiple: Int = 32

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
  private val (connectorSource, videoConnector, audioConnector) =
    Ltx2.connectors(ops, diffusionModel)
  private val decoder = LtxVideoVae.open(ops, vae)
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
    val layout = Ltx2Layout(
      frames = (frames - 1) / 8 + 1,
      height = height / 32,
      width = width / 32,
      audioFrames = math.round(frames.toDouble / rate * 25).toInt,
      fps = rate
    )
    val c = transformer.config
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val (videoText, audioText) = text(request.prompt)
      held ++= Seq(videoText, audioText)
      val guided = request.cfgScale != 1f
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
          audioVelocity
        )
        for {
          (v, a) <- negative
          (ov, oa) <- others
        } {
          transformer.velocity(layout, video, audio, v, a, timestep, ov, oa)
          // v = uncond + scale × (cond − uncond)
          Seq(ov -> videoVelocity, oa -> audioVelocity).foreach {
            (other, velocity) =>
              ops.scale(other, 1 - request.cfgScale, other)
              ops.scale(velocity, request.cfgScale, velocity)
              ops.add(velocity, other, velocity)
          }
        }
        ops.scale(videoVelocity, schedule(i + 1) - schedule(i), videoVelocity)
        ops.add(video, videoVelocity, video)
        ops.scale(audioVelocity, schedule(i + 1) - schedule(i), audioVelocity)
        ops.add(audio, audioVelocity, audio)
        progress(i + 1, steps)
      }
      val perFrame = layout.height * layout.width
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
      Video(images.toSeq, rate, None)
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    decoder.close()
    audioConnector.release()
    videoConnector.release()
    connectorSource.close()
    transformer.close()
    features.release()
    gemma.close()
  }
}

object LtxPipeline {

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
