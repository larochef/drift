package drift.runner.diffusion

import drift.runner.diffusion.Pipeline.Family
import drift.runner.models.*

import java.awt.image.BufferedImage
import java.nio.file.Path

/** One video request, in sd-server's terms (`vid_gen`): `frames` at the model's
  * own rate, `cfgScale` 1 runs the prompt alone (above, each step also runs
  * `negativePrompt`), `shift` the video schedule's flow shift.
  */
final case class VideoRequest(
    prompt: String,
    negativePrompt: String,
    width: Int,
    height: Int,
    frames: Int,
    steps: Int,
    cfgScale: Float,
    seed: Long,
    shift: Double,
    /** The high-noise expert's steps (after which the low-noise one takes
      * `steps` more), when the model has two and the request says.
      */
    highNoiseSteps: Option[Int] = None,
    /** The high-noise expert's CFG scale, else `cfgScale`. */
    highNoiseCfgScale: Option[Float] = None,
    /** Without `highNoiseSteps`: the σ below which the low-noise expert takes
      * over.
      */
    moeBoundary: Float = 0.875f,
    /** The first frame, already the video's size (I2V). */
    initImage: Option[BufferedImage] = None,
    /** The last frame, already the video's size. */
    endImage: Option[BufferedImage] = None,
    /** The LoRAs, each with its multiplier and, for two-expert models, whether
      * it goes on the high-noise expert.
      */
    loras: Seq[VideoLora] = Nil,
    /** Reference media, in the order the model reads them (MiniMax H3's
      * ref2va).
      */
    references: Seq[Media] = Nil,
    /** Media held at given frames (MiniMax H3's guides). */
    guides: Seq[VideoGuide] = Nil,
    /** A control video through the loaded ControlNet. */
    control: Option[VideoControl] = None,
    /** The frame rate, for the models that read it (LTX); the others run at
      * their own.
      */
    fps: Option[Int] = None,
    /** The audio's CFG scale, for the models with a soundtrack (LTX), else
      * `cfgScale`.
      */
    audioCfgScale: Option[Float] = None,
    /** The modality guidance's scales (LTX): above 1, each step also runs the
      * streams isolated from each other and pushes the video (`modalityScale`)
      * and the audio (`audioModalityScale`) away from that pass.
      */
    modalityScale: Float = 1f,
    audioModalityScale: Float = 1f,
    /** Euler ancestral (fresh noise at every step, ComfyUI's flow variant)
      * rather than Euler, for the models that take it (LTX).
      */
    ancestral: Boolean = false
)

/** A LoRA of a video request: `highNoise` puts it on the high-noise expert of a
  * two-expert model (Wan 2.2 A14B), else it goes on the low-noise one.
  */
final case class VideoLora(path: Path, multiplier: Float, highNoise: Boolean)

/** A decoded media input of a video request, at its own size and rates. */
enum Media {
  case Still(image: BufferedImage)
  case Clip(
      frames: Seq[BufferedImage],
      fps: Double,
      soundtrack: Option[Soundtrack]
  )
  case Sound(soundtrack: Soundtrack)
}

/** `media` held at `frameIndex` of the video; a negative index counts from the
  * end (−1 the last frame).
  */
final case class VideoGuide(media: Media, frameIndex: Int)

/** A control video (`frames` at `fps`) applied at `strength` over the steps
  * from fraction `start` to `end`; with a `mask` (white regenerates, one image
  * or one per frame), the rest comes from `source`.
  */
final case class VideoControl(
    frames: Seq[BufferedImage],
    fps: Double,
    strength: Float,
    start: Float,
    end: Float,
    mask: Option[Seq[BufferedImage]],
    source: Option[Seq[BufferedImage]]
)

/** A soundtrack: `samples` interleaved over `channels`, in [−1, 1]. */
final case class Soundtrack(samples: Array[Float], channels: Int, rate: Int)

object Soundtrack {

  /** The loudest a sample may be: a little under full scale, the room the
    * webm's Vorbis encoding overshoots by.
    */
  private val Ceiling = 0.97f

  /** A decoder's raw `samples` as a soundtrack: when their peak passes the
    * ceiling, every sample scaled down by the same factor. The references clamp
    * each sample to [−1, 1] instead, which flattens a hot track's peaks into
    * audible saturation (LTX 2.5 on long captions: hundreds of samples at the
    * clamp).
    */
  def fitted(samples: Array[Float], channels: Int, rate: Int): Soundtrack = {
    val peak = samples.foldLeft(0f)((m, s) => math.max(m, math.abs(s)))
    val factor = if (peak > Ceiling) Ceiling / peak else 1f
    Soundtrack(
      if (factor == 1f) samples else samples.map(_ * factor),
      channels,
      rate
    )
  }
}

/** A generated video: its frames at `fps`, and a soundtrack when the model
  * makes one and its audio decoder is loaded.
  */
final case class Video(
    frames: Seq[BufferedImage],
    fps: Int,
    soundtrack: Option[Soundtrack]
)

/** A video's estimate while it is sampled (`specs/54-step-preview.md`):
  * `frames` small pictures, a latent frame each, spread evenly over the clip's
  * `seconds`; `frame(i)` draws one, only when called.
  */
final case class PreviewClip(
    frames: Int,
    seconds: Double,
    frame: Int => BufferedImage
)

/** A video model family end to end, from a prompt to frames. */
trait VideoPipeline extends Pipeline {

  def mode: String = "vid_gen"

  /** sd-cpp's 20. */
  def defaultSteps: Int = 20

  /** The frame rate the model generates at, whatever a request asks. */
  def fps: Int

  /** The canvas' size multiple. */
  def sizeMultiple: Int

  /** Whether requests may carry an init image (the first frame). */
  def takesInitImage: Boolean = false

  /** Whether requests may carry an end image (the last frame). */
  def takesEndImage: Boolean = false

  /** Whether requests may carry LoRAs. */
  def takesLoras: Boolean = false

  /** Whether requests may carry reference media. */
  def takesReferences: Boolean = false

  /** Whether requests may carry guides. */
  def takesGuides: Boolean = false

  /** Whether requests may carry a control video (a ControlNet is loaded). */
  def takesControl: Boolean = false

  /** Whether a high-noise expert is loaded beside the model. */
  def hasHighNoiseExpert: Boolean = false

  /** Whether the model denoises a soundtrack beside the frames. */
  def makesSoundtrack: Boolean = false

  /** Whether the soundtrack is decoded (its audio VAE loaded): the videos have
    * sound.
    */
  def decodesSoundtrack: Boolean = false

  /** An init or end image put on the `width` × `height` canvas: stretched. A
    * model may fit the `follower` (the end image of a request with both)
    * otherwise.
    */
  def onCanvas(
      image: BufferedImage,
      width: Int,
      height: Int,
      follower: Boolean
  ): BufferedImage = Images.resized(image, width, height)

  /** The frames a request of `frames` gets: the next count the model takes. */
  def alignedFrames(frames: Int): Int

  /** The stages generating `request` goes through, in order, as `stage`
    * announces them: drift's bar over the whole generation.
    */
  def stages(request: VideoRequest): Seq[String]

  /** Announces a stage of `stages` as it starts: the server's to set around
    * `generate`.
    */
  var stage: String => Unit = _ => ()

  /** Told the fraction done of the step under way, by a pipeline whose steps
    * can last minutes: the server's to set around `generate`.
    */
  var withinStep: Double => Unit = _ => ()

  /** Handed each step's estimate of the finished video, before the step is
    * counted (`specs/54-step-preview.md`): `preview(pass, step, steps, clip)`,
    * `pass` the sampling pass from 1 (a two-expert model's second expert is 2),
    * `clip` read and drawn only if called, there and then. The server's to set
    * around `generate`; a family that cannot draw one never calls it.
    */
  var preview: (Int, Int, Int, () => PreviewClip) => Unit = (_, _, _, _) => ()

  /** The video of `request`; `progress(0, steps)` before the first step (of
    * each pass), `progress(step, steps)` after each.
    */
  def generate(request: VideoRequest, progress: (Int, Int) => Unit): Video
}

object VideoPipeline {

  /** Whether `stage`, one of a pipeline's `stages`, is a sampling pass: the
    * stages a preview is drawn through.
    */
  def samples(stage: String): Boolean = stage.startsWith("sampling")

  /** The video families, in the order they are tried: MiniMax H3, which takes a
    * video VAE, the Qwen3-VL text encoder (`--llm`) and Qwen3's
    * `tokenizer.json`; Wan, which takes the Wan 2.1 VAE, UMT5 (`--t5xxl`) and
    * its `tokenizer.json`, and a high-noise expert for Wan 2.2 A14B; LTX 2.5,
    * which takes its conv VAE and its Gemma 4 text encoder (`--llm`). MiniMax
    * H3 and LTX 2.5 decode their soundtrack with an audio VAE (LTX's with its
    * vocoder), else their videos are silent.
    */
  val Families: Seq[Family] = Seq(
    Family(
      "MiniMax H3",
      Seq("minimax-h3"),
      MiniMaxH3Config.holds,
      (ops, files) =>
        new MiniMaxH3Pipeline(
          ops,
          files.diffusionModel,
          files.needed(files.vae, "--vae", "MiniMax H3's video VAE"),
          files.needed(
            files.textEncoder,
            "--llm",
            "MiniMax H3's Qwen3-VL text encoder"
          ),
          files
            .needed(files.tokenizer, "--tokenizer", "Qwen3's tokenizer.json"),
          files.audioVae,
          files.controlNet,
          files.textEncoderVision
        ),
      takesControlNet = true
    ),
    Family(
      "Wan",
      Seq("wan-2.2-14b"),
      WanConfig.holds,
      (ops, files) =>
        new WanPipeline(
          ops,
          files.diffusionModel,
          files.highNoiseModel,
          files.needed(files.vae, "--vae", "the Wan 2.1 VAE"),
          files.needed(files.t5, "--t5xxl", "UMT5-XXL"),
          files.needed(files.tokenizer, "--tokenizer", "UMT5's tokenizer.json")
        )
    ),
    Family(
      "LTX 2.5",
      Seq("ltx-2.5"),
      Ltx2Config.holds,
      (ops, files) =>
        new LtxPipeline(
          ops,
          files.diffusionModel,
          files.needed(files.vae, "--vae", "LTX 2.5's convolutional video VAE"),
          files.needed(
            files.textEncoder,
            "--llm",
            "LTX 2.5's Gemma 4 text encoder (with its projections)"
          ),
          files.fps.getOrElse(24),
          files.audioVae
        )
    )
  )
}
