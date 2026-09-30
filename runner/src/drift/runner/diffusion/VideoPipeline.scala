package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops

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
    /** The frame rate, for the models that read it (LTX); the others run at
      * their own.
      */
    fps: Option[Int] = None
)

/** A soundtrack: `samples` interleaved over `channels`, in [−1, 1]. */
final case class Soundtrack(samples: Array[Float], channels: Int, rate: Int)

/** A generated video: its frames at `fps`, and a soundtrack when the model
  * makes one and its audio decoder is loaded.
  */
final case class Video(
    frames: Seq[BufferedImage],
    fps: Int,
    soundtrack: Option[Soundtrack]
)

/** A video model family end to end, from a prompt to frames. */
trait VideoPipeline extends AutoCloseable {

  /** The family's name, as messages give it. */
  def family: String

  /** The frame rate the model generates at, whatever a request asks. */
  def fps: Int

  /** The canvas' size multiple. */
  def sizeMultiple: Int

  /** Whether requests may carry an init image (the first frame). */
  def takesInitImage: Boolean = false

  /** The frames a request of `frames` gets: the next count the model takes. */
  def alignedFrames(frames: Int): Int

  /** The video of `request`; `progress(step, steps)` after each step. */
  def generate(request: VideoRequest, progress: (Int, Int) => Unit): Video
}

object VideoPipeline {

  private enum Family { case MiniMaxH3, Wan, Ltx }

  private def family(ops: Ops, diffusionModel: Path): Option[Family] = {
    val source = WeightSource.open(ops, diffusionModel)
    try
      if (MiniMaxH3Config.holds(source)) Some(Family.MiniMaxH3)
      else if (WanConfig.holds(source)) Some(Family.Wan)
      else if (Ltx2Config.holds(source)) Some(Family.Ltx)
      else None
    finally source.close()
  }

  /** Whether `diffusionModel` is a video model this runner draws. */
  def holds(ops: Ops, diffusionModel: Path): Boolean =
    family(ops, diffusionModel).isDefined

  /** The pipeline of `diffusionModel`'s family: MiniMax H3, which takes a video
    * VAE, the Qwen3-VL text encoder (`--llm`) and Qwen3's `tokenizer.json`;
    * Wan, which takes the Wan 2.1 VAE, UMT5 (`--t5xxl`) and its
    * `tokenizer.json`, and a high-noise expert for Wan 2.2 A14B; LTX 2.5, which
    * takes its conv VAE and its Gemma 4 text encoder (`--llm`).
    */
  def open(
      ops: Ops,
      diffusionModel: Path,
      highNoiseModel: Option[Path],
      vaeFile: Option[Path],
      textEncoderFile: Option[Path],
      t5File: Option[Path],
      tokenizer: Option[Path],
      fps: Option[Int]
  ): VideoPipeline = {
    def needed(file: Option[Path], flag: String, what: String) =
      file.getOrElse(
        throw new IllegalArgumentException(
          s"${diffusionModel.getFileName} needs $what: pass $flag <file>"
        )
      )
    family(ops, diffusionModel) match {
      case Some(Family.MiniMaxH3) =>
        new MiniMaxH3Pipeline(
          ops,
          diffusionModel,
          needed(vaeFile, "--vae", "MiniMax H3's video VAE"),
          needed(
            textEncoderFile,
            "--llm",
            "MiniMax H3's Qwen3-VL text encoder"
          ),
          needed(tokenizer, "--tokenizer", "Qwen3's tokenizer.json")
        )
      case Some(Family.Wan) =>
        new WanPipeline(
          ops,
          diffusionModel,
          highNoiseModel,
          needed(vaeFile, "--vae", "the Wan 2.1 VAE"),
          needed(t5File, "--t5xxl", "UMT5-XXL"),
          needed(tokenizer, "--tokenizer", "UMT5's tokenizer.json")
        )
      case Some(Family.Ltx) =>
        new LtxPipeline(
          ops,
          diffusionModel,
          needed(vaeFile, "--vae", "LTX 2.5's convolutional video VAE"),
          needed(
            textEncoderFile,
            "--llm",
            "LTX 2.5's Gemma 4 text encoder (with its projections)"
          ),
          fps.getOrElse(24)
        )
      case None =>
        throw new IllegalArgumentException(
          s"${diffusionModel.getFileName} is no video model the runner draws"
        )
    }
  }
}
