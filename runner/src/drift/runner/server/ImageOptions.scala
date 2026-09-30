package drift.runner.server

import java.nio.file.{Path, Paths}

/** `sd-server`'s flags as drift passes them for an image configuration
  * (`specs/42`, step 12): the checkpoints, the defaults a request starts from
  * (size, steps, CFG scale, distilled guidance, flow shift, seed, prompts), and
  * where to listen; a video's frames and its audio VAE. A whole model in one
  * file (HiDream O1) comes as `--model`, with no VAE or text encoder of its
  * own. Flags that change how sd-cpp runs but not the image are accepted with a
  * note; flags asking for what the runner cannot do yet are refused by name,
  * and so is any unknown flag.
  */
final case class ImageOptions(
    diffusionModel: Path,
    vae: Option[Path],
    llm: Option[Path],
    host: String,
    port: Int,
    width: Int,
    height: Int,
    /** `--steps`, when given; else the family's default (`ImageServer`). */
    steps: Option[Int],
    cfgScale: Double,
    /** The distilled guidance scale (`--guidance`), when given: read by the
      * models that embed one (FLUX.2 [dev]), noted and ignored by the others.
      */
    guidance: Option[Double],
    flowShift: Double,
    seed: Long,
    prompt: String,
    negativePrompt: String,
    /** Where requests' LoRA paths are resolved (`--lora-model-dir`). */
    loraDirectory: Option[Path],
    /** A `tokenizer.json` for text encoders whose file has none (PiD's Gemma
      * 2).
      */
    tokenizer: Option[Path],
    /** The text encoder's vision tower (`--llm_vision`, an mmproj), for editing
      * with reference images.
      */
    llmVision: Option[Path],
    /** The audio VAE of a model that makes a soundtrack (`--audio-vae`). */
    audioVae: Option[Path],
    /** A video's frames (`--video-frames`). */
    videoFrames: Int,
    /** Wan 2.2 A14B's high-noise expert (`--high-noise-diffusion-model`). */
    highNoiseModel: Option[Path],
    /** A T5 text encoder (`--t5xxl`): Wan's UMT5. */
    t5xxl: Option[Path],
    /** The high-noise expert's steps and CFG scale, when given. */
    highNoiseSteps: Option[Int],
    highNoiseCfgScale: Option[Double],
    /** The σ below which the low-noise expert takes over (sd-cpp's 0.875). */
    moeBoundary: Double,
    /** The frame rate (`--fps`), which LTX reads; MiniMax H3 and Wan run at
      * their own.
      */
    fps: Option[Int],
    notes: Seq[String]
)

object ImageOptions {

  /** sd-server's distilled guidance when neither the flags nor the request give
    * one.
    */
  val DefaultGuidance = 3.5

  /** Accepted and ignored: they change how sd-cpp runs, not the image. The
    * value is how many values the flag takes.
    */
  private val Harmless: Map[String, Int] = Map(
    "--diffusion-fa" -> 0,
    "--mmap" -> 0,
    "--offload-to-cpu" -> 0,
    "--vae-conv-direct" -> 0,
    "--diffusion-conv-direct" -> 0,
    "--cache-mode" -> 1, // spectrum and the like: the runner computes every step
    "--hires-upscalers-dir" -> 1,
    "--sampling-method" -> 1, // Euler: flow-matching models' only sampler here
    "--scheduler" -> 1,
    "-t" -> 1,
    "--threads" -> 1,
    "--rng" -> 1,
    "--sampler-rng" -> 1,
    "--vae-format" -> 1, // the runner reads the VAE's layout from its weights
    "--high-noise-sampling-method" -> 1, // Euler, as for the low-noise steps
    "--high-noise-scheduler" -> 1,
    "-v" -> 0,
    "--verbose" -> 0,
    "--color" -> 0
  )

  /** Refused until the runner does what they ask. */
  private val NotYet: Map[String, String] = Map(
    "--vae-tiling" -> "tiled VAE decoding comes later",
    "--control-net" -> "ControlNets come later",
    "--taesd" -> "TAESD previews come later"
  )

  def parse(arguments: Seq[String]): Either[String, ImageOptions] = {
    val values = scala.collection.mutable.Map.empty[String, String]
    val notes = Seq.newBuilder[String]
    var rest = arguments.toList
    var problem = Option.empty[String]
    while (rest.nonEmpty && problem.isEmpty) {
      val flag = rest.head
      rest = rest.tail
      def value(): Option[String] = rest match {
        case v :: tail =>
          rest = tail
          Some(v)
        case Nil =>
          problem = Some(s"$flag needs a value")
          None
      }
      flag match {
        case "--diffusion-model" | "--model" | "-m" | "--vae" | "--llm" |
            "--listen-ip" | "--listen-port" | "-W" | "--width" | "-H" |
            "--height" | "--steps" | "--cfg-scale" | "--guidance" |
            "--flow-shift" | "-s" | "--seed" | "-p" | "--prompt" | "-n" |
            "--negative-prompt" | "--lora-model-dir" | "--tokenizer" |
            "--llm_vision" | "--audio-vae" | "--video-frames" |
            "--high-noise-diffusion-model" | "--t5xxl" | "--high-noise-steps" |
            "--high-noise-cfg-scale" | "--moe-boundary" | "--fps" =>
          value().foreach(v => values(canonical(flag)) = v)
        case other if NotYet.contains(other) =>
          problem = Some(s"$other is not supported yet: ${NotYet(other)}")
        case other if Harmless.contains(other) =>
          val taken = (0 until Harmless(other)).flatMap(_ => value())
          notes += s"$other ${taken.mkString(" ")} accepted: it changes how sd-cpp runs, not the image"
            .replace("  ", " ")
        case other => problem = Some(s"unknown flag $other")
      }
    }
    def path(flag: String) =
      values
        .get(flag)
        .map(Paths.get(_))
        .toRight(s"no ${flag.stripPrefix("--")}: pass $flag <file>")
    problem.map(Left(_)).getOrElse {
      for {
        diffusion <- path("--diffusion-model").left.flatMap(_ =>
          path("--model").left.map(_ =>
            "no diffusion model: pass --diffusion-model <file> (or --model <file>)"
          )
        )
      } yield ImageOptions(
        diffusion,
        values.get("--vae").map(Paths.get(_)),
        values.get("--llm").map(Paths.get(_)),
        values.getOrElse("--listen-ip", "127.0.0.1"),
        values.get("--listen-port").fold(1234)(_.toInt),
        values.get("-W").fold(1024)(_.toInt),
        values.get("-H").fold(1024)(_.toInt),
        values.get("--steps").map(_.toInt),
        values.get("--cfg-scale").fold(1.0)(_.toDouble),
        values.get("--guidance").map(_.toDouble),
        values.get("--flow-shift").fold(1.15)(_.toDouble),
        values.get("-s").fold(-1L)(_.toLong),
        values.getOrElse("-p", ""),
        values.getOrElse("-n", ""),
        values.get("--lora-model-dir").map(Paths.get(_)),
        values.get("--tokenizer").map(Paths.get(_)),
        values.get("--llm_vision").map(Paths.get(_)),
        values.get("--audio-vae").map(Paths.get(_)),
        values.get("--video-frames").fold(33)(_.toInt),
        values.get("--high-noise-diffusion-model").map(Paths.get(_)),
        values.get("--t5xxl").map(Paths.get(_)),
        values.get("--high-noise-steps").map(_.toInt).filter(_ >= 0),
        values.get("--high-noise-cfg-scale").map(_.toDouble),
        values.get("--moe-boundary").fold(0.875)(_.toDouble),
        values.get("--fps").map(_.toInt),
        notes.result()
      )
    }
  }

  private def canonical(flag: String): String = flag match {
    case "--width"           => "-W"
    case "--height"          => "-H"
    case "--seed"            => "-s"
    case "--prompt"          => "-p"
    case "--negative-prompt" => "-n"
    case "-m"                => "--model"
    case other               => other
  }
}
