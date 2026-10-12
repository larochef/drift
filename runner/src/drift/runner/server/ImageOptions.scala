package drift.runner.server

import drift.protocol.ImageServerFlags

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
    /** `--steps`, when given; else the family's default (`LaunchDefaults`). */
    steps: Option[Int],
    cfgScale: Double,
    /** The distilled guidance scale (`--guidance`), when given: read by the
      * models that embed one (FLUX.2 [dev]), noted and ignored by the others.
      */
    guidance: Option[Double],
    /** `--flow-shift`, when given; else the family's own (`LaunchDefaults`). */
    flowShift: Option[Double],
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
    /** A video model's ControlNet (`--control-net`): MiniMax H3's Fun
      * ControlNet union.
      */
    controlNet: Option[Path],
    /** LLaDA-Image's QueryFormer, text projection and SigVQ encoder in one file
      * (`--embeddings-connectors`, sd-cpp's flag).
      */
    connectors: Option[Path],
    /** LLaDA-Image's small models each in its own file, by name
      * (`--llada-queryformer`, `--llada-text-projection`, `--llada-sigvq`).
      */
    lladaParts: Map[String, Path],
    /** Noise levels in place of the model's schedule (`--sigmas`, comma
      * separated), for the models that take them; none is the schedule.
      */
    sigmas: Seq[Double],
    /** The sampler of a request that names none (`--sampling-method`), where
      * the model has it among its own; else the model's.
      */
    sampler: Option[String],
    notes: Seq[String]
)

object ImageOptions {

  /** sd-server's distilled guidance when neither the flags nor the request give
    * one.
    */
  val DefaultGuidance = 3.5

  def parse(arguments: Seq[String]): Either[String, ImageOptions] =
    ImageServerFlags
      .read(arguments)
      .flatMap(read => from(read.values, read.notes))

  private def from(
      values: Map[String, String],
      notes: Seq[String]
  ): Either[String, ImageOptions] = {
    def path(flag: String) =
      values
        .get(flag)
        .map(Paths.get(_))
        .toRight(s"no ${flag.stripPrefix("--")}: pass $flag <file>")
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
      values.get("--flow-shift").map(_.toDouble),
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
      values.get("--control-net").map(Paths.get(_)),
      values.get("--embeddings-connectors").map(Paths.get(_)),
      Seq("queryformer", "text_projection", "sigvq")
        .flatMap(name =>
          values
            .get(s"--llada-${name.replace('_', '-')}")
            .map(file => name -> Paths.get(file))
        )
        .toMap,
      values
        .get("--sigmas")
        .toSeq
        .flatMap(_.split(',').map(_.trim).filter(_.nonEmpty))
        .map(_.toDouble),
      values.get("--sampling-method"),
      notes
    )
  }
}
