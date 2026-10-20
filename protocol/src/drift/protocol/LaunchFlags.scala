package drift.protocol

/** The flags one of the runner's servers takes, and how its command line is
  * read with them: a flag it reads takes one value, a flag that changes how the
  * upstream program runs but not what it makes is accepted with a note, a flag
  * asking for what the runner cannot do yet is refused with why, and any other
  * is refused by name — dropping one would change what was asked.
  */
trait LaunchFlags {

  /** Read: each takes one value. */
  def Valued: Set[String]

  /** Accepted and ignored, each with how many values it takes. */
  def Harmless: Map[String, Int]

  /** Refused until the runner does what they ask, with why. */
  def NotYet: Map[String, String]

  /** Other spellings of a read flag, each with the one its value is kept by. */
  def Aliases: Map[String, String]

  /** Why a harmless flag is one, as its note says. */
  def harmlessBecause: String

  /** Whether a launch may carry the flag. */
  def accepts(flag: String): Boolean =
    Valued(flag) || Harmless.contains(flag)

  /** A command line read: the value of each flag given, by the spelling
    * `Aliases` keeps it by, and a note per harmless flag — or the first thing
    * wrong with it.
    */
  def read(arguments: Seq[String]): Either[String, LaunchFlags.Read] = {
    val values = Map.newBuilder[String, String]
    val notes = Seq.newBuilder[String]
    var rest = arguments.toList
    var problem = Option.empty[String]
    while (rest.nonEmpty && problem.isEmpty) {
      val flag = rest.head
      rest = rest.tail
      def value(): Option[String] = rest match {
        case first :: tail =>
          rest = tail
          Some(first)
        case Nil =>
          problem = Some(s"$flag needs a value")
          None
      }
      if (Valued(flag))
        value().foreach(said => values += Aliases.getOrElse(flag, flag) -> said)
      else if (NotYet.contains(flag))
        problem = Some(s"$flag is not supported yet: ${NotYet(flag)}")
      else if (Harmless.contains(flag)) {
        val taken = (0 until Harmless(flag)).flatMap(_ => value())
        notes += s"$flag ${taken.mkString(" ")} accepted: $harmlessBecause"
          .replace("  ", " ")
      } else problem = Some(s"unknown flag $flag")
    }
    problem.toLeft(LaunchFlags.Read(values.result(), notes.result()))
  }
}

object LaunchFlags {
  final case class Read(values: Map[String, String], notes: Seq[String])
}

/** The flags drift's runner takes when it is started as an image, video or
  * upscale server, by their sd-server names: one list for the runner, which
  * refuses any other by name, and for the backend, whose launches are checked
  * against it.
  */
object ImageServerFlags extends LaunchFlags {

  val Valued: Set[String] = Set(
    "--diffusion-model",
    "--model",
    "-m",
    "--vae",
    "--llm",
    "--listen-ip",
    "--listen-port",
    "-W",
    "--width",
    "-H",
    "--height",
    "--steps",
    "--cfg-scale",
    "--guidance",
    "--flow-shift",
    "-s",
    "--seed",
    "-p",
    "--prompt",
    "-n",
    "--negative-prompt",
    "--lora-model-dir",
    "--tokenizer",
    "--llm_vision",
    "--audio-vae",
    "--video-frames",
    "--high-noise-diffusion-model",
    "--t5xxl",
    "--high-noise-steps",
    "--high-noise-cfg-scale",
    "--moe-boundary",
    "--fps",
    "--control-net",
    "--sigmas",
    "--sampling-method",
    "--embeddings-connectors",
    "--llada-queryformer",
    "--llada-text-projection",
    "--llada-sigvq"
  )

  val Harmless: Map[String, Int] = Map(
    "--diffusion-fa" -> 0,
    "--mmap" -> 0,
    "--offload-to-cpu" -> 0,
    "--vae-conv-direct" -> 0,
    "--diffusion-conv-direct" -> 0,
    "--cache-mode" -> 1, // spectrum and the like: the runner computes every step
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

  val NotYet: Map[String, String] = Map(
    "--vae-tiling" -> "tiled VAE decoding comes later",
    "--taesd" -> "TAESD previews come later"
  )

  val Aliases: Map[String, String] = Map(
    "--width" -> "-W",
    "--height" -> "-H",
    "--seed" -> "-s",
    "--prompt" -> "-p",
    "--negative-prompt" -> "-n",
    "-m" -> "--model"
  )

  val harmlessBecause = "it changes how sd-cpp runs, not the image"
}

/** The flags drift's runner takes when it is started as a chat server, by their
  * llama-server names. Flags that change speed but not the result (GPU layers,
  * flash attention) are accepted; so are the drafting kinds the runner does not
  * do, which would give the same output.
  */
object ChatServerFlags extends LaunchFlags {

  val Valued: Set[String] = Set(
    "-m",
    "--model",
    "--spec-draft-model",
    "-md",
    "--model-draft",
    "--mmproj",
    "-mm",
    "-c",
    "--ctx-size",
    "--host",
    "--port",
    "--spec-type",
    "--spec-draft-n-max"
  )

  val Harmless: Map[String, Int] = Map(
    "-ngl" -> 1,
    "--n-gpu-layers" -> 1,
    "--gpu-layers" -> 1,
    "-fa" -> 1,
    "--flash-attn" -> 1,
    "--jinja" -> 0,
    "--no-mmap" -> 0,
    "--mmap" -> 0,
    "-t" -> 1,
    "--threads" -> 1,
    "-np" -> 1,
    "--parallel" -> 1,
    "--metrics" -> 0,
    "--no-webui" -> 0,
    "--lora-init-without-apply" -> 0,
    "--no-mmproj-offload" -> 0,
    "--mmproj-offload" -> 0,
    "--spec-draft-ngl" -> 1,
    "-ngld" -> 1,
    "--gpu-layers-draft" -> 1,
    "--n-gpu-layers-draft" -> 1,
    "--spec-draft-device" -> 1,
    "-devd" -> 1,
    "--device-draft" -> 1,
    // the runner drafts --spec-draft-n-max tokens every step
    "--spec-draft-n-min" -> 1,
    "--spec-draft-p-min" -> 1,
    "--draft-p-min" -> 1
  )

  val NotYet: Map[String, String] = Map(
    "--lora" -> "LoRAs on chat models come later",
    "--chat-template" -> "the runner uses the model's own template",
    "--chat-template-file" -> "the runner uses the model's own template"
  )

  val Aliases: Map[String, String] = Map(
    "--model" -> "-m",
    "-md" -> "--spec-draft-model",
    "--model-draft" -> "--spec-draft-model",
    "-mm" -> "--mmproj",
    "--ctx-size" -> "-c"
  )

  val harmlessBecause =
    "it changes how llama.cpp runs, not what it answers"
}

/** The model kinds drift's runner runs, as drift's architectures name theirs
  * (`modelKind`): what the runner answers `--model-kinds` with, and what the
  * reference data is checked against.
  */
object RunnerModelKinds {

  /** The image server's: the diffusion families it draws, the video ones, and
    * the upscaler.
    */
  val Images: Seq[String] = Seq(
    "krea2",
    "flux3-klein",
    "flux2-klein",
    "flux2-dev",
    "qwen-image-2.1",
    "pid-flux2",
    "pid-flux1",
    "pid-qwen-image",
    "hidream-o1",
    "mage-flow",
    "nucleus-image",
    "llada-image",
    "grn",
    "minimax-h3",
    "wan-2.2-14b",
    "ltx-2.5",
    "seedvr2"
  )

  /** The chat server's. */
  val Chat: Seq[String] = Seq(
    "qwen3",
    "qwen3vl",
    "qwen35",
    "qwen35moe",
    "qwen4exp"
  )
}
