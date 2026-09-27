package drift.runner.server

import java.nio.file.{Path, Paths}

/** The runner's command line: llama-server's own flags, so that drift launches
  * it exactly as it launches llama-server (`specs/42`, step 7). A flag the
  * runner cannot honour yet is refused by name, never dropped: dropping one
  * would change what drift asked for. Flags that change speed but not the
  * result (GPU layers, flash attention) are accepted. Speculative decoding
  * through the model's MTP layer (`--spec-type draft-mtp`) drafts
  * `--spec-draft-n-max` tokens per step (2 when not given); other drafting
  * kinds are accepted and not done: they would give the same output.
  */
final case class ServerOptions(
    model: Path,
    context: Int,
    host: String,
    port: Int,
    /** The MTP head in a file of its own (`--spec-draft-model`, `-md`,
      * `--model-draft`).
      */
    draftModel: Option[Path],
    /** The vision tower (`--mmproj`): the model then reads images. */
    visionModel: Option[Path],
    /** Tokens drafted per step by the model's MTP layer; 0 for none. */
    drafts: Int,
    notes: Seq[String]
)

object ServerOptions {

  /** Accepted and ignored: they change how llama.cpp runs, not what it says. */
  private val Harmless: Map[String, Int] = Map(
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

  /** Refused until the runner does what they ask. */
  private val NotYet: Map[String, String] = Map(
    "--lora" -> "LoRAs on chat models come later",
    "--chat-template" -> "the runner uses the model's own template",
    "--chat-template-file" -> "the runner uses the model's own template"
  )

  def parse(arguments: Seq[String]): Either[String, ServerOptions] = {
    var model = Option.empty[Path]
    var draftModel = Option.empty[Path]
    var visionModel = Option.empty[Path]
    var context = 4096
    var host = "127.0.0.1"
    var port = 8080
    var multiToken = false
    var draftMaximum = 2
    val notes = Seq.newBuilder[String]
    var rest = arguments.toList
    var problem = Option.empty[String]
    def value(flag: String): Option[String] = rest match {
      case v :: tail =>
        rest = tail
        Some(v)
      case Nil =>
        problem = Some(s"$flag needs a value")
        None
    }
    while (rest.nonEmpty && problem.isEmpty) {
      val flag = rest.head
      rest = rest.tail
      flag match {
        case "-m" | "--model" =>
          value(flag).foreach(v => model = Some(Paths.get(v)))
        case "--spec-draft-model" | "-md" | "--model-draft" =>
          value(flag).foreach(v => draftModel = Some(Paths.get(v)))
        case "--mmproj" | "-mm" =>
          value(flag).foreach(v => visionModel = Some(Paths.get(v)))
        case "-c" | "--ctx-size" => value(flag).foreach(v => context = v.toInt)
        case "--host"            => value(flag).foreach(host = _)
        case "--port"            => value(flag).foreach(v => port = v.toInt)
        case "--spec-type"       =>
          value(flag).foreach { kind =>
            multiToken = kind == "draft-mtp"
            if (!multiToken)
              notes += s"--spec-type $kind accepted, not done: the runner drafts with a model's MTP layer only"
          }
        case "--spec-draft-n-max" =>
          value(flag).foreach(v => draftMaximum = v.toInt)
        case other if NotYet.contains(other) =>
          problem = Some(s"$other is not supported yet: ${NotYet(other)}")
        case other if Harmless.contains(other) =>
          val values = (0 until Harmless(other)).flatMap(_ => value(other))
          notes += s"$other ${values.mkString(" ")} accepted: it changes how llama.cpp runs, not what it answers"
            .replace("  ", " ")
        case other => problem = Some(s"unknown flag $other")
      }
    }
    problem
      .map(Left(_))
      .getOrElse(
        model
          .toRight("no model: pass -m <file.gguf>")
          .map(
            ServerOptions(
              _,
              context,
              host,
              port,
              draftModel,
              visionModel,
              if (multiToken) draftMaximum else 0,
              notes.result()
            )
          )
      )
  }
}
