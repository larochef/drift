package drift.runner.server

import drift.protocol.ChatServerFlags

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

  def parse(arguments: Seq[String]): Either[String, ServerOptions] =
    ChatServerFlags.read(arguments).flatMap { read =>
      val values = read.values
      val kind = values.get("--spec-type")
      val multiToken = kind.contains("draft-mtp")
      values
        .get("-m")
        .toRight("no model: pass -m <file.gguf>")
        .map(model =>
          ServerOptions(
            Paths.get(model),
            values.get("-c").fold(4096)(_.toInt),
            values.getOrElse("--host", "127.0.0.1"),
            values.get("--port").fold(8080)(_.toInt),
            values.get("--spec-draft-model").map(Paths.get(_)),
            values.get("--mmproj").map(Paths.get(_)),
            if (multiToken) values.get("--spec-draft-n-max").fold(2)(_.toInt)
            else 0,
            read.notes ++ kind
              .filterNot(_ => multiToken)
              .map(kind =>
                s"--spec-type $kind accepted, not done: the runner drafts with a model's MTP layer only"
              )
          )
        )
    }
}
