package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.Ops

import java.nio.file.{Files, Path}

/** Opens the model a file holds, by its architecture: a GGUF's
  * `general.architecture`, or `model_type` in the `config.json` beside
  * safetensors.
  */
object Models {

  private val openers: Map[String, (Ops, Path, Option[Path]) => CausalModel] =
    Map(
      "qwen3" -> { (ops, path, draftModel) =>
        draftModel.foreach(draft =>
          throw new FormatException(
            s"$path has no MTP head; it takes no draft model ($draft)"
          )
        )
        Qwen3.open(ops, path)
      },
      "qwen3vl" -> { (ops, path, draftModel) =>
        draftModel.foreach(draft =>
          throw new FormatException(
            s"$path has no MTP head; it takes no draft model ($draft)"
          )
        )
        Qwen3.open(ops, path)
      },
      "qwen35" -> Qwen35.open,
      "qwen3_5" -> Qwen35.open,
      "qwen3_5_text" -> Qwen35.open,
      "qwen35moe" -> Qwen35.open,
      "qwen3_5_moe" -> Qwen35.open,
      "qwen3_5_moe_text" -> Qwen35.open,
      "qwen4exp" -> Qwen4Exp.open,
      "qwen4_exp" -> Qwen4Exp.open,
      "qwen4_exp_text" -> Qwen4Exp.open
    )

  /** `draftModel`: an MTP head in a file of its own, for models that take one.
    */
  def open(ops: Ops, path: Path, draftModel: Option[Path]): CausalModel = {
    val architecture =
      if (path.getFileName.toString.endsWith(".gguf")) {
        val (file, mapped) = Gguf.open(path)
        try file.architecture
        finally mapped.close()
      } else {
        val config = path.resolveSibling("config.json")
        if (!Files.isRegularFile(config))
          throw new FormatException(s"no config.json beside $path")
        ModelConfig.read(config).string("model_type")
      }
    openers.get(architecture) match {
      case Some(open) => open(ops, path, draftModel)
      case None       =>
        throw new FormatException(
          s"$path is a '$architecture' model; the runner runs ${openers.keys.toSeq.sorted.mkString(", ")} so far"
        )
    }
  }
}
