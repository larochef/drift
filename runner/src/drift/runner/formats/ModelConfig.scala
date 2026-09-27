package drift.runner.formats

import java.nio.file.{Files, Path}

/** A HuggingFace `config.json`: the hyperparameters a bare safetensors file
  * does not carry. Multimodal configs nest the language model under
  * `text_config`, which `text` finds.
  */
final class ModelConfig(val json: ujson.Value, val source: String) {

  private def field(key: String): ujson.Value =
    json.objOpt
      .flatMap(_.get(key))
      .filter(_ != ujson.Null)
      .getOrElse(throw new FormatException(s"$source has no $key"))

  def has(key: String): Boolean =
    json.objOpt.exists(_.get(key).exists(_ != ujson.Null))

  def int(key: String): Int = {
    val value = field(key).num
    if (value != value.toInt)
      throw new FormatException(s"$source: $key is $value, not an integer")
    value.toInt
  }

  def long(key: String): Long = field(key).num.toLong
  def double(key: String): Double = field(key).num
  def boolean(key: String): Boolean = field(key).bool
  def string(key: String): String = field(key).str
  def strings(key: String): Seq[String] = field(key).arr.map(_.str).toSeq
  def ints(key: String): Seq[Int] = field(key).arr.map(_.num.toInt).toSeq

  /** A nested object, such as `rope_parameters` or `mtp`. */
  def section(key: String): ModelConfig =
    new ModelConfig(field(key), s"$source/$key")

  /** The language model's section: `text_config` when there is one. */
  def text: ModelConfig =
    if (has("text_config")) section("text_config") else this
}

object ModelConfig {
  def read(path: Path): ModelConfig =
    new ModelConfig(ujson.read(Files.readString(path)), path.toString)
}
