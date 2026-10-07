package drift.runner

import java.nio.file.Paths

import drift.runner.text.TokenizerJson

/** A `tokenizer.json`'s ids for each text given, one line each: to set beside
  * HuggingFace's own.
  *
  * `./mill runner.gpuTest.runMain drift.runner.TokenizerPrint <tokenizer.json> <text>…`
  */
object TokenizerPrint {
  def main(arguments: Array[String]): Unit = {
    val tokenizer = TokenizerJson.load(Paths.get(arguments.head))
    arguments.tail.foreach(text =>
      println(
        tokenizer
          .encode(text.replace("\\n", "\n"), addSpecial = true)
          .mkString("[", ", ", "]")
      )
    )
  }
}
