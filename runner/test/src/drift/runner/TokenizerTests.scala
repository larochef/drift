package drift.runner

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import utest.*

import java.nio.file.Path

import drift.runner.text.{Tokenizer, TokenizerJson}

/** The runner's tokenizers against HuggingFace's own (the Rust library, through
  * DJL) on the models' official `tokenizer.json`: the same ids for the corpus
  * and for seeded random text, the same decoded text; the GGUF loader against
  * the JSON one; streaming decoding against whole decoding.
  */
object TokenizerTests extends TestSuite {

  private val texts =
    TokenizerCorpus.cases ++ TokenizerCorpus.random(seed = 42, count = 2000)

  private lazy val loaded: Map[String, Tokenizer] =
    TokenizerFiles.Repositories
      .map(r => r -> TokenizerJson.load(TokenizerFiles.json(r)))
      .toMap

  private def reference(repository: String): HuggingFaceTokenizer =
    referenceAt(TokenizerFiles.json(repository))

  private def referenceAt(path: Path): HuggingFaceTokenizer =
    HuggingFaceTokenizer
      .builder()
      .optTokenizerPath(path)
      .optAddSpecialTokens(true)
      .optTruncation(false)
      .optPadding(false)
      .build()

  private def show(text: String) = text.take(60).flatMap {
    case c if c < ' ' || c > '~' => f"\\u${c.toInt}%04x"
    case c                       => c.toString
  }

  private def againstReference(repository: String): Unit =
    against(repository, loaded(repository), reference(repository))

  private def against(
      repository: String,
      ours: Tokenizer,
      theirs: HuggingFaceTokenizer
  ): Unit = {
    try {
      val mismatches = texts.filter { text =>
        !ours
          .encode(text, addSpecial = true)
          .sameElements(theirs.encode(text).getIds.map(_.toInt))
      }
      mismatches.take(3).foreach { text =>
        println(s"  $repository differs on \"${show(text)}\"")
        println(
          s"    ours:   ${ours.encode(text, addSpecial = true).take(20).mkString(" ")}"
        )
        println(
          s"    theirs: ${theirs.encode(text).getIds.take(20).mkString(" ")}"
        )
      }
      println(
        s"  $repository: ${texts.size - mismatches.size}/${texts.size} texts encode the same"
      )
      assert(mismatches.isEmpty)
      val decodeMismatches = texts.filter { text =>
        val ids = theirs.encode(text).getIds
        ours.decode(ids.map(_.toInt).toSeq, skipSpecial = false) != theirs
          .decode(ids, false)
      }
      decodeMismatches
        .take(3)
        .foreach(text => println(s"  decode differs on \"${show(text)}\""))
      assert(decodeMismatches.isEmpty)
    } finally theirs.close()
  }

  val tests = Tests {
    test("Qwen3 (Z-Image, Flux.2 Klein text encoders)") {
      againstReference("Qwen/Qwen3-4B")
    }
    test("Qwen 3.6") { againstReference("Qwen/Qwen3.6-35B-A3B") }
    test("Qwen 3.8 Flash Next") { againstReference("Qwen/Qwen3.8-Flash-Next") }
    test("Gemma 4") { againstReference("google/gemma-4-26b-a4b-it") }
    test("gpt-oss") { againstReference("openai/gpt-oss-20b") }
    test("Mistral Small's Tekken (FLUX.2 [dev]'s text encoder)") {
      val path = TokenizerFiles.tekkenJson
      val ours = TokenizerJson.load(path)
      against("Mistral Small (Tekken)", ours, referenceAt(path))
      TokenizerFiles.gguf(TokenizerFiles.MistralGguf) match {
        case None =>
          println(
            s"  ${TokenizerFiles.MistralGguf} is not in the cache: skipped"
          )
        case Some(gguf) =>
          val fromGguf = TokenizerFiles.ggufTokenizer(gguf)
          val mismatches = texts.filter(text =>
            !fromGguf
              .encode(text, addSpecial = true)
              .sameElements(ours.encode(text, addSpecial = true))
          )
          mismatches
            .take(3)
            .foreach(text =>
              println(s"  the GGUF differs on \"${show(text)}\"")
            )
          assert(mismatches.isEmpty)
      }
    }
    test("GGUF vocabularies tokenize as their tokenizer.json") {
      TokenizerFiles.Ggufs.foreach { (repository, fileName) =>
        TokenizerFiles.gguf(fileName) match {
          case None =>
            println(s"  $fileName is not in the HuggingFace cache: skipped")
          case Some(path) =>
            val fromGguf = TokenizerFiles.ggufTokenizer(path)
            val fromJson = loaded(repository)
            val mismatches = texts.filter(text =>
              !fromGguf
                .encode(text, addSpecial = false)
                .sameElements(fromJson.encode(text, addSpecial = false))
            )
            mismatches
              .take(3)
              .foreach(text =>
                println(s"  $fileName differs on \"${show(text)}\"")
              )
            println(
              s"  $fileName: ${texts.size - mismatches.size}/${texts.size} texts the same"
            )
            assert(mismatches.isEmpty)
        }
      }
    }
    test("encoding speed, for the record") {
      val text = (TokenizerCorpus.cases ++ TokenizerCorpus.random(7, 3000))
        .mkString("\n")
      loaded.foreach { (repository, tokenizer) =>
        tokenizer.encode(text, addSpecial = false) // warm-up
        val start = System.nanoTime()
        val ids = tokenizer.encode(text + " fresh", addSpecial = false)
        val milliseconds = (System.nanoTime() - start) / 1e6
        println(
          f"  $repository: ${text.length / 1024}%d K characters, ${ids.length}%d tokens in $milliseconds%.1f ms"
        )
      }
    }
    test(
      "streaming decoding releases whole characters and ends with the same text"
    ) {
      loaded.values.foreach { tokenizer =>
        texts.take(60).foreach { text =>
          val ids = tokenizer.encode(text, addSpecial = false)
          val decoder = tokenizer.streamingDecoder(skipSpecial = false)
          val pieces = ids.map(decoder.next) :+ decoder.finish()
          assert(
            pieces
              .forall(piece => !piece.exists(_ == '�') || text.contains('�'))
          )
          assert(
            pieces.mkString == tokenizer.decode(ids.toSeq, skipSpecial = false)
          )
        }
      }
    }
  }
}
