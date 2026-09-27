package drift.runner

import java.nio.file.Paths
import java.time.LocalDateTime

import drift.runner.decode.{Generator, Sampler, Sampling, Speculation}
import drift.runner.models.Models
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.text.ChatTemplate

/** A first conversation with a real model, end to end on the GPU: the GGUF's
  * own tokenizer and chat template, greedy decoding, text streamed; twice, the
  * second run warm. A draft count drafts with the model's MTP layer.
  *
  * `./mill runner.gpuTest.runMain drift.runner.ChatDemo <model.gguf> "<prompt>"
  * [float|int8] [drafts] [MTP head GGUF]`
  */
object ChatDemo {
  def main(arguments: Array[String]): Unit = {
    val path = Paths.get(arguments(0))
    val prompt = arguments
      .lift(1)
      .getOrElse("Explain in two sentences why the sky is blue.")
    val inputs =
      if (arguments.lift(2).contains("int8")) MatVecInputs.Int8
      else MatVecInputs.Float
    val ops = new HipOps(Gpu.hip, inputs)
    val model = Models.open(ops, path, arguments.lift(4).map(Paths.get(_)))
    val gguf = TokenizerFiles.ggufTokenizer(path)
    val (file, mapped) = drift.runner.formats.Gguf.open(path)
    val template = new ChatTemplate(
      file.string("tokenizer.chat_template"),
      () => LocalDateTime.now()
    )
    mapped.close()
    val text = template.render(
      ujson.Arr(ujson.Obj("role" -> "user", "content" -> prompt)),
      None,
      addGenerationPrompt = true,
      Map("enable_thinking" -> ujson.False)
    )
    val ids = gguf.encode(text, addSpecial = true)
    println(
      s"${path.getFileName}, inputs $inputs\nprompt: ${ids.length} tokens\n---"
    )
    val generator = new Generator(
      ops,
      model,
      gguf,
      context = 4096,
      pageSize = 64,
      prefillChunk = 512,
      Speculation.multiToken(
        ops,
        model,
        arguments.lift(3).fold(0)(_.toInt),
        prefillChunk = 512
      )
    )
    val stops = Seq("<|im_end|>", "<|endoftext|>").flatMap(gguf.id).toSet
    (1 to 2).foreach { _ =>
      val result = generator.generate(
        ids,
        200,
        new Sampler(Sampling.Greedy),
        stops,
        piece => { print(piece); System.out.flush(); true }
      )
      println(
        f"\n---\nprompt ${result.promptTokens} tokens at ${result.promptTokensPerSecond}%.0f t/s, ${result.ids.size} tokens at ${result.tokensPerSecond}%.1f t/s, ${result.accepted} of ${result.drafted} drafts accepted"
      )
    }
    generator.close()
    model.close()
    ops.close()
  }
}
