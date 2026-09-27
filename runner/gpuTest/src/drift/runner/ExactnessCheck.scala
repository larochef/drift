package drift.runner

import java.nio.file.Paths
import java.time.LocalDateTime

import drift.runner.models.Models
import drift.runner.ops.{CpuOps, HipOps, MatVecInputs, Ops}
import drift.runner.text.ChatTemplate

/** Which GPU path is closest to the exact result: the reference backend (double
  * sums, x never rounded) computes a prompt's next-token log-probabilities on a
  * real GGUF, and the GPU computes them through the prefill GEMM (F16 operands)
  * and through the matrix-vector path (int8 x).
  *
  * `./mill runner.gpuTest.runMain drift.runner.ExactnessCheck <model.gguf>`
  */
object ExactnessCheck {

  def main(arguments: Array[String]): Unit = {
    val path = Paths.get(arguments(0))
    val tokenizer = TokenizerFiles.ggufTokenizer(path)
    val (file, mapped) = drift.runner.formats.Gguf.open(path)
    val template = new ChatTemplate(
      file.string("tokenizer.chat_template"),
      () => LocalDateTime.now()
    )
    mapped.close()
    val prompt = tokenizer.encode(
      template.render(
        ujson.Arr(
          ujson.Obj(
            "role" -> "user",
            "content" -> "List three facts about the Moon, one line each."
          )
        ),
        None,
        addGenerationPrompt = true,
        Map("enable_thinking" -> ujson.False)
      ),
      addSpecial = true
    )

    def logProbabilities(ops: Ops, chunk: Int): Array[Double] = {
      val model = Models.open(ops, path, None)
      try {
        val pageSize = 64
        val sequence = model.newSequence(128, pageSize)
        sequence.reserve(prompt.length)
        var logits = Array.emptyFloatArray
        prompt.grouped(chunk).zipWithIndex.foreach { (part, i) =>
          logits = ops.toFloats(
            model.forward(part, i * chunk, sequence, allLogits = false)
          )
        }
        val largest = logits.max.toDouble
        val total =
          math.log(logits.map(l => math.exp(l - largest)).sum) + largest
        logits.map(_ - total)
      } finally model.close()
    }

    val started = System.nanoTime()
    val cpu = new CpuOps
    val exact =
      try logProbabilities(cpu, prompt.length)
      finally cpu.close()
    println(
      f"reference: ${(System.nanoTime() - started) / 1e9}%.0f s for ${prompt.length} tokens"
    )
    val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
    val gemm = logProbabilities(hip, prompt.length)
    val matVec = logProbabilities(hip, 8)
    hip.close()
    val floatHip = new HipOps(Gpu.hip, MatVecInputs.Float)
    val floatX = logProbabilities(floatHip, 8)
    floatHip.close()

    val top = exact.indices.sortBy(-exact(_)).take(10)
    top.foreach { id =>
      println(
        f"  ${ujson.write(tokenizer.decode(Seq(id), skipSpecial = false))}%-12s exact ${exact(id)}%8.4f   GEMM ${gemm(id)}%8.4f   int8 x ${matVec(id)}%8.4f   float x ${floatX(id)}%8.4f"
      )
    }
    def worst(path: Array[Double]) =
      top.map(id => math.abs(path(id) - exact(id))).max
    println(
      f"worst gap on the exact top 10: GEMM ${worst(gemm)}%.4f, int8 x ${worst(matVec)}%.4f, float x ${worst(floatX)}%.4f"
    )
  }
}
