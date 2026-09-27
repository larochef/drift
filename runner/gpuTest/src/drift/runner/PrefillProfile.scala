package drift.runner

import java.nio.file.Paths

import drift.runner.models.Models
import drift.runner.ops.{HipOps, MatVecInputs}

/** How fast a prompt runs: a prefill of `tokens` (512 when not given) in one
  * chunk, after a warm-up prefill of the same length.
  *
  * `./mill runner.gpuTest.runMain drift.runner.PrefillProfile <model.gguf> [tokens]`
  */
object PrefillProfile {
  def main(arguments: Array[String]): Unit = {
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val model = Models.open(ops, Paths.get(arguments(0)), None)
    val tokens = arguments.lift(1).fold(512)(_.toInt)
    val sequence = model.newSequence(tokens, 64)
    def prefill(): Double = {
      sequence.reset()
      sequence.reserve(tokens)
      val start = System.nanoTime()
      ops.toFloats(
        model.forward(
          Array.tabulate(tokens)(i => 1000 + i * 7 % 5000),
          0,
          sequence,
          allLogits = false
        )
      )
      (System.nanoTime() - start) / 1e9
    }
    prefill() // warm-up
    val seconds = (1 to 2).map(_ => prefill()).min
    println(
      f"prefill of $tokens tokens: $seconds%.2f s, ${tokens / seconds}%.0f tok/s"
    )
    model.close()
    ops.close()
  }
}
