package drift.runner

import java.nio.file.Paths

import drift.runner.models.Models
import drift.runner.ops.{HipOps, MatVecInputs}

/** Where a decode step's time goes: the host's submission of the step's kernels
  * against the step's wall time. With a token count, each step is a speculative
  * verification of that many tokens instead (the states kept per token), as MTP
  * drafting runs it.
  *
  * `./mill runner.gpuTest.runMain drift.runner.DecodeProfile <model.gguf> [verified tokens]`
  */
object DecodeProfile {
  def main(arguments: Array[String]): Unit = {
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val model = Models.open(ops, Paths.get(arguments(0)), None)
    val sequence = model.newSequence(256, 64)
    sequence.reserve(200)
    ops.toFloats(
      model.forward(Array.range(100, 132), 0, sequence, allLogits = false)
    )
    val verified = arguments.lift(1).map(_.toInt)
    def step(position: Int): (Double, Double) = {
      val start = System.nanoTime()
      val logits = verified match {
        case Some(count) =>
          model.verify(
            Array.tabulate(count)(1000 + position + _),
            position,
            sequence
          )
        case None =>
          model.forward(
            Array(1000 + position),
            position,
            sequence,
            allLogits = false
          )
      }
      val submitted = System.nanoTime()
      ops.toFloats(logits)
      val done = System.nanoTime()
      ((submitted - start) / 1e6, (done - start) / 1e6)
    }
    (32 until 100).foreach(step) // warm-up
    val samples = (100 until 160).map(step)
    println(
      f"host submission ${samples.map(_._1).sum / samples.size}%.2f ms, whole step ${samples.map(_._2).sum / samples.size}%.2f ms"
    )
    model.close()
    ops.close()
  }
}
