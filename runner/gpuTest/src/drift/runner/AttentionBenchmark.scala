package drift.runner

import java.lang.foreign.MemorySegment

import drift.runner.ops.{Attention, HipOps, MatVecInputs}
import drift.runner.tensor.{DType, Shape}

/** Non-causal attention throughput at the diffusion transformers' shapes: 32
  * heads of 128 over one page, FLUX.2 at 1024² alone (512 text + 4096 image
  * tokens) and with one 1024² reference (+ 4096). A step runs it 32 times
  * (FLUX.2 [klein] 9B's blocks).
  *
  * `./mill runner.gpuTest.runMain drift.runner.AttentionBenchmark`
  */
object AttentionBenchmark {

  def main(arguments: Array[String]): Unit = {
    val hip = Gpu.hip
    val ops = new HipOps(hip, MatVecInputs.Float)
    val (heads, d) = (32, 128)
    try
      for (tokens <- Seq(4608, 8704)) {
        val shape = Shape.of(tokens, heads, d)
        def random(seed: Long) =
          ops.fromFloats(shape, TestData.gaussian(seed, tokens * heads * d))
        val (q, k, v) = (random(1), random(2), random(3))
        val out = ops.allocate(DType.F32, shape)
        val cache = ops.allocateCache(1, (tokens + 15) / 16 * 16, heads, d)
        val pageTable = ops.fromInts(Shape.of(1), Array(0))
        ops.cacheWrite(k, v, cache, pageTable, 0)
        val attention =
          Attention(
            (1 / math.sqrt(d)).toFloat,
            causal = false,
            None,
            None,
            None
          )
        def run() =
          ops.attention(q, cache, pageTable, 0, tokens, attention, out)
        run()
        hip.synchronize()
        val iterations = 5
        val (start, end) = (hip.createEvent(), hip.createEvent())
        hip.record(start, MemorySegment.NULL)
        (1 to iterations).foreach(_ => run())
        hip.record(end, MemorySegment.NULL)
        val milliseconds = hip.elapsedMilliseconds(start, end) / iterations
        val flops = 4.0 * tokens * tokens * d * heads
        println(
          f"$tokens%5d tokens: $milliseconds%8.2f ms  ${flops / (milliseconds * 1e9)}%6.1f TFLOPS  (× 32 blocks: ${32 * milliseconds / 1000}%.2f s a step)"
        )
        hip.destroyEvent(start)
        hip.destroyEvent(end)
        Seq(q, k, v, out, pageTable, cache.keys, cache.values).foreach(
          ops.release
        )
      }
    finally ops.close()
  }
}
