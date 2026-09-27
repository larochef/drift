package drift.runner

import java.lang.foreign.MemorySegment

import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.tensor.{DType, KQuants, Shape}

/** The step 3 gate of `specs/42`: how fast `linear` streams its weights, next
  * to llama.cpp's `test-backend-ops perf -o MUL_MAT` on the same shapes (the
  * fork's build, with the 16384×8192 cases added). M of 1 to 8 are
  * matrix-vector passes, reported in GB/s of weights; M = 512 is the prefill
  * GEMM, reported in TFLOPS.
  *
  * `./mill runner.gpuTest.runMain drift.runner.MatVecBenchmark`
  */
object MatVecBenchmark {

  /** Qwen 3.6 35B-A3B's small shapes: router, shared expert, output
    * projections, q, and an expert's down row.
    */
  private val Shapes =
    Seq((256, 2048), (512, 2048), (2048, 4096), (8192, 2048), (2048, 512))
  private val Rows = Seq(1)
  private val Types: Seq[DType] = Seq(DType.F32, KQuants.Q4_K, KQuants.Q6_K)

  def main(arguments: Array[String]): Unit = {
    val hip = Gpu.hip
    println(hip.deviceName)
    val inputs =
      if (arguments.headOption.contains("float")) MatVecInputs.Float
      else MatVecInputs.Int8
    println(s"inputs: $inputs")
    val ops = new HipOps(hip, inputs)
    try
      for {
        (n, k) <- Shapes
        dtype <- Types
        m <- Rows
      } {
        val weight = ops.fromBytes(
          dtype,
          Shape.of(n, k),
          TestData.quantized(dtype, n, k, 1)
        )
        val x = ops.fromFloats(Shape.of(m, k), TestData.gaussian(2, m * k))
        val out = ops.allocate(DType.F32, Shape.of(m, n))
        (1 to 200).foreach(_ =>
          ops.linear(x, weight, out)
        ) // warm-up: the JIT compiles the host side
        hip.synchronize()
        val iterations = if (m > 8) 20 else 100
        val (start, end) = (hip.createEvent(), hip.createEvent())
        hip.record(start, MemorySegment.NULL)
        (1 to iterations).foreach(_ => ops.linear(x, weight, out))
        hip.record(end, MemorySegment.NULL)
        val microseconds =
          hip.elapsedMilliseconds(start, end) * 1000 / iterations
        val rate =
          if (m > 8) f"${2.0 * m * n * k / (microseconds * 1e6)}%6.1f TFLOPS"
          else f"${weight.byteSize / (microseconds * 1e3)}%6.1f GB/s"
        println(
          f"${dtype.name}%-18s [$n%5d, $k%5d] m=$m%-4d $microseconds%9.1f µs  $rate"
        )
        hip.destroyEvent(start)
        hip.destroyEvent(end)
      }
    finally ops.close()
  }
}
