package drift.runner

import java.lang.foreign.MemorySegment

import drift.runner.native.HipBlas

/** hipBLAS GEMM throughput at the diffusion transformers' shapes, F16 against
  * BF16 (`c[m, n] = a[m, k] · b[n, k]ᵀ`, F32 accumulation), on device buffers
  * of zeros (the speed does not depend on the values).
  *
  * `./mill runner.gpuTest.runMain drift.runner.GemmBenchmark`
  */
object GemmBenchmark {

  /** Krea 2 at 1024²: 512 text + 4096 image tokens, width 6144, MLP 16384. */
  private val Shapes = Seq(
    (4608, 6144, 6144),
    (4608, 1536, 6144),
    (4608, 16384, 6144),
    (4608, 6144, 16384)
  )

  def main(arguments: Array[String]): Unit = {
    val hip = Gpu.hip
    val blas = new HipBlas(hip)
    try
      for {
        (m, n, k) <- Shapes
        (name, dataType) <- Seq(
          "F16" -> HipBlas.RealF16,
          "BF16" -> HipBlas.RealBF16
        )
      } {
        val (a, b, c) =
          (
            hip.allocate(2L * m * k),
            hip.allocate(2L * n * k),
            hip.allocate(2L * m * n)
          )
        Seq((a, 2L * m * k), (b, 2L * n * k), (c, 2L * m * n)).foreach(hip.zero)
        (1 to 3).foreach(_ => blas.gemm(a, b, c, m, n, k, dataType))
        hip.synchronize()
        val iterations = 10
        val (start, end) = (hip.createEvent(), hip.createEvent())
        hip.record(start, MemorySegment.NULL)
        (1 to iterations).foreach(_ => blas.gemm(a, b, c, m, n, k, dataType))
        hip.record(end, MemorySegment.NULL)
        val milliseconds = hip.elapsedMilliseconds(start, end) / iterations
        println(
          f"$name%-5s [$m%5d, $k%5d] · [$n%5d, $k%5d]ᵀ  $milliseconds%8.2f ms  ${2.0 * m * n * k / (milliseconds * 1e9)}%6.1f TFLOPS"
        )
        hip.destroyEvent(start)
        hip.destroyEvent(end)
        Seq(a, b, c).foreach(hip.free)
      }
    finally blas.close()
  }
}
