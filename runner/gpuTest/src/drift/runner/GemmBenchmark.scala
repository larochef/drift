package drift.runner

import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.{Arena, MemorySegment}

import drift.runner.native.HipBlas

/** hipBLAS GEMM throughput at the diffusion transformers' shapes, F16 against
  * BF16 (`c[m, n] = a[m, k] · b[n, k]ᵀ`, F32 accumulation) and against int8
  * operands summed in I32 (`bugs/53`), on device buffers of zeros (the speed
  * does not depend on the values).
  *
  * `./mill runner.gpuTest.runMain drift.runner.GemmBenchmark`
  */
object GemmBenchmark {

  /** Krea 2 at 1024²: 512 text + 4096 image tokens, width 6144, MLP 16384. */
  private val Shapes = Seq(
    (4608, 6144, 6144),
    (4608, 1536, 6144),
    (4608, 16384, 6144),
    (4608, 6144, 16384),
    // LTX 2.5, 97 frames of 512²: 3328 video tokens, width 4096, MLP 16384
    (3328, 4096, 4096),
    (3328, 16384, 4096),
    (3328, 4096, 16384)
  )

  def main(arguments: Array[String]): Unit = {
    val hip = Gpu.hip
    val blas = new HipBlas(hip)
    try {
      int8Check(blas)
      for {
        (m, n, k) <- Shapes
        (name, dataType) <- Seq(
          "F16" -> HipBlas.RealF16,
          "BF16" -> HipBlas.RealBF16,
          "I8" -> HipBlas.RealI8
        )
      } {
        val (a, b, c) =
          (
            hip.allocate(2L * m * k),
            hip.allocate(2L * n * k),
            hip.allocate(4L * m * n)
          )
        Seq((a, 2L * m * k), (b, 2L * n * k), (c, 4L * m * n)).foreach(hip.zero)
        def product(): Unit =
          if (dataType == HipBlas.RealI8) blas.gemmInt8(a, b, c, m, n, k)
          else blas.gemm(a, b, c, m, n, k, dataType)
        (1 to 3).foreach(_ => product())
        hip.synchronize()
        val iterations = 10
        val (start, end) = (hip.createEvent(), hip.createEvent())
        hip.record(start, MemorySegment.NULL)
        (1 to iterations).foreach(_ => product())
        hip.record(end, MemorySegment.NULL)
        val milliseconds = hip.elapsedMilliseconds(start, end) / iterations
        println(
          f"$name%-5s [$m%5d, $k%5d] · [$n%5d, $k%5d]ᵀ  $milliseconds%8.2f ms  ${2.0 * m * n * k / (milliseconds * 1e9)}%6.1f TFLOPS"
        )
        hip.destroyEvent(start)
        hip.destroyEvent(end)
        Seq(a, b, c).foreach(hip.free)
      }
    } finally blas.close()
  }

  /** The int8 product on values whose sums are known: a row of x is its index
    * (mod 100) everywhere, a row of the weight −3 or 2 by parity.
    */
  private def int8Check(blas: HipBlas): Unit = {
    val hip = Gpu.hip
    val (m, n, k) = (3328, 4096, 4096)
    val arena = Arena.ofConfined()
    try {
      val (x, weight, sums) = (
        arena.allocate(m.toLong * k),
        arena.allocate(n.toLong * k),
        arena.allocate(4L * m * n)
      )
      (0 until m).foreach(row =>
        x.asSlice(row.toLong * k, k).fill((row % 100).toByte)
      )
      (0 until n).foreach(row =>
        weight
          .asSlice(row.toLong * k, k)
          .fill((if (row % 2 == 0) -3 else 2).toByte)
      )
      val (a, b, c) =
        (
          hip.allocate(m.toLong * k),
          hip.allocate(n.toLong * k),
          hip.allocate(4L * m * n)
        )
      hip.copy(a, x, m.toLong * k)
      hip.copy(b, weight, n.toLong * k)
      blas.gemmInt8(a, b, c, m, n, k)
      hip.synchronize()
      hip.copy(sums, c, 4L * m * n)
      val wrong = (0 until m).iterator
        .flatMap(row => (0 until n).iterator.map(row -> _))
        .count { (row, column) =>
          sums.getAtIndex(JAVA_INT, row.toLong * n + column) !=
            (row % 100) * (if (column % 2 == 0) -3 else 2) * k
        }
      println(
        s"I8 product [$m, $k] · [$n, $k]ᵀ: $wrong of ${m.toLong * n} sums wrong"
      )
      Seq(a, b, c).foreach(hip.free)
    } finally arena.close()
  }
}
