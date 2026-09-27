package drift.runner

import utest.*

import drift.runner.ops.{CpuOps, HipOps, MatVecInputs, Ops}
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

/** Every kernel against the `Cpu` backend, on awkward shapes. */
object HipOpsTests extends TestSuite {

  /** Runs `run` on both backends for each shape and compares the results. */
  private def againstCpu(shapes: Seq[Shape], tolerance: Tolerance)(
      run: (Ops, Shape) => Array[Float]
  ): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
    try
      shapes.foreach { shape =>
        val comparison =
          Comparison.of(run(cpu, shape), run(hip, shape), tolerance)
        println(s"  $shape: ${comparison.summary}")
        assert(comparison.passed)
      }
    finally {
      hip.close()
      cpu.close()
    }
  }

  val tests = Tests {
    test("the device under test") {
      val (free, total) = Gpu.hip.memoryInfo
      println(
        s"  ${Gpu.hip.deviceName} from ${Gpu.hip.rocmRoot}: ${free >> 20} of ${total >> 20} MiB free"
      )
      assert(total > 0)
    }
    test("values survive a round trip through device memory") {
      val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
      try {
        val values = TestData.gaussian(1, 1000)
        assert(
          hip
            .toFloats(hip.fromFloats(Shape.of(1000), values))
            .sameElements(values)
        )
      } finally hip.close()
    }
    test("add") {
      againstCpu(
        Seq(
          Shape.of(1),
          Shape.of(7),
          Shape.of(256),
          Shape.of(3, 333),
          Shape.of(1000003)
        ),
        Tolerance.Exact
      ) { (ops, shape) =>
        val count = shape.elementCount.toInt
        val out = ops.allocate(DType.F32, shape)
        ops.add(
          ops.fromFloats(shape, TestData.gaussian(2, count)),
          ops.fromFloats(shape, TestData.gaussian(3, count)),
          out
        )
        ops.toFloats(out)
      }
    }
    test("rmsNorm") {
      againstCpu(
        Seq(
          Shape.of(1, 1),
          Shape.of(3, 7),
          Shape.of(2, 255),
          Shape.of(5, 4097),
          Shape.of(64, 2560)
        ),
        Tolerance(1e-5, 1e-5)
      ) { (ops, shape) =>
        val columns = shape.last
        val out = ops.allocate(DType.F32, shape)
        ops.rmsNorm(
          ops.fromFloats(shape, TestData.gaussian(4, shape.elementCount.toInt)),
          ops
            .fromFloats(Shape.of(columns), TestData.gaussian(5, columns.toInt)),
          1e-6f,
          0f,
          out
        )
        ops.toFloats(out)
      }
    }
  }
}
