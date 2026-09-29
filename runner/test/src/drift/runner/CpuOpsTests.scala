package drift.runner

import utest.*

import drift.runner.ops.CpuOps
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

/** The reference backend against values worked out by hand. */
object CpuOpsTests extends TestSuite {

  val tests = Tests {
    test("add") {
      val ops = new CpuOps
      try {
        val shape = Shape.of(2, 2)
        val out = ops.allocate(DType.F32, shape)
        ops.add(
          ops.fromFloats(shape, Array(1f, 2f, 3f, 4f)),
          ops.fromFloats(shape, Array(10f, 20f, 30f, 40f)),
          out
        )
        assert(ops.toFloats(out).sameElements(Array(11f, 22f, 33f, 44f)))
      } finally ops.close()
    }
    test("rmsNorm normalises each row by its root mean square") {
      val ops = new CpuOps
      try {
        // row 1: mean(1, 4, 9, 16) = 7.5; row 2: mean(4 × 4) = 4
        val x = ops.fromFloats(
          Shape.of(2, 4),
          Array(1f, 2f, 3f, 4f, 2f, -2f, 2f, -2f)
        )
        val weight = ops.fromFloats(Shape.of(4), Array(1f, 1f, 1f, 2f))
        val out = ops.allocate(DType.F32, Shape.of(2, 4))
        ops.rmsNorm(x, weight, 0f, 0f, out)
        val r = math.sqrt(7.5)
        val expected =
          Array(1 / r, 2 / r, 3 / r, 8 / r, 1.0, -1.0, 1.0, -2.0).map(_.toFloat)
        val comparison =
          Comparison.of(expected, ops.toFloats(out), Tolerance(1e-6, 1e-6))
        assert(comparison.passed)
      } finally ops.close()
    }
    test("epsilon keeps a zero row finite") {
      val ops = new CpuOps
      try {
        val out = ops.allocate(DType.F32, Shape.of(1, 3))
        ops.rmsNorm(
          ops.fromFloats(Shape.of(1, 3), Array(0f, 0f, 0f)),
          ops.fromFloats(Shape.of(3), Array(1f, 1f, 1f)),
          1e-6f,
          0f,
          out
        )
        assert(ops.toFloats(out).forall(_ == 0f))
      } finally ops.close()
    }
    test("mismatched shapes are refused") {
      val ops = new CpuOps
      try {
        val error = assertThrows[IllegalArgumentException] {
          ops.add(
            ops.fromFloats(Shape.of(2), Array(1f, 2f)),
            ops.fromFloats(Shape.of(3), Array(1f, 2f, 3f)),
            ops.allocate(DType.F32, Shape.of(2))
          )
        }
        assert(error.getMessage.contains("equal shapes"))
      } finally ops.close()
    }
  }
}
