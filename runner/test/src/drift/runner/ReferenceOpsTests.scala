package drift.runner

import utest.*

import drift.runner.ops.{Activation, CpuOps, Rope, RopeLayout, RopeSections}
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

/** The reference backend's operations against values known independently:
  * tables, identities, hand-worked rotations.
  */
object ReferenceOpsTests extends TestSuite {

  private def withCpu[A](body: CpuOps => A): A = {
    val ops = new CpuOps
    try body(ops)
    finally ops.close()
  }

  private def close(
      expected: Seq[Double],
      actual: Array[Float],
      tolerance: Double
  ): Unit = {
    val comparison = Comparison.of(
      expected.map(_.toFloat).toArray,
      actual,
      Tolerance(tolerance, 0)
    )
    assert(comparison.passed)
  }

  val tests = Tests {
    test("erf against tabulated values") {
      // Abramowitz & Stegun, table 7.1
      Seq(
        0.0 -> 0.0,
        0.5 -> 0.5204998778,
        1.0 -> 0.8427007929,
        2.0 -> 0.9953222650,
        3.0 -> 0.9999779095
      )
        .foreach { (x, erf) =>
          assert(math.abs(CpuOps.erf(x) - erf) < 1e-9)
          assert(math.abs(CpuOps.erf(-x) + erf) < 1e-9)
        }
    }
    test("activations at known points") {
      withCpu { ops =>
        val x = ops.fromFloats(Shape.of(3), Array(-1f, 0f, 2f))
        val out = ops.allocate(DType.F32, Shape.of(3))
        ops.activation(Activation.Silu, x, out)
        close(
          Seq(-1 / (1 + math.E), 0, 2 / (1 + math.exp(-2))),
          ops.toFloats(out),
          1e-7
        )
        ops.activation(Activation.Sigmoid, x, out)
        close(
          Seq(1 / (1 + math.E), 0.5, 1 / (1 + math.exp(-2))),
          ops.toFloats(out),
          1e-7
        )
        ops.activation(Activation.GeluErf, x, out)
        close(Seq(-0.1586552539, 0, 1.9544997361), ops.toFloats(out), 1e-7)
        ops.activation(
          Activation.GeluTanh,
          x,
          out
        ) // within 1e-3 of the exact GELU
        close(Seq(-0.1586552539, 0, 1.9544997361), ops.toFloats(out), 1e-3)
      }
    }
    test("softmax rows sum to one and keep their order") {
      withCpu { ops =>
        val x =
          ops.fromFloats(Shape.of(2, 3), Array(1f, 2f, 3f, 1000f, 1000f, 1000f))
        val out = ops.allocate(DType.F32, Shape.of(2, 3))
        ops.softmax(x, 1f, out)
        val values = ops.toFloats(out)
        val e = Seq(math.exp(-2), math.exp(-1), 1.0)
        close(
          e.map(_ / e.sum) ++ Seq.fill(3)(1.0 / 3),
          values,
          1e-7
        ) // no overflow at 1000
      }
    }
    test("layer norm gives zero mean and unit variance") {
      withCpu { ops =>
        val x = ops.fromFloats(Shape.of(1, 4), Array(1f, 2f, 3f, 6f))
        val out = ops.allocate(DType.F32, Shape.of(1, 4))
        ops.layerNorm(x, None, None, 0f, out)
        val values = ops.toFloats(out).map(_.toDouble)
        assert(math.abs(values.sum) < 1e-6)
        assert(math.abs(values.map(v => v * v).sum / 4 - 1) < 1e-6)
      }
    }
    test("RoPE turns each pair by position × frequency") {
      withCpu { ops =>
        // one token at position 1, one head of 4: pairs turn by 1 and 1/√theta
        val theta = 100f
        val x = ops.fromFloats(Shape.of(1, 1, 4), Array(1f, 1f, 0f, 0f))
        val positions = ops.fromInts(Shape.of(1), Array(1))
        val out = ops.allocate(DType.F32, Shape.of(1, 1, 4))
        ops.rope(
          x,
          positions,
          Rope(theta, 4, RopeLayout.Interleaved, RopeSections.Single),
          out
        )
        val (a0, a1) = (1.0, 0.1) // θ^0 and θ^(-2/4)
        // interleaved pairs (0,1) and (2,3): (1, 1) turned by a0, (0, 0) stays
        close(
          Seq(math.cos(a0) - math.sin(a0), math.sin(a0) + math.cos(a0), 0, 0),
          ops.toFloats(out),
          1e-6
        )
        ops.rope(
          x,
          positions,
          Rope(theta, 4, RopeLayout.Neox, RopeSections.Single),
          out
        )
        // NeoX pairs (0,2) and (1,3): (1, 0) turned by a0, (1, 0) by a1
        close(
          Seq(math.cos(a0), math.cos(a1), math.sin(a0), math.sin(a1)),
          ops.toFloats(out),
          1e-6
        )
      }
    }
    test(
      "partial RoPE leaves the rest of the head, position 0 changes nothing"
    ) {
      withCpu { ops =>
        val values = Array(1f, 2f, 3f, 4f, 5f, 6f)
        val x = ops.fromFloats(Shape.of(1, 1, 6), values)
        val out = ops.allocate(DType.F32, Shape.of(1, 1, 6))
        ops.rope(
          x,
          ops.fromInts(Shape.of(1), Array(7)),
          Rope(1e4f, 2, RopeLayout.Neox, RopeSections.Single),
          out
        )
        assert(ops.toFloats(out).drop(2).sameElements(values.drop(2)))
        ops.rope(
          x,
          ops.fromInts(Shape.of(1), Array(0)),
          Rope(1e4f, 6, RopeLayout.Neox, RopeSections.Single),
          out
        )
        assert(ops.toFloats(out).sameElements(values))
      }
    }
    test("mRoPE with three equal positions is plain RoPE") {
      withCpu { ops =>
        val x = ops.fromFloats(Shape.of(2, 1, 64), TestData.gaussian(1, 128))
        val (plain, sectioned) = (
          ops.allocate(DType.F32, Shape.of(2, 1, 64)),
          ops.allocate(DType.F32, Shape.of(2, 1, 64))
        )
        ops.rope(
          x,
          ops.fromInts(Shape.of(2), Array(5, 9)),
          Rope(1e6f, 64, RopeLayout.Neox, RopeSections.Single),
          plain
        )
        for (
          sections <- Seq(
            RopeSections.Contiguous(11, 11, 10),
            RopeSections.Interleaved(11, 11, 10)
          )
        ) {
          ops.rope(
            x,
            ops.fromInts(Shape.of(3, 2), Array(5, 9, 5, 9, 5, 9)),
            Rope(1e6f, 64, RopeLayout.Neox, sections),
            sectioned
          )
          assert(ops.toFloats(sectioned).sameElements(ops.toFloats(plain)))
        }
      }
    }
    test("conversions round to nearest even") {
      withCpu { ops =>
        // 1 + 2⁻⁸ is halfway between two bf16 values: it rounds to the even one, 1
        val values = Array(
          1f,
          1f + math.pow(2, -8).toFloat,
          1f + 3 * math.pow(2, -8).toFloat,
          65504f,
          -0.5f
        )
        val x = ops.fromFloats(Shape.of(5), values)
        val (bf16, back) = (
          ops.allocate(DType.BF16, Shape.of(5)),
          ops.allocate(DType.F32, Shape.of(5))
        )
        ops.convert(x, bf16)
        ops.convert(bf16, back)
        assert(
          ops
            .toFloats(back)
            .sameElements(
              Array(1f, 1f, 1f + math.pow(2, -6).toFloat, 65536f, -0.5f)
            )
        )
        val f16 = ops.allocate(DType.F16, Shape.of(5))
        ops.convert(x, f16)
        ops.convert(f16, back)
        assert(ops.toFloats(back).sameElements(values))
      }
    }
  }
}
