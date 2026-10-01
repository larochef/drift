package drift.runner

import utest.*

import drift.runner.ops.{CpuOps, HipOps, MatVecInputs}
import drift.runner.tensor.{DType, Shape}

/** MiniMax H3 beyond text to video on the kernels: LoRAs, the encoders, the
  * keyframes' presentation and step, the guides' step, the references'
  * presentation and step, the ControlNet's step.
  */
object MiniMaxH3GpuTests extends TestSuite {

  /** A `HipOps` with BF16 products, as the pipeline runs. */
  private def withOps[A](run: HipOps => A): A = {
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    ops.wideProducts = true
    try run(ops)
    finally ops.close()
  }

  /** `conv3x3` with F16 weights on the kernels against the reference backend,
    * reflect-padded at strides 1 and 2: the worst difference relative to the
    * largest output.
    */
  private def halfConvolutionError(
      in: Int,
      out: Int,
      height: Int,
      width: Int
  ): Double = {
    val random = new java.util.Random(in * 31 + out)
    val x = Array.fill(height * width * in)(random.nextGaussian().toFloat)
    val w = Array.fill(out * in * 9)((random.nextGaussian() * 0.1).toFloat)
    val b = Array.fill(out)(random.nextGaussian().toFloat)
    def run(ops: drift.runner.ops.Ops): Seq[Array[Float]] = Seq(1, 2).map {
      stride =>
        val weight = ops.allocate(DType.F16, Shape.of(out, in * 9L))
        ops.convert(ops.fromFloats(Shape.of(out, in * 9L), w), weight)
        val result = ops.allocate(
          DType.F32,
          Shape.of(height / stride, width / stride, out)
        )
        ops.conv3x3(
          ops.fromFloats(Shape.of(height, width, in), x),
          weight,
          ops.fromFloats(Shape.of(out), b),
          result,
          stride,
          reflect = true
        )
        ops.toFloats(result)
    }
    val cpu = new CpuOps
    val expected =
      try run(cpu)
      finally cpu.close()
    withOps { ops =>
      run(ops)
        .zip(expected)
        .map((actual, wanted) =>
          TinyMiniMaxH3Case.relativeError(actual, wanted)
        )
        .max
    }
  }

  val tests = Tests {
    test("3×3 convolutions, F16 weights, reflect-padded") {
      Seq(
        (3, 8, 64, 64),
        (8, 8, 32, 32),
        (8, 16, 16, 16),
        (16, 16, 8, 8),
        (128, 128, 16, 16)
      ).foreach { (in, out, height, width) =>
        val error = halfConvolutionError(in, out, height, width)
        println(
          f"  $in → $out at $height × $width: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-3)
      }
    }
    test("MiniMax H3's LoRAs, full and pruned files") {
      withOps { ops =>
        Seq(false, true).foreach { pruned =>
          val (error, _) = TinyMiniMaxH3Case.loraError(ops, pruned)
          println(
            f"  pruned $pruned: worst velocity error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 2e-2) // BF16 products
        }
      }
    }
    test("MiniMax H3's video encoder (reflect-padded convolutions)") {
      withOps { ops =>
        val errors = TinyMiniMaxH3Case.videoEncoderErrors(ops)
        println(
          errors
            .map(e => f"${e * 100}%.4f%%")
            .mkString("  errors: ", ", ", " of the largest")
        )
        assert(errors.forall(_ < 1e-2)) // F16 patches
      }
    }
    test("MiniMax H3's audio encoder (Snake)") {
      withOps { ops =>
        val error = TinyMiniMaxH3Case.audioEncoderError(ops)
        println(f"  worst latent error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      }
    }
    test("MiniMax H3's presentation: two keyframes through Qwen3-VL") {
      withOps { ops =>
        val error = TinyMiniMaxH3Case.presentationError(ops)
        println(
          f"  worst hidden-state error ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-2)
      }
    }
    test("MiniMax H3's fl2va step") {
      withOps { ops =>
        val errors = TinyMiniMaxH3Case.fl2vaErrors(ops)
        println(
          f"  velocities ${errors(4) * 100}%.4f%% and ${errors(5) * 100}%.4f%% of the largest"
        )
        assert(errors(4) < 2e-2, errors(5) < 2e-2)
      }
    }
    test("MiniMax H3's guides step") {
      withOps { ops =>
        val errors = TinyMiniMaxH3Case.guidesErrors(ops)
        println(
          f"  velocities ${errors(2) * 100}%.4f%% and ${errors(3) * 100}%.4f%% of the largest"
        )
        assert(errors(2) < 2e-2, errors(3) < 2e-2)
      }
    }
    test("MiniMax H3's ref2va presentation: an image, two clips' frame pairs") {
      withOps { ops =>
        val (ids, rows, hidden, _) =
          TinyMiniMaxH3ReferencesCase.presentationErrors(ops)
        println(
          f"  worst hidden-state error ${hidden * 100}%.4f%% of the largest"
        )
        assert(ids == 0, rows == 0, hidden < 2e-2)
      }
    }
    test("MiniMax H3's ref2va step") {
      withOps { ops =>
        val errors = TinyMiniMaxH3ReferencesCase.ref2vaErrors(ops)
        println(
          f"  velocities ${errors(4) * 100}%.4f%% and ${errors(5) * 100}%.4f%% of the largest"
        )
        assert(errors(4) < 2e-2, errors(5) < 2e-2)
      }
    }
    test("MiniMax H3's Fun ControlNet step, with and without a mask") {
      withOps { ops =>
        TinyMiniMaxH3ReferencesCase.controlErrors(ops).foreach { errors =>
          println(
            f"  velocities ${errors(2) * 100}%.4f%% and ${errors(3) * 100}%.4f%% of the largest"
          )
          assert(errors(2) < 2e-2, errors(3) < 2e-2)
        }
      }
    }
  }
}
