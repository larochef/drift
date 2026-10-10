package drift.runner

import utest.*

import drift.runner.ops.{HipOps, MatVecInputs}

/** The kernels run the official tiny HiDream O1 to its clean image: a prompt
  * alone, two references, and a LoRA with its whole-tensor differences.
  */
object TinyHiDreamO1GpuTests extends TestSuite {

  private def withOps[A](body: HipOps => A): A = {
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try body(ops)
    finally ops.close()
  }

  val tests = Tests {
    test("a prompt alone") {
      withOps { ops =>
        val (positions, error) = TinyHiDreamO1Case.textError(ops)
        println(f"  worst x̂ error: ${error * 100}%.4f%% of the largest")
        assert(positions, error < 1e-2)
      }
    }
    test("two references") {
      withOps { ops =>
        val (positions, error) = TinyHiDreamO1Case.referencesError(ops)
        println(f"  worst x̂ error: ${error * 100}%.4f%% of the largest")
        assert(positions, error < 1e-2)
      }
    }
    test("LoRAs of every naming") {
      withOps { ops =>
        Seq("ai_toolkit", "diffsynth", "comfy_extracted", "kohya").foreach {
          name =>
            val outcome = TinyHiDreamO1Case.lora(ops, name)
            println(
              f"  $name: error ${outcome.error * 100}%.4f%% of the largest (the update moves it by ${outcome.moved * 100}%.2f%%)"
            )
            assert(outcome.moved > 0.02, outcome.error < outcome.moved / 5)
        }
      }
    }
  }
}
