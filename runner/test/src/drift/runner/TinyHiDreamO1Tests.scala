package drift.runner

import utest.*

import drift.runner.ops.CpuOps

/** The reference backend runs the official tiny HiDream O1 to its clean image:
  * a prompt alone, two references, and a LoRA of each naming the published
  * files use — every one applied whole, and said so in the log.
  */
object TinyHiDreamO1Tests extends TestSuite {

  private def withOps[A](body: CpuOps => A): A = {
    val ops = new CpuOps
    try body(ops)
    finally ops.close()
  }

  /** A file's x̂ follows the official model with the file merged, and the file
    * moves it: an update left out would leave the error at the update's size.
    */
  private def applied(outcome: TinyHiDreamO1Case.LoraOutcome): Unit = {
    println(
      f"  error ${outcome.error * 100}%.4f%% of the largest (the update moves it by ${outcome.moved * 100}%.2f%%)"
    )
    outcome.report.foreach(line => println(s"  $line"))
    assert(outcome.moved > 0.02, outcome.error < outcome.moved / 10)
  }

  val tests = Tests {
    test("a prompt alone: positions and x̂ as the official model") {
      withOps { ops =>
        val (positions, error) = TinyHiDreamO1Case.textError(ops)
        println(f"  worst x̂ error: ${error * 100}%.4f%% of the largest")
        assert(positions, error < 2e-3)
      }
    }
    test("two references: seen by the tower and read as patches") {
      withOps { ops =>
        val (positions, error) = TinyHiDreamO1Case.referencesError(ops)
        println(f"  worst x̂ error: ${error * 100}%.4f%% of the largest")
        assert(positions, error < 2e-3)
      }
    }
    test("an ai-toolkit LoRA (Civitai's trainer): lora_A and lora_B") {
      withOps { ops =>
        val outcome = TinyHiDreamO1Case.lora(ops, "ai_toolkit")
        applied(outcome)
        assert(
          outcome.use.unmatched.isEmpty,
          outcome.use.unread.isEmpty,
          outcome.use.outside == 0,
          outcome.report == Seq("LoRA: 26 of 26 updates applied (1 file)")
        )
      }
    }
    test("a DiffSynth-Studio LoRA: whole names and the vision tower's") {
      withOps { ops =>
        val outcome = TinyHiDreamO1Case.lora(ops, "diffsynth")
        applied(outcome)
        // the tower's update is counted, and is no error
        assert(
          outcome.use.unmatched.isEmpty,
          outcome.use.unread.isEmpty,
          outcome.use.outside == 1,
          outcome.report == Seq("LoRA: 21 of 22 updates applied (1 file)")
        )
      }
    }
    test("a ComfyUI extraction: pairs and whole-tensor differences") {
      withOps { ops =>
        val outcome = TinyHiDreamO1Case.lora(ops, "comfy_extracted")
        applied(outcome)
        // 12 layer norms, the final norm and 4 biases; the embeddings' pair
        // and the tower's norm left to the checkpoint
        assert(
          outcome.use.updates == 45,
          outcome.use.applied == 43,
          outcome.use.outside == 2,
          outcome.use.unmatched.isEmpty,
          outcome.use.unread.isEmpty
        )
      }
    }
    test("a kohya LoRA: lora_unet_ names and an alpha") {
      withOps { ops =>
        val outcome = TinyHiDreamO1Case.lora(ops, "kohya")
        applied(outcome)
        assert(outcome.use.unmatched.isEmpty, outcome.use.unread.isEmpty)
      }
    }
  }
}
