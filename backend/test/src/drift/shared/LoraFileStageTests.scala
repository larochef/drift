package drift.shared

import utest.*

/** A LoRA file's stage from the names it goes by: "high noise" anywhere first,
  * then the short forms as words on their own.
  */
object LoraFileStageTests extends TestSuite {
  val tests = Tests {
    import LoraFileStage.*
    test("the long forms win wherever they are") {
      assert(of("wan22_high_noise.safetensors", "low") == HighNoise)
      assert(of("lora.safetensors", "Low Noise v1") == LowNoise)
    }
    test("a short form is a word on its own") {
      assert(of("wan22_hn_rank64.gguf") == HighNoise)
      assert(of("model_LOW.safetensors") == LowNoise)
      assert(of("high/adapter.safetensors") == HighNoise)
      assert(of("shallow_focus.safetensors") == General)
      assert(of("highres_fix.safetensors") == General)
      assert(of("moss_image_sft_4step_v2.0.safetensors", "kromoss_8_step") ==
        General)
    }
  }
}
