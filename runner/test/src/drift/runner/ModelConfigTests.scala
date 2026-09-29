package drift.runner

import utest.*

import drift.runner.formats.{FormatException, ModelConfig}

object ModelConfigTests extends TestSuite {

  /** An excerpt of Qwen 3.8 Flash Next's config.json. */
  private val config = new ModelConfig(
    ujson.read("""{
      "architectures": ["Qwen4ExpForConditionalGeneration"],
      "text_config": {
        "hidden_size": 2560,
        "layer_types": ["linear_attention", "full_attention"],
        "mtp": {"num_hidden_layers": 1, "rope_theta": 10000000},
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"mrope_section": [11, 11, 10]},
        "pad_token_id": null
      }
    }"""),
    "config.json"
  )

  val tests = Tests {
    test("the language model's section") {
      val text = config.text
      assert(text.int("hidden_size") == 2560)
      assert(
        text.strings("layer_types") == Seq("linear_attention", "full_attention")
      )
      assert(text.double("partial_rotary_factor") == 0.25)
      assert(text.section("mtp").long("rope_theta") == 10000000L)
      assert(
        text.section("rope_parameters").ints("mrope_section") == Seq(11, 11, 10)
      )
    }
    test("a null counts as absent") {
      assert(!config.text.has("pad_token_id"))
      val error = assertThrows[FormatException](config.text.int("pad_token_id"))
      assert(error.getMessage == "config.json/text_config has no pad_token_id")
    }
    test("a fraction is not an integer") {
      val error =
        assertThrows[FormatException](config.text.int("partial_rotary_factor"))
      assert(error.getMessage.contains("not an integer"))
    }
  }
}
