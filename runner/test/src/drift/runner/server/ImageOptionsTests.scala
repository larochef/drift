package drift.runner.server

import utest.*

/** sd-server's flags as drift passes them for Krea 2 and FLUX.2 [dev]. */
object ImageOptionsTests extends TestSuite {

  private val Krea2 = Seq(
    "--diffusion-model",
    "/m/krea.safetensors",
    "--vae",
    "/m/wan.safetensors",
    "--llm",
    "/m/qwen.gguf",
    "--diffusion-fa",
    "--cfg-scale",
    "1.0",
    "--steps",
    "4",
    "-H",
    "1024",
    "-W",
    "1024",
    "--cache-mode",
    "spectrum",
    "--mmap",
    "--lora-model-dir",
    "/l",
    "--hires-upscalers-dir",
    "/u",
    "--listen-ip",
    "127.0.0.1",
    "--listen-port",
    "8123"
  )

  val tests = Tests {
    test("drift's Krea 2 launch") {
      val Right(options) = ImageOptions.parse(Krea2): @unchecked
      assert(options.loraDirectory.map(_.toString).contains("/l"))
      assert(
        options.diffusionModel.toString == "/m/krea.safetensors",
        options.llm.toString == "/m/qwen.gguf"
      )
      assert(
        options.port == 8123,
        options.steps == 4,
        options.cfgScale == 1.0,
        options.width == 1024
      )
      assert(
        options.notes.exists(_.startsWith("--cache-mode spectrum accepted"))
      )
    }
    test("the distilled guidance is read, absent unless given") {
      val Right(plain) = ImageOptions.parse(Krea2): @unchecked
      assert(plain.guidance.isEmpty)
      val Right(dev) =
        ImageOptions.parse(Krea2 ++ Seq("--guidance", "4.0")): @unchecked
      assert(dev.guidance.contains(4.0))
    }
    test(
      "what the runner cannot do yet is refused by name, and unknown flags too"
    ) {
      assert(
        ImageOptions
          .parse(Krea2 :+ "--vae-tiling")
          .left
          .exists(_.contains("--vae-tiling"))
      )
      assert(
        ImageOptions.parse(Krea2 ++ Seq("--frobnicate", "1")) == Left(
          "unknown flag --frobnicate"
        )
      )
      assert(
        ImageOptions
          .parse(Krea2.drop(2))
          .left
          .exists(_.contains("--diffusion-model"))
      )
    }
  }
}
