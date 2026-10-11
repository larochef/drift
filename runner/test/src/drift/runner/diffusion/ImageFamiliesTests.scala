package drift.runner.diffusion

import drift.runner.Fixtures
import drift.runner.ops.CpuOps
import drift.runner.weights.WeightSource

import utest.*

/** Which family a diffusion model's file is of (`ImagePipeline.Families`): told
  * from its weights, in an order that matters, and no family's by default.
  */
object ImageFamiliesTests extends TestSuite {

  private def familyOf(fixture: String): Option[String] = {
    val ops = new CpuOps
    val source =
      WeightSource.open(ops, Fixtures.path(s"tiny/$fixture/model.safetensors"))
    try ImagePipeline.Families.find(_.holds(source)).map(_.name)
    finally source.close()
  }

  val tests = Tests {
    test("each family's own model is recognised as its own") {
      val expected = Seq(
        "hidream_o1" -> "HiDream O1",
        "flux2" -> "FLUX.2",
        "flux2_dev" -> "FLUX.2",
        "qwen_image21" -> "Qwen Image 2.1",
        "pid" -> "PiD",
        "nucleus" -> "Nucleus-Image",
        "grn" -> "GRN",
        "llada_image" -> "LLaDA-Image",
        "krea2" -> "Krea 2"
      )
      val found = expected.map((fixture, _) => fixture -> familyOf(fixture))
      assert(found == expected.map((fixture, name) => fixture -> Some(name)))
    }
    test("Mage-Flow's blocks on other latent channels are not Mage-Flow") {
      // The tiny model's latents are narrower than the released one's, as the
      // first Qwen Image's are wider: the same blocks, and no family's — it
      // was Krea 2's by default once.
      assert(familyOf("mage_flow").isEmpty)
    }
    test("a file that is no diffusion model is no family's") {
      assert(familyOf("flux2_vae").isEmpty)
      assert(familyOf("mage_vae").isEmpty)
    }
    test("opening one is refused, with the families there are") {
      val problem =
        try {
          ImagePipeline.open(
            new CpuOps,
            Fixtures.path("tiny/flux2_vae/model.safetensors"),
            None,
            None,
            None,
            None
          )
          ""
        } catch { case error: IllegalArgumentException => error.getMessage }
      assert(problem.contains("none of the image models"))
      assert(problem.contains("Krea 2"), problem.contains("FLUX.2"))
    }
  }
}
