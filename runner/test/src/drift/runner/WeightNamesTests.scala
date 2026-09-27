package drift.runner

import utest.*

import drift.runner.formats.FormatException
import drift.runner.models.WeightNames

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

/** The names the loaders read (`WeightNames`), on real files' names:
  * ERNIE-Image's diffusers-named FLUX.2 VAE must come to exactly the names of
  * BFL's own (`fixtures/names`, their headers' tensor names), and the single
  * files' prefixes must fall away.
  */
object WeightNamesTests extends TestSuite {

  private def names(file: String): Seq[String] =
    Files.readAllLines(Fixtures.path(s"names/$file")).asScala.toSeq

  val tests = Tests {
    test("diffusers' FLUX.2 VAE comes to BFL's names, one for one") {
      val diffusers = names("flux2-vae-diffusers.txt")
      val canonical = WeightNames.canonical(diffusers)
      assert(canonical.keySet == names("flux2-vae-original.txt").toSet)
      assert(
        canonical("decoder.up.3.block.0.conv1.weight") ==
          "decoder.up_blocks.0.resnets.0.conv1.weight"
      )
      assert(
        canonical("encoder.mid.attn_1.proj_out.weight") ==
          "encoder.mid_block.attentions.0.to_out.0.weight"
      )
      assert(canonical("encoder.quant_conv.weight") == "quant_conv.weight")
    }
    test("the original names are kept as they are") {
      val original = names("flux2-vae-original.txt")
      assert(WeightNames.canonical(original) == original.map(n => n -> n).toMap)
    }
    test("ComfyUI's prefixes fall away") {
      val canonical = WeightNames.canonical(
        Seq(
          "model.diffusion_model.txtfusion.projector.weight",
          "diffusion_model.blocks.0.attn.wq.weight",
          "net.s_embedder.proj.weight",
          "first_stage_model.decoder.conv_in.weight",
          "first.weight"
        )
      )
      assert(
        canonical.keySet == Set(
          "txtfusion.projector.weight",
          "blocks.0.attn.wq.weight",
          "s_embedder.proj.weight",
          "decoder.conv_in.weight",
          "first.weight"
        )
      )
    }
    test("a tensor stored twice under two prefixes is refused by name") {
      val error = intercept[FormatException](
        WeightNames.canonical(
          Seq("first.weight", "model.diffusion_model.first.weight")
        )
      )
      assert(error.getMessage.contains("first.weight"))
    }
  }
}
