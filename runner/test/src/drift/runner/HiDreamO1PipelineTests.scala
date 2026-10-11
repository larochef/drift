package drift.runner

import utest.*

import drift.runner.diffusion.HiDreamO1Pipeline
import drift.runner.models.HiDreamO1

/** HiDream O1's host-side pieces against the official `models/pipeline.py`: the
  * distilled schedule and the `(C p1 p2)` patch layout.
  */
object HiDreamO1PipelineTests extends TestSuite {
  val tests = Tests {
    test("28 steps are the official DEFAULT_TIMESTEPS, then 0") {
      val sigmas = HiDreamO1Pipeline.distilledSigmas(28)
      assert(
        sigmas.size == 29,
        sigmas.head == 0.999f,
        sigmas(27) == 0.008f,
        sigmas.last == 0f
      )
    }
    test("other step counts follow the same curve, first and last kept") {
      val sigmas = HiDreamO1Pipeline.distilledSigmas(10)
      assert(
        sigmas.size == 11,
        sigmas.head == 0.999f,
        math.abs(sigmas(9) - 0.008f) < 1e-6,
        sigmas.sliding(2).forall(pair => pair(0) > pair(1))
      )
    }
    test("a patch holds its pixels channel-major, and back") {
      // 64 × 32: two patches side by side
      val (width, height) = (64, 32)
      val pixels = Array.tabulate(width * height * 3)(_.toFloat)
      val patches = HiDreamO1Pipeline.patches(pixels, width, height)
      // patch 1, channel 2, row 3, column 5 is pixel (3, 32 + 5), channel 2
      val at = 1 * 3072 + 2 * 1024 + 3 * 32 + 5
      assert(
        patches(at) == pixels((3 * width + 37) * 3 + 2),
        HiDreamO1Pipeline.pixels(patches, width, height).sameElements(pixels)
      )
    }
    test("a step's preview is the patches' picture, 8 pixels to one") {
      // 64 × 32, red only: the left patch at −1, the right one rising by row
      val (width, height) = (64, 32)
      val pixels = new Array[Float](width * height * 3)
      for {
        y <- 0 until height
        x <- 0 until width
      } pixels((y * width + x) * 3) = if (x < 32) -1f else y / 16f - 1f
      val picture = HiDreamO1Pipeline.reduced(
        HiDreamO1Pipeline.patches(pixels, width, height),
        width,
        height
      )
      def red(x: Int, y: Int) = (picture.getRGB(x, y) >> 16) & 0xff
      assert(
        picture.getWidth == 8,
        picture.getHeight == 4,
        red(3, 2) == 0,
        // rows 8 to 15 average 11.5 / 16 − 1
        red(4, 1) == math.round(11.5f / 32 * 255),
        (picture.getRGB(4, 1) & 0xffff) == 0x8080
      )
    }
    test("the sampler a request leaves open is inference.py's") {
      import HiDreamO1Pipeline.*
      assert(
        ownSampler(guided = true, references = 0) == UniPcSampler,
        ownSampler(guided = true, references = 1) == UniPcSampler,
        ownSampler(guided = false, references = 0) == Flash,
        // the Dev checkpoint edits one picture with plain Euler
        ownSampler(guided = false, references = 1) == Euler,
        ownSampler(guided = false, references = 3) == Flash
      )
    }
    test("a reference is scaled and cut as resize_pilimage does") {
      // (width, height, side) → the size scaled to, the size cut to
      val official = Seq(
        (1024, 768, 2048) -> ((2347, 1760), (2336, 1760)),
        (3000, 2000, 2048) -> ((2496, 1664), (2496, 1664)),
        (1000, 1500, 1536) -> ((1248, 1872), (1248, 1856)),
        (2048, 2048, 2048) -> ((2048, 2048), (2048, 2048)),
        (1777, 999, 1024) -> ((1366, 768), (1344, 768)),
        (640, 480, 512) -> ((576, 432), (576, 416))
      )
      official.foreach { case ((width, height, side), sizes) =>
        val made = HiDreamO1Pipeline.referenceSize(width, height, side)
        assert(made == sizes)
      }
    }
    test("a reference twice the side or more is halved first") {
      // 5000 × 4200 for a side of 1024: halved twice, then 1088 × 928
      val pixels = new Array[Float](500 * 420 * 3)
      val made = HiDreamO1Pipeline.referencePixels(pixels, 500, 420, 96)
      // 500 × 420 → 250 × 210 (210 ≥ 192) → 125 × 105, then 96² of area
      assert(
        (made.width, made.height) == (96, 64),
        made.pixels.length == 96 * 64 * 3,
        HiDreamO1Pipeline.referenceSize(1250, 1050, 1024) ==
          ((1105, 928), (1088, 928))
      )
    }
    test("fewer pixels for each of more references") {
      import HiDreamO1Pipeline.*
      assert(
        Seq(1, 2, 3, 4, 5, 8, 9).map(referenceSide(_, 2048, 1536)) ==
          Seq(2048, 1536, 1024, 1024, 768, 768, 512),
        Seq(1, 4, 5, 8, 9).map(conditionSide) == Seq(384, 384, 288, 288, 192),
        // calculate_dimensions, for the sizes above
        conditionSize(384, 2336, 1760) == (416, 320),
        conditionSize(384, 1248, 1856) == (288, 448),
        conditionSize(288, 2496, 1664) == (352, 224)
      )
    }
    test("the patches sit where get_rope_index_fix_point puts them") {
      // a 2 × 3 target, then references of 2 × 3 and 2 × 2
      val positions =
        HiDreamO1.patchPositions(Seq((2, 3), (2, 3), (2, 2))).grouped(16).toSeq
      assert(
        positions(0).toSeq ==
          Seq.fill(6)(4096) ++ Seq.fill(6)(4099) ++ Seq.fill(4)(4102),
        positions(1).toSeq ==
          Seq(4096, 4096, 4096, 4097, 4097, 4097) ++
          Seq(4099, 4099, 4099, 4100, 4100, 4100) ++
          Seq(4102, 4102, 4103, 4103),
        positions(2).toSeq ==
          Seq(4096, 4097, 4098, 4096, 4097, 4098) ++
          Seq(4099, 4100, 4101, 4099, 4100, 4101) ++
          Seq(4102, 4103, 4102, 4103)
      )
    }
    test("a LoRA's target under every naming is the checkpoint's name") {
      val sites = Set(
        "model.language_model.layers.0.self_attn.q_proj",
        "model.t_embedder1.mlp.0",
        "model.x_embedder.proj2"
      )
      def site(target: String) = HiDreamO1.site(target, sites)
      assert(
        // ai-toolkit and ComfyUI, the prefix gone at load
        site("language_model.layers.0.self_attn.q_proj") == sites.head,
        // DiffSynth-Studio
        site("model.language_model.layers.0.self_attn.q_proj") == sites.head,
        // PEFT
        site("base_model.model.model.x_embedder.proj2") ==
          "model.x_embedder.proj2",
        // kohya
        site("lora_unet_language_model_layers_0_self_attn_q_proj") ==
          sites.head,
        site("lora_unet_t_embedder1_mlp_0") == "model.t_embedder1.mlp.0",
        HiDreamO1.outside(site("model.visual.blocks.3.attn.qkv")),
        HiDreamO1.outside(site("language_model.embed_tokens")),
        HiDreamO1.outside(site("lora_unet_visual_blocks_3_attn_qkv")),
        !HiDreamO1.outside(site("language_model.layers.0.self_attn.q_proj"))
      )
    }
    test("an edit's reference of the target's shape is read at its size") {
      // 1024 × 1536 edited at 1664 × 2496: the same shape, the target's size
      val side = HiDreamO1Pipeline.editedSide(1024, 1536, 1664, 2496)
      assert(
        HiDreamO1Pipeline.referenceSize(1024, 1536, side)._2 == (1664, 2496)
      )
      // another shape keeps the official `max_size`
      assert(HiDreamO1Pipeline.editedSide(1024, 1024, 1664, 2496) == 2496)
    }
    test("a LoRA trained on the full checkpoint is known by its metadata") {
      def trained(on: String) =
        HiDreamO1Pipeline.trainedOnFull(Map("ss_sd_model_name" -> on))
      assert(
        // ai-toolkit's, as Civitai's files carry it
        trained(
          "urn:air:hidream-o1:repository:huggingface:HiDream-ai/HiDream-O1-Image@main.tar"
        ),
        !trained("HiDream-ai/HiDream-O1-Image-Dev"),
        !trained("black-forest-labs/FLUX.2-klein-9B"),
        !HiDreamO1Pipeline.trainedOnFull(Map.empty)
      )
      val full = Map("ss_sd_model_name" -> "HiDream-ai/HiDream-O1-Image")
      assert(
        HiDreamO1Pipeline.fullOnly(Seq(full, full)),
        // a file that says nothing may be the Dev distillation
        !HiDreamO1Pipeline.fullOnly(Seq(full, Map.empty)),
        !HiDreamO1Pipeline.fullOnly(Nil)
      )
    }
  }
}
