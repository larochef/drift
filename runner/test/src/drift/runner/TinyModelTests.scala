package drift.runner

import utest.*

import drift.runner.diffusion.{VideoLora, WanPipeline}
import drift.runner.models.VaeTiles
import drift.runner.ops.CpuOps

import java.nio.file.Paths

/** The reference backend runs transformers' tiny models to their logits. The
  * only rounding is the F16 key-value cache (2⁻¹¹ relative). Qwen 3.5 MoE's
  * transformers forward uses the chunked delta rule, the runner the recurrent
  * one: the same maths, computed another way.
  */
object TinyModelTests extends TestSuite {

  private def check(
      model: TinyModelCase,
      prefill: Int,
      positions: Int = 20
  ): Unit = {
    val ops = new CpuOps
    try {
      val error = model.relativeError(model.run(ops, prefill), positions)
      println(f"  worst logit error: ${error * 100}%.4f%% of the largest")
      assert(error < 2e-3)
    } finally ops.close()
  }

  val tests = Tests {
    test("Qwen 3, all at once") { check(TinyModelCase.Qwen3, 20) }
    test("Qwen 3, prefill then decode") { check(TinyModelCase.Qwen3, 12) }
    test("Qwen 3.5 dense, all at once") { check(TinyModelCase.Qwen35, 20) }
    test("Qwen 3.5 dense, prefill then decode") {
      check(TinyModelCase.Qwen35, 12)
    }
    test("Qwen 3.5 MoE, all at once") { check(TinyModelCase.Qwen35Moe, 20) }
    test("Qwen 3.5 MoE, prefill then decode (the states carried)") {
      check(TinyModelCase.Qwen35Moe, 12)
    }
    test("Qwen 3.8 Flash Next, all at once (its dense positions)") {
      check(TinyModelCase.Qwen4Exp, 20, TinyModelCase.Qwen4ExpDensePositions)
    }
    test("Qwen 3.8 Flash Next, prefill then decode across an EOS") {
      check(TinyModelCase.Qwen4Exp, 4, TinyModelCase.Qwen4ExpDensePositions)
    }
    test(
      "Qwen 3.8 Flash Next's MTP head drafts llama.cpp's way (its own file)"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyModelCase.Qwen4Exp.draftError(ops)
        println(
          f"  worst draft logit error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("Qwen 3.5 MoE sees: patches, resize, positions, tower, logits") {
      val patches = TinyVisionCase.patchError
      println(f"  patches: worst ${patches}%.4f pixel levels")
      assert(patches < 1e-3)
      val ((height, width), resized) = TinyVisionCase.photoResize
      println(
        f"  70×100 resized to $height×$width: worst $resized%.2f pixel levels"
      )
      assert(height == 224, width == 320, resized <= 1.01)
      assert(TinyVisionCase.positionsMatch)
      val ops = new CpuOps
      try {
        val features = TinyVisionCase.featuresError(ops)
        println(
          f"  vision tokens: worst ${features * 100}%.4f%% of the largest"
        )
        assert(features < 2e-3)
        Seq(22, 9).foreach { prefill =>
          val error = TinyVisionCase.logitsError(ops, prefill)
          println(
            f"  prefill $prefill: worst logit error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 2e-3)
        }
      } finally ops.close()
    }
    test("Qwen3-VL encodes an image: tower, deepstack, last hidden state") {
      val ops = new CpuOps
      try {
        val (features, deepstack, hidden) = TinyVisionCase.qwen3vlErrors(ops)
        println(
          f"  tokens ${features * 100}%.4f%%, deepstack ${deepstack * 100}%.4f%%, hidden ${hidden * 100}%.4f%% of the largest"
        )
        assert(features < 2e-3, deepstack < 2e-3, hidden < 2e-3)
      } finally ops.close()
    }
    test("Qwen Image 2.1 edits: two references in the prefix") {
      val ops = new CpuOps
      try {
        val error = TinyQwenImage21Case.editError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("Krea 2's transformer: the text fused, then one velocity") {
      val ops = new CpuOps
      try {
        val error = TinyKrea2Case.velocityError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value cache of the main attention
      } finally ops.close()
    }
    test("MiniMax H3's packed t2va layout") {
      val error = TinyMiniMaxH3Case.layoutError()
      assert(error < 1e-5) // the fixture's positions are F32

    }
    test("MiniMax H3's transformer: the text refined, then one velocity") {
      val ops = new CpuOps
      try {
        val error = TinyMiniMaxH3Case.velocityError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value cache of the attention
      } finally ops.close()
    }
    test("MiniMax H3's video VAE: two chunks of four tiles") {
      val ops = new CpuOps
      try {
        val error = TinyMiniMaxH3Case.framesError(ops)
        println(f"  worst pixel error: $error%.5f")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("UMT5's encoder (Wan's text encoder)") {
      val ops = new CpuOps
      try {
        val error = TinyWanCase.umt5Error(ops)
        println(
          f"  worst hidden-state error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-4)
      } finally ops.close()
    }
    test(
      "Wan's I2V transformer: the text's keys and values, then one velocity"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyWanCase.velocityError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value caches
      } finally ops.close()
    }
    test(
      "Wan with each expert's LoRA (every naming, I64 alphas), then without"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyWanCase.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("Wan's LoRAs by expert: high-noise ones on the high-noise model") {
      val (high, low) = (
        VideoLora(Paths.get("high"), 1f, highNoise = true),
        VideoLora(Paths.get("low"), 1f, highNoise = false)
      )
      assert(
        WanPipeline.experts(Seq(high, low), twoExperts = true) ==
          (Seq(high), Seq(low))
      )
      assert(
        WanPipeline.experts(Seq(high, low), twoExperts = false) ==
          (Nil, Seq(high, low))
      )
    }
    test("Wan's I2V conditioning: first, first and last, last alone") {
      val ops = new CpuOps
      try {
        val errors = TinyWanCase.conditionErrors(ops)
        println(
          s"  worst errors: ${errors.map(e => f"${e * 100}%.4f%%").mkString(", ")} of the largest"
        )
        assert(errors.forall(_ < 4e-2)) // BF16 weights, as the VAE's test
      } finally ops.close()
    }
    test("The Wan 2.1 VAE on video: causal streams both ways") {
      val ops = new CpuOps
      try {
        val (encoded, decoded) = TinyWanCase.videoVaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, frame error ${decoded * 100}%.4f%% of the largest"
        )
        assert(encoded < 4e-2 && decoded < 4e-2) // BF16 weights, as for images
      } finally ops.close()
    }
    test("Gemma 4's text model (LTX 2.5's text encoder): every hidden state") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.gemmaError(ops)
        println(
          f"  worst hidden-state error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-4)
      } finally ops.close()
    }
    test("LTX 2.5's text features and connectors") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.connectorsError(ops)
        println(f"  worst connector error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value cache
      } finally ops.close()
    }
    test("LTX 2.5's joint audio and video transformer: one step") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.transformerError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value caches
      } finally ops.close()
    }
    test(
      "LTX 2.5's joint transformer: a conditioned step, keyframes appended"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.conditionError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value caches
      } finally ops.close()
    }
    test("LTX 2.5's LoRAs: transformer and connectors, both namings") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.loraError(ops)
        println(f"  worst error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("LTX 2.5's conv VAE encoder: causal, space-to-depth") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.encoderError(ops)
        println(f"  worst latent error: ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights
      } finally ops.close()
    }
    test("LTX 2.5's conv VAE decoder: residuals and depth-to-space") {
      val ops = new CpuOps
      try {
        val error = TinyLtxCase.vaeError(ops)
        println(f"  worst frame error: ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights
      } finally ops.close()
    }
    test("MiniMax H3's audio decoder: BigVGAN, one channel at a time") {
      val ops = new CpuOps
      try {
        val error = TinyAudioCase.minimaxH3Error(ops)
        println(f"  worst sample error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("LTX 2.5's audio decoder, vocoder and bandwidth extension") {
      val ops = new CpuOps
      try {
        val error = TinyAudioCase.ltxError(ops)
        println(f"  worst sample error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("The Wan 2.1 VAE's decoder on one image") {
      val ops = new CpuOps
      try {
        val error = TinyWanVaeCase.imageError(ops)
        println(f"  worst image error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 weights (2⁻⁸), then float sums
      } finally ops.close()
    }
    test("VaeTiles stitches tiles that disagree without a step (bug 42)") {
      // 2048 × 3072 in tiles of 1024 overlapping by 128 at least, as a
      // SeedVR2 ×2 of a 1024 × 1536 picture decodes — at a 16th of the size.
      // Every tile is one flat value of its own: whatever the tiles hold, the
      // stitched picture may only change by a ramp's slope between two
      // neighbouring pixels.
      val (height, width, tile, overlap, step) = (192, 128, 64, 8, 1)
      val (rowStarts, tileHeight, rowOverlaps) =
        VaeTiles.split(height, tile, overlap, step)
      val (columnStarts, tileWidth, columnOverlaps) =
        VaeTiles.split(width, tile, overlap, step)
      val tiles = rowStarts.indices.map(i =>
        columnStarts.indices.map(j =>
          Array.fill(tileHeight * tileWidth)(((i * 7 + j * 3) % 5).toFloat)
        )
      )
      val stitched = VaeTiles.stitch(
        tiles,
        1,
        tileHeight,
        tileWidth,
        rowOverlaps,
        columnOverlaps,
        height,
        width,
        1
      )
      // the values span 4, the narrowest ramp is the smallest overlap
      val slope = 4f / (rowOverlaps ++ columnOverlaps).min
      val steps = for {
        y <- 0 until height
        x <- 0 until width
        (dy, dx) <- Seq((1, 0), (0, 1))
        if y + dy < height && x + dx < width
      } yield math.abs(
        stitched((y + dy) * width + x + dx) - stitched(y * width + x)
      )
      assert(steps.max <= slope * 2 + 1e-4f)
      // and a tile is itself where no other reaches
      assert(stitched(0) == tiles(0)(0)(0))
    }
    test("VaeTiles cuts frames in tiles and stitches them back") {
      val (frames, height, width, channels) = (2, 48, 80, 3)
      val values =
        Array.tabulate(frames * height * width * channels)(i =>
          (i % 251) / 251f
        )
      // every tile as it is: the overlaps blend equal values
      val (same, count) = VaeTiles.mapped(
        values,
        frames,
        height,
        width,
        channels,
        32,
        8,
        8,
        1,
        1,
        channels
      )((tile, _, _) => (tile, frames))
      assert(count == frames)
      assert(same.indices.forall(i => math.abs(same(i) - values(i)) < 1e-6))
      // an "encoder" at 1/8: each tile's mean per 8 × 8 block, one frame out
      val (small, made) = VaeTiles.mapped(
        values,
        frames,
        height,
        width,
        channels,
        32,
        8,
        8,
        1,
        8,
        1
      ) { (tile, tileHeight, tileWidth) =>
        (
          Array.tabulate(tileHeight / 8 * (tileWidth / 8))(i =>
            tile(
              (i / (tileWidth / 8) * 8 * tileWidth + i % (tileWidth / 8) * 8) * channels
            )
          ),
          1
        )
      }
      assert(made == 1 && small.length == height / 8 * (width / 8))
      assert(small.indices.forall { i =>
        val (y, x) = (i / (width / 8) * 8, i % (width / 8) * 8)
        math.abs(small(i) - values((y * width + x) * channels)) < 1e-6
      })
    }
    test("SeedVR2's windows as the reference cuts them") {
      val mismatches = TinySeedVr2Case.windowMismatches
      assert(mismatches.isEmpty)
    }
    test("SeedVR2's transformer over windows of video and text") {
      val ops = new CpuOps
      try {
        val (last, middle) = TinySeedVr2Case.velocityErrors(ops)
        println(
          f"  worst velocity error: ${last * 100}%.4f%% at 1000, ${middle * 100}%.4f%% at 637.5"
        )
        assert(last < 2e-2 && middle < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("SeedVR2's 7B transformer with its plain MLPs and angles") {
      val ops = new CpuOps
      try {
        val (last, middle) =
          TinySeedVr2Case.velocityErrors(ops, "tiny/seedvr2_7b")
        println(
          f"  worst velocity error: ${last * 100}%.4f%% at 1000, ${middle * 100}%.4f%% at 637.5"
        )
        assert(last < 2e-2 && middle < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("SeedVR2's VAE both ways on one frame") {
      val ops = new CpuOps
      try {
        val (latent, image) = TinySeedVr2VaeCase.errors(ops, "picture")
        println(
          f"  worst latent error: ${latent * 100}%.4f%%, image: ${image * 100}%.4f%% of the largest"
        )
        assert(latent < 2e-2 && image < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("SeedVR2's VAE both ways on 5 frames") {
      val ops = new CpuOps
      try {
        val (latent, image) = TinySeedVr2VaeCase.errors(ops, "clip")
        println(
          f"  worst latent error: ${latent * 100}%.4f%%, image: ${image * 100}%.4f%% of the largest"
        )
        assert(latent < 2e-2 && image < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test(
      "Krea 2 with a LoRA (both namings, an alpha, two tables), then without"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyKrea2Case.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("PiD: patch and pixel blocks, latent injections, degrade σ") {
      val ops = new CpuOps
      try {
        val error = TinyPidCase.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("PiD's FLUX.1 and Qwen Image network: a 16-channel latent") {
      val ops = new CpuOps
      try {
        val error =
          TinyPidCase.velocityError(ops, TinyPidCase.Variant.Sixteen)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("The FLUX.1 VAE both ways (PiD's official AutoEncoder)") {
      val ops = new CpuOps
      try {
        val (encoded, decoded) = TinyPidCase.flux1VaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(encoded < 2e-2 && decoded < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("The Wan 2.1 VAE's encoder: PiD's Qwen Image latent") {
      val ops = new CpuOps
      try {
        val error = TinyPidCase.qwenImageLatentError(ops)
        println(f"  worst latent error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 weights
      } finally ops.close()
    }
    test("Gemma 2 as a text encoder: softcap, window, sandwich norms") {
      val ops = new CpuOps
      try {
        val (taps, normed) = TinyModelCase.gemma2EncodeErrors(ops)
        println(
          f"  worst hidden-state error ${taps * 100}%.4f%%, final ${normed * 100}%.4f%% of the largest"
        )
        assert(taps < 2e-3 && normed < 2e-3)
      } finally ops.close()
    }
    test("FLUX.2's transformer, with a reference image") {
      val ops = new CpuOps
      try {
        val error = TinyFlux2Case.velocityError(ops, "flux2")
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test(
      "FLUX.2 with a LoRA (both namings, fused targets split), then without"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyFlux2Case.loraVelocityError(ops, "flux2")
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("FLUX.2 [dev]'s transformer: the guidance embedding") {
      val ops = new CpuOps
      try {
        val error = TinyFlux2Case.velocityError(ops, "flux2_dev")
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("FLUX.2 [dev] with a LoRA (kohya's names, the guidance MLP)") {
      val ops = new CpuOps
      try {
        val error = TinyFlux2Case.loraVelocityError(ops, "flux2_dev")
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("Mistral as a text encoder, from safetensors and from a GGUF") {
      Seq("model.safetensors", "model.gguf").foreach { file =>
        val ops = new CpuOps
        try {
          val error = TinyModelCase.mistralEncodeError(ops, file)
          println(
            f"  $file: worst hidden-state error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 2e-3)
        } finally ops.close()
      }
    }
    test("The FLUX.2 VAE both ways") {
      Seq("model.safetensors", "diffusers.safetensors").foreach { file =>
        val ops = new CpuOps
        try {
          val (encoded, decoded) = TinyFlux2Case.vaeErrors(ops, file)
          println(
            f"  $file: worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
          )
          assert(
            encoded < 2e-2 && decoded < 2e-2
          ) // BF16 weights (2⁻⁸), then float sums
        } finally ops.close()
      }
    }
    test("Qwen Image 2.1's transformer: the text a prefix, then one velocity") {
      val ops = new CpuOps
      try {
        val error = TinyQwenImage21Case.velocityError(ops)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3) // the F16 key-value cache
      } finally ops.close()
    }
    test(
      "Qwen Image 2.1 with a LoRA (both namings, gate_up split), then without"
    ) {
      val ops = new CpuOps
      try {
        val error = TinyQwenImage21Case.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("The Qwen Image 2.1 VAE both ways") {
      val ops = new CpuOps
      try {
        val (encoded, decoded) = TinyQwenImage21Case.vaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(
          encoded < 2e-2 && decoded < 2e-2
        ) // BF16 weights (2⁻⁸), then float sums
      } finally ops.close()
    }
    test("Qwen 3 as a text encoder: the residual stream after 0 and 1 layers") {
      val ops = new CpuOps
      try
        Seq(false, true).foreach { padded =>
          val error = TinyModelCase.encodeError(ops, padded)
          println(
            f"  padded $padded: worst hidden-state error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 2e-3) // the F16 key-value cache, as for the logits
        }
      finally ops.close()
    }
    test("Qwen 3.5's MTP layer drafts vLLM's way (dense, and experts fused)") {
      val ops = new CpuOps
      try {
        val error = Seq(TinyModelCase.Qwen35, TinyModelCase.Qwen35Moe)
          .map(_.draftError(ops))
          .max
        println(
          f"  worst draft logit error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
  }
}
