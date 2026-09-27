package drift.runner

import utest.*

import drift.runner.ops.CpuOps

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
    test("The Wan 2.1 VAE's decoder on one image") {
      val ops = new CpuOps
      try {
        val error = TinyWanVaeCase.imageError(ops)
        println(f"  worst image error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 weights (2⁻⁸), then float sums
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
