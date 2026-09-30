package drift.runner

import utest.*

import drift.runner.ops.{HipOps, MatVecInputs}

/** The kernels run transformers' tiny models to their logits: 20 tokens take
  * the prefill GEMM (F16 operands) where a GEMM applies, single tokens the
  * matrix-vector path.
  */
object TinyModelGpuTests extends TestSuite {

  private def check(
      model: TinyModelCase,
      prefill: Int,
      inputs: MatVecInputs,
      positions: Int = 20
  ): Unit = {
    val ops = new HipOps(Gpu.hip, inputs)
    try {
      val error = model.relativeError(model.run(ops, prefill), positions)
      println(
        f"  $inputs, prefill $prefill: worst logit error ${error * 100}%.4f%% of the largest"
      )
      assert(error < 1e-2)
    } finally ops.close()
  }

  val tests = Tests {
    test("Qwen 3") {
      check(TinyModelCase.Qwen3, 20, MatVecInputs.Float)
      MatVecInputs.values.foreach(check(TinyModelCase.Qwen3, 12, _))
    }
    test("Qwen 3.5 dense") {
      check(TinyModelCase.Qwen35, 20, MatVecInputs.Float)
      MatVecInputs.values.foreach(check(TinyModelCase.Qwen35, 12, _))
    }
    test("Qwen 3.5 MoE") {
      check(TinyModelCase.Qwen35Moe, 20, MatVecInputs.Float)
      MatVecInputs.values.foreach(check(TinyModelCase.Qwen35Moe, 12, _))
    }
    test("Qwen 3.5 MoE sees: its tower and logits with an image") {
      MatVecInputs.values.foreach { inputs =>
        val ops = new HipOps(Gpu.hip, inputs)
        try {
          val features = TinyVisionCase.featuresError(ops)
          println(
            f"  $inputs: vision tokens worst ${features * 100}%.4f%% of the largest"
          )
          assert(features < 1e-2)
          Seq(22, 9).foreach { prefill =>
            val error = TinyVisionCase.logitsError(ops, prefill)
            println(
              f"  $inputs, prefill $prefill: worst logit error ${error * 100}%.4f%% of the largest"
            )
            assert(error < 1e-2)
          }
        } finally ops.close()
      }
    }
    test("Qwen3-VL encodes an image; Qwen Image 2.1 edits") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (features, deepstack, hidden) = TinyVisionCase.qwen3vlErrors(ops)
        val edit = TinyQwenImage21Case.editError(ops)
        println(
          f"  tokens ${features * 100}%.4f%%, deepstack ${deepstack * 100}%.4f%%, hidden ${hidden * 100}%.4f%%, edit velocity ${edit * 100}%.4f%% of the largest"
        )
        assert(features < 1e-2, deepstack < 1e-2, hidden < 1e-2, edit < 1e-2)
      } finally ops.close()
    }
    test("Qwen 3.8 Flash Next (its dense positions)") {
      val dense = TinyModelCase.Qwen4ExpDensePositions
      MatVecInputs.values.foreach(check(TinyModelCase.Qwen4Exp, 4, _, dense))
      check(TinyModelCase.Qwen4Exp, 20, MatVecInputs.Float, dense)
    }
    test("Qwen 3 as a text encoder, whole and padded") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try
        Seq(false, true).foreach { padded =>
          val error = TinyModelCase.encodeError(ops, padded)
          println(
            f"  padded $padded: worst hidden-state error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 1e-2)
        }
      finally ops.close()
    }
    test("PiD: patch and pixel blocks, latent injections, degrade σ") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyPidCase.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("PiD's FLUX.1 and Qwen Image network: a 16-channel latent") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error =
          TinyPidCase.velocityError(ops, TinyPidCase.Variant.Sixteen)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("The FLUX.1 VAE both ways (PiD's official AutoEncoder)") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (encoded, decoded) = TinyPidCase.flux1VaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(encoded < 4e-2 && decoded < 4e-2) // BF16 weights
      } finally ops.close()
    }
    test("The Wan 2.1 VAE's encoder: PiD's Qwen Image latent") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyPidCase.qwenImageLatentError(ops)
        println(f"  worst latent error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights
      } finally ops.close()
    }
    test("Gemma 2 as a text encoder: softcap, window, sandwich norms") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (taps, normed) = TinyModelCase.gemma2EncodeErrors(ops)
        println(
          f"  worst hidden-state error ${taps * 100}%.4f%%, final ${normed * 100}%.4f%% of the largest"
        )
        assert(taps < 1e-2 && normed < 1e-2)
      } finally ops.close()
    }
    test("FLUX.2's transformer, with a reference image") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyFlux2Case.velocityError(ops, "flux2")
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test(
      "FLUX.2 with a LoRA (both namings, fused targets split), then without"
    ) {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyFlux2Case.loraVelocityError(ops, "flux2")
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("FLUX.2 [dev]'s transformer: the guidance embedding") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyFlux2Case.velocityError(ops, "flux2_dev")
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("FLUX.2 [dev] with a LoRA (kohya's names, the guidance MLP)") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyFlux2Case.loraVelocityError(ops, "flux2_dev")
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("Mistral as a text encoder, from safetensors and from a GGUF") {
      Seq("model.safetensors", "model.gguf").foreach { file =>
        val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
        try {
          val error = TinyModelCase.mistralEncodeError(ops, file)
          println(
            f"  $file: worst hidden-state error ${error * 100}%.4f%% of the largest"
          )
          assert(error < 1e-2)
        } finally ops.close()
      }
    }
    test("The FLUX.2 VAE both ways") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (encoded, decoded) =
          TinyFlux2Case.vaeErrors(ops, "model.safetensors")
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(
          encoded < 4e-2 && decoded < 4e-2
        ) // BF16 weights and convolution inputs
      } finally ops.close()
    }
    test("Krea 2's transformer") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyKrea2Case.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("MiniMax H3's transformer, its products in BF16") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      ops.wideProducts = true
      try {
        val error = TinyMiniMaxH3Case.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2) // BF16 products
      } finally ops.close()
    }
    test("MiniMax H3's video VAE") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyMiniMaxH3Case.framesError(ops)
        println(f"  worst pixel error $error%.5f")
        assert(error < 1e-2) // F16 products, as the released recipe
      } finally ops.close()
    }
    test("UMT5's encoder") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyWanCase.umt5Error(ops)
        println(
          f"  worst hidden-state error ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("Wan's I2V transformer, its products in BF16") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      ops.wideProducts = true
      try {
        val error = TinyWanCase.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("The Wan 2.1 VAE on video") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (encoded, decoded) = TinyWanCase.videoVaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, frame error ${decoded * 100}%.4f%% of the largest"
        )
        assert(encoded < 4e-2 && decoded < 4e-2)
      } finally ops.close()
    }
    test("Gemma 4's text model (LTX 2.5)") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.gemmaError(ops)
        println(
          f"  worst hidden-state error ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("LTX 2.5's text features and connectors") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.connectorsError(ops)
        println(f"  worst connector error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("LTX 2.5's joint transformer") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.transformerError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("LTX 2.5's conv VAE decoder") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.vaeError(ops)
        println(f"  worst frame error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2)
      } finally ops.close()
    }
    test("The Wan 2.1 VAE's decoder") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyWanVaeCase.imageError(ops)
        println(f"  worst image error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights and BF16 convolution inputs
      } finally ops.close()
    }
    test(
      "Krea 2 with a LoRA (both namings, an alpha, two tables), then without"
    ) {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyKrea2Case.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("Qwen Image 2.1's transformer: the text a prefix, then one velocity") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyQwenImage21Case.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test(
      "Qwen Image 2.1 with a LoRA (both namings, gate_up split), then without"
    ) {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyQwenImage21Case.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("The Qwen Image 2.1 VAE both ways") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (encoded, decoded) = TinyQwenImage21Case.vaeErrors(ops)
        println(
          f"  worst latent error ${encoded * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(
          encoded < 4e-2 && decoded < 4e-2
        ) // BF16 weights and convolution inputs
      } finally ops.close()
    }
    test("drafts change no token: the states kept per token, rolled back") {
      GeneratorTests.draftsChangeNothing(
        new HipOps(Gpu.hip, MatVecInputs.Float)
      )
      GeneratorTests.multiTokenChangesNothing(
        new HipOps(Gpu.hip, MatVecInputs.Float)
      )
    }
    test("a verification's logits are its tokens' decode steps', bit for bit") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try
        Seq(
          TinyModelCase.Qwen35,
          TinyModelCase.Qwen35Moe,
          TinyModelCase.Qwen4Exp
        )
          .foreach { model =>
            val gap = model.verifyGap(ops, 12, 4)
            println(f"  largest gap $gap%.3g")
            assert(gap == 0.0)
          }
      finally ops.close()
    }
    test("the MTP heads of Qwen 3.5 (dense and MoE) and Qwen 3.8 Flash Next") {
      Seq(TinyModelCase.Qwen35, TinyModelCase.Qwen35Moe, TinyModelCase.Qwen4Exp)
        .foreach(draftsLike)
    }
  }

  private def draftsLike(model: TinyModelCase): Unit = {
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      val error = model.draftError(ops)
      println(
        f"  worst draft logit error ${error * 100}%.4f%% of the largest"
      )
      assert(error < 1e-2)
    } finally ops.close()
  }
}
