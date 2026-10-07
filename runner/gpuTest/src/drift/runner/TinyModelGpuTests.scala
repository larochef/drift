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
    test(
      "Mage-Flow: double-stream blocks with a reference and the BF16 timestep"
    ) {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyMageFlowCase.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("LLaDA-Image: text to image and editing through its transformer") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (plain, editing) = TinyLladaImageCase.velocityErrors(ops)
        println(
          f"  worst velocity error ${plain * 100}%.4f%%, editing ${editing * 100}%.4f%% of the largest"
        )
        assert(plain < 2e-2 && editing < 2e-2)
      } finally ops.close()
    }
    test(
      "LLaDA-Image's text path: QueryFormer then LLaDA2 then the projection"
    ) {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (queries, projected) = TinyLladaImageCase.connectorErrors(ops)
        val hidden = TinyLladaImageCase.backboneError(ops)
        println(
          f"  worst errors: queries ${queries * 100}%.4f%%, hidden state ${hidden * 100}%.4f%%, caption features ${projected * 100}%.4f%% of the largest"
        )
        assert(queries < 2e-2 && hidden < 2e-2 && projected < 2e-2)
      } finally ops.close()
    }
    test("LLaDA-Image's SigVQ: an image to its codes and semantic features") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (same, error) = TinyLladaImageCase.sigvqError(ops)
        println(
          f"  codes as the official ones: $same, worst semantic error ${error * 100}%.4f%% of the largest"
        )
        assert(same && error < 2e-2)
      } finally ops.close()
    }
    test("Nucleus-Image: text as keys and values and experts that choose") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyNucleusCase.velocityError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("The Mage-Flow VAE both ways (the official encoder and denoiser)") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (moments, decoded) = TinyMageFlowCase.vaeErrors(ops)
        println(
          f"  worst moments error ${moments * 100}%.4f%%, image error ${decoded * 100}%.4f%% of the largest"
        )
        assert(moments < 2e-2 && decoded < 2e-2)
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
    test("Wan with each expert's LoRA, then without, in BF16") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      ops.wideProducts = true
      try {
        val error = TinyWanCase.loraVelocityError(ops)
        println(
          f"  worst LoRA velocity error ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-2)
      } finally ops.close()
    }
    test("Wan's I2V conditioning with an end image") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val errors = TinyWanCase.conditionErrors(ops)
        println(
          s"  worst errors ${errors.map(e => f"${e * 100}%.4f%%").mkString(", ")} of the largest"
        )
        assert(errors.forall(_ < 4e-2))
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
    test("LTX 2.5's joint transformer, conditioned") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.conditionError(ops)
        println(f"  worst velocity error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("LTX 2.5's LoRAs") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.loraError(ops)
        println(f"  worst error ${error * 100}%.4f%% of the largest")
        assert(error < 1e-2)
      } finally ops.close()
    }
    test("LTX 2.5's conv VAE encoder") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyLtxCase.encoderError(ops)
        println(f"  worst latent error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2)
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
    test("MiniMax H3's audio decoder") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyAudioCase.minimaxH3Error(ops)
        println(f"  worst sample error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights and BF16 convolution inputs
      } finally ops.close()
    }
    test("LTX 2.5's audio decoder, vocoder and bandwidth extension") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val error = TinyAudioCase.ltxError(ops)
        println(f"  worst sample error ${error * 100}%.4f%% of the largest")
        assert(error < 4e-2) // BF16 weights and BF16 convolution inputs
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
    test("SeedVR2's transformer over windows of video and text") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (last, middle) = TinySeedVr2Case.velocityErrors(ops)
        println(
          f"  worst velocity error: ${last * 100}%.4f%% at 1000, ${middle * 100}%.4f%% at 637.5"
        )
        assert(last < 4e-2 && middle < 4e-2) // BF16 weights and F16 keys
      } finally ops.close()
    }
    test("SeedVR2's 7B transformer with its plain MLPs and angles") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (last, middle) =
          TinySeedVr2Case.velocityErrors(ops, "tiny/seedvr2_7b")
        println(
          f"  worst velocity error: ${last * 100}%.4f%% at 1000, ${middle * 100}%.4f%% at 637.5"
        )
        assert(last < 4e-2 && middle < 4e-2) // BF16 weights and F16 keys
      } finally ops.close()
    }
    test("SeedVR2's VAE both ways on one frame") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (latent, image) = TinySeedVr2VaeCase.errors(ops, "picture")
        println(
          f"  worst latent error: ${latent * 100}%.4f%%, image: ${image * 100}%.4f%% of the largest"
        )
        assert(
          latent < 4e-2 && image < 4e-2
        ) // BF16 weights and convolution inputs
      } finally ops.close()
    }
    test("SeedVR2's VAE both ways on 5 frames") {
      val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
      try {
        val (latent, image) = TinySeedVr2VaeCase.errors(ops, "clip")
        println(
          f"  worst latent error: ${latent * 100}%.4f%%, image: ${image * 100}%.4f%% of the largest"
        )
        assert(
          latent < 4e-2 && image < 4e-2
        ) // BF16 weights and convolution inputs
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
