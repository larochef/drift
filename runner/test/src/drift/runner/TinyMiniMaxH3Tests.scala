package drift.runner

import utest.*

import drift.runner.models.MiniMaxH3Config
import drift.runner.ops.CpuOps

/** MiniMax H3 beyond text to video on the reference backend: LoRAs, the seeded
  * noise, the encoders, keyframes and guides.
  */
object TinyMiniMaxH3Tests extends TestSuite {

  val tests = Tests {
    test("MiniMax H3's LoRA names: every published naming placed") {
      val placements = Seq(
        "blocks.3.attn.qkv_proj" -> Seq(
          "blocks.3.attn.q",
          "blocks.3.attn.k",
          "blocks.3.attn.v"
        ),
        "blocks.3.mlp.fc1" -> Seq("blocks.3.mlp.gate", "blocks.3.mlp.value"),
        "blocks.49.adaln_proj.linear" -> Seq("blocks.49.adaln_proj.linear"),
        "token_refiner.blocks.1.mlp.fc2" -> Seq(
          "token_refiner.blocks.1.mlp.fc2"
        ),
        "lora_unet_blocks_24_attn_out_proj" -> Seq("blocks.24.attn.out_proj"),
        "lora_unet_token_refiner_blocks_0_attn_qkv_proj" ->
          Seq(
            "token_refiner.blocks.0.attn.q",
            "token_refiner.blocks.0.attn.k",
            "token_refiner.blocks.0.attn.v"
          ),
        "lora_unet_final_layer_video_out" -> Seq("final_layer.video_out"),
        "transformer_blocks.7.ff.net.0.proj" -> Seq(
          "blocks.7.mlp.value",
          "blocks.7.mlp.gate"
        ),
        "transformer_blocks.7.attn.to_out.0" -> Seq("blocks.7.attn.out_proj"),
        "token_refiner.refiner_blocks.1.attn.to_v" -> Seq(
          "token_refiner.blocks.1.attn.v"
        ),
        "norm_out.linear" -> Seq("final_layer.adaln_proj.linear"),
        "context_embedder" -> Seq("condition_proj"),
        "time_embedder.proj_in" -> Seq("time_embedder.proj_in"),
        "blocks.3.norm1" -> Nil,
        "lora_unet_something_else" -> Nil
      )
      placements.foreach((target, sites) =>
        assert(MiniMaxH3Config.placement(target) == sites)
      )
    }
    test("MiniMax H3's LoRAs at run time, full file") {
      val ops = new CpuOps
      try {
        val (error, unapplied) =
          TinyMiniMaxH3Case.loraError(ops, pruned = false)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3, unapplied.isEmpty)
      } finally ops.close()
    }
    test("MiniMax H3's LoRAs at run time, pruned file") {
      val ops = new CpuOps
      try {
        val (error, unapplied) = TinyMiniMaxH3Case.loraError(ops, pruned = true)
        println(f"  worst velocity error: ${error * 100}%.4f%% of the largest")
        assert(error < 2e-3, unapplied == Seq("time_embedder.proj_in"))
      } finally ops.close()
    }
    test("MiniMax H3's video encoder: tiles, the keyframe draw, a clip") {
      val ops = new CpuOps
      try {
        val errors = TinyMiniMaxH3Case.videoEncoderErrors(ops)
        println(
          errors
            .map(e => f"${e * 100}%.4f%%")
            .mkString("  errors: ", ", ", " of the largest")
        )
        assert(errors.forall(_ < 1e-3)) // F16 weights
      } finally ops.close()
    }
    test("MiniMax H3's fl2va: layout, draws, one conditioned step") {
      val ops = new CpuOps
      try {
        val Seq(
          positions,
          keyframes,
          video,
          audio,
          videoVelocity,
          audioVelocity
        ) =
          TinyMiniMaxH3Case.fl2vaErrors(ops)
        println(
          f"  positions $positions%.2e, draws $keyframes%.2e $video%.2e $audio%.2e"
        )
        println(
          f"  velocities ${videoVelocity * 100}%.4f%% and ${audioVelocity * 100}%.4f%% of the largest"
        )
        assert(positions < 1e-5, keyframes < 1e-5, video < 1e-5, audio < 1e-5)
        assert(videoVelocity < 2e-3, audioVelocity < 2e-3)
      } finally ops.close()
    }
    test("MiniMax H3's guides: ComfyUI's layout, noise and one step") {
      val ops = new CpuOps
      try {
        val Seq(positions, mixed, video, audio) =
          TinyMiniMaxH3Case.guidesErrors(ops)
        println(f"  positions $positions%.2e, mixed rows $mixed%.2e")
        println(
          f"  velocities ${video * 100}%.4f%% and ${audio * 100}%.4f%% of the largest"
        )
        assert(positions < 1e-5, mixed < 1e-5, video < 2e-3, audio < 2e-3)
      } finally ops.close()
    }
    test("MiniMax H3's audio encoder") {
      val ops = new CpuOps
      try {
        val error = TinyMiniMaxH3Case.audioEncoderError(ops)
        println(f"  worst latent error: ${error * 100}%.4f%% of the largest")
        assert(error < 1e-3)
      } finally ops.close()
    }
    test("MiniMax H3's presentation: two keyframes through Qwen3-VL") {
      val ops = new CpuOps
      try {
        val error = TinyMiniMaxH3Case.presentationError(ops)
        println(
          f"  worst hidden-state error: ${error * 100}%.4f%% of the largest"
        )
        assert(error < 2e-3)
      } finally ops.close()
    }
    test("torchaudio's resampling to 32 kHz") {
      val error = TinyMiniMaxH3Case.resamplingError()
      println(f"  worst sample error: ${error * 100}%.5f%% of the largest")
      assert(error < 1e-4) // torchaudio's kernel and sums in F32
    }
    test("torch's CPU randn") {
      val error = TinyMiniMaxH3Case.randomError()
      println(f"  largest difference: $error%.2e")
      assert(error < 1e-5)
    }
  }
}
