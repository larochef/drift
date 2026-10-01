package drift.runner

import utest.*

import drift.runner.ops.CpuOps

/** MiniMax H3's references (ref2va) and Fun ControlNet union against the golden
  * tiny cases.
  */
object TinyMiniMaxH3ReferencesTests extends TestSuite {
  val tests = Tests {
    test("MiniMax H3's ref2va presentation: an image, two clips, sounds") {
      val ops = new CpuOps
      try {
        val (ids, rows, hidden, sizes) =
          TinyMiniMaxH3ReferencesCase.presentationErrors(ops)
        println(
          f"  ids off: $ids, vision rows off: $rows, video sizes off: $sizes"
        )
        println(
          f"  worst hidden-state error: ${hidden * 100}%.4f%% of the largest"
        )
        assert(ids == 0, rows == 0, sizes == 0, hidden < 2e-3)
      } finally ops.close()
    }
    test("MiniMax H3's ref2va normalization: 24 fps, canvases, LANCZOS") {
      val (timing, canvases, resized) =
        TinyMiniMaxH3ReferencesCase.normalizationErrors()
      println(
        f"  frames off: $timing, canvases off: $canvases, LANCZOS within $resized%.3f of 255"
      )
      assert(timing == 0, canvases == 0, resized <= 1.01)
    }
    test("MiniMax H3's ref2va: layout, draws, one step") {
      val ops = new CpuOps
      try {
        val Seq(
          positions,
          references,
          video,
          audio,
          videoVelocity,
          audioVelocity
        ) =
          TinyMiniMaxH3ReferencesCase.ref2vaErrors(ops)
        println(
          f"  positions $positions%.2e, draws $references%.2e $video%.2e $audio%.2e"
        )
        println(
          f"  velocities ${videoVelocity * 100}%.4f%% and ${audioVelocity * 100}%.4f%% of the largest"
        )
        assert(positions < 1e-5, references < 1e-5, video < 1e-5, audio < 1e-5)
        assert(videoVelocity < 2e-3, audioVelocity < 2e-3)
      } finally ops.close()
    }
    test(
      "MiniMax H3's Fun ControlNet: ComfyUI's patch, with and without a mask"
    ) {
      val ops = new CpuOps
      try {
        TinyMiniMaxH3ReferencesCase
          .controlErrors(ops)
          .zip(Seq("plain", "masked"))
          .foreach { (errors, name) =>
            val Seq(input, latent, video, audio) = errors
            println(
              f"  $name: encoder input $input%.2e, latents $latent%.2e, velocities ${video * 100}%.4f%% and ${audio * 100}%.4f%%"
            )
            assert(input < 1e-5, latent < 1e-5, video < 2e-3, audio < 2e-3)
          }
      } finally ops.close()
    }
  }
}
