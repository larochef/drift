package drift.runner

import utest.*

import drift.runner.diffusion.HiDreamO1Pipeline

/** HiDream O1's host-side pieces against the official `models/pipeline.py`:
  * the distilled schedule and the `(C p1 p2)` patch layout.
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
  }
}
