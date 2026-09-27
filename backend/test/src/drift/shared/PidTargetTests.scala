package drift.shared

import utest.*

/** What size a PiD job decodes to (`specs/26-tiled-pid.md`).
  *
  * Worth testing because the failure is silent and expensive: PiD decodes a
  * quarter of the target, so a target that is not bigger than the source
  * shrinks the image first and hands back something the same size that looks
  * like the original — an hour of GPU for a slightly worse picture.
  */
object PidTargetTests extends TestSuite {

  val tests = Tests {

    test("no size given is ×4, the longest side capped") {
      assert(
        PidUpscaleRequest.target((1024, 1536), None, None) ==
          Right((4096, 6144))
      )
      // ×4 would be 18432 on the long side; the cap holds it at 16384.
      assert(
        PidUpscaleRequest.target((2048, 4608), None, None) ==
          Right((7280, 16384))
      )
    }

    test("a source already at the cap is refused, not shrunk") {
      val refused = PidUpscaleRequest.target((16384, 16384), None, None)
      assert(refused.isLeft)
      val reason = refused.left.getOrElse("")
      assert(reason.contains("4096×4096"))
      assert(reason.contains("nothing left to upscale"))
    }

    test("an explicit target no larger than the source is refused too") {
      assert(
        PidUpscaleRequest.target((2048, 2048), Some(2048), Some(2048)).isLeft
      )
      // Larger on one side is a crop of another ratio, which PiD allows.
      assert(
        PidUpscaleRequest.target((2048, 2048), Some(2048), Some(4096)) ==
          Right((2048, 4096))
      )
    }

    test("the sides an explicit target must have") {
      assert(
        PidUpscaleRequest.target((512, 512), Some(1000), Some(1002)).isLeft
      )
      assert(PidUpscaleRequest.target((512, 512), Some(1024), None).isLeft)
    }
    test("drift's runner decodes 1024 → 4096 as one tile; sd-cpp in tiles") {
      def runtime(tag: String) = Runtime(
        id = tag,
        label = tag,
        tool = RuntimeTool.SdCpp,
        backend = RuntimeBackend.Rocm,
        releaseTag = tag,
        installedAt = "/nowhere",
        modelKinds = None
      )
      def tiles(tag: String, target: (Int, Int)) =
        PidUpscaleRequest
          .tilesFor(target, PidUpscaleRequest.maxTileFor(runtime(tag)))
          .flatten
      assert(tiles(Runtime.DriftRunnerTag, (4096, 4096)).size == 1)
      assert(tiles("master-892-abc", (4096, 4096)).size > 1)
      // beyond one runner tile, tiles again: n tiles cover n × 4096
      assert(tiles(Runtime.DriftRunnerTag, (8192, 8192)).size == 4)
      assert(tiles(Runtime.DriftRunnerTag, (16384, 16384)).size == 16)
      assert(tiles("master-892-abc", (8192, 8192)).size == 49)
    }
  }
}
