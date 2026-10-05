package drift.runner.diffusion

import utest.*

import java.awt.image.BufferedImage

/** How far one SeedVR2 pass enlarges (bug 40), and what a mask keeps. */
object UpscalePassesTests extends TestSuite {

  val tests = Tests {
    test("×4 is two passes of ×2, ×2 and less are one") {
      assert(
        SeedVr2Pipeline.passes((1024, 1536), (4096, 6144)) ==
          List((2048, 3072), (4096, 6144))
      )
      assert(
        SeedVr2Pipeline.passes((1024, 1536), (2048, 3072)) == List((2048, 3072))
      )
      assert(
        SeedVr2Pipeline.passes((1024, 1024), (1536, 1536)) == List((1536, 1536))
      )
      assert(
        SeedVr2Pipeline.passes((1024, 1024), (512, 512)) == List((512, 512))
      )
    }

    test("a size in between doubles first, then goes to the size asked") {
      assert(
        SeedVr2Pipeline.passes((1000, 500), (3000, 1500)) ==
          List((2000, 1000), (3000, 1500))
      )
      assert(
        SeedVr2Pipeline.passes((256, 256), (2048, 2048)) ==
          List((512, 512), (1024, 1024), (2048, 2048))
      )
    }

    test("a mask weighs each latent token by how much of it is repainted") {
      val mask = new BufferedImage(32, 16, BufferedImage.TYPE_INT_RGB)
      for {
        x <- 16 until 32
        y <- 0 until 16
      } mask.setRGB(x, y, 0xffffff)
      // half of one token's column of pixels
      for (y <- 0 until 8) mask.setRGB(8, y, 0xffffff)
      val weights =
        Inpainting.weights(mask, gridHeight = 2, gridWidth = 4, channels = 3)
      assert(weights.length == 2 * 4 * 3)
      def token(row: Int, column: Int) = weights((row * 4 + column) * 3)
      assert(token(0, 0) == 0f, token(1, 0) == 0f)
      assert(token(0, 2) == 1f, token(1, 3) == 1f)
      assert(math.abs(token(0, 1) - 0.125f) < 1e-6f, token(1, 1) == 0f)
      // the same weight on every channel of a token
      assert(weights.slice(6, 9).distinct.length == 1)
    }
  }
}
