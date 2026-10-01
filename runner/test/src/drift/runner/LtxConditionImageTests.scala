package drift.runner

import utest.*

import drift.runner.diffusion.LtxPipeline

import java.awt.image.BufferedImage

/** LTX 2.5's image conditions before the VAE: covered and centre-cropped to the
  * canvas, and through H.264 at CRF 18 by ffmpeg.
  */
object LtxConditionImageTests extends TestSuite {

  private def gradient(width: Int, height: Int): BufferedImage = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    for {
      y <- 0 until height
      x <- 0 until width
    }
      image.setRGB(
        x,
        y,
        (x * 255 / width) << 16 | (y * 255 / height) << 8 | 128
      )
    image
  }

  val tests = Tests {
    test("covered: the short side fills the canvas, the long one is cropped") {
      val covered = LtxPipeline.covered(gradient(200, 100), 64, 64)
      assert(covered.getWidth == 64, covered.getHeight == 64)
      // the middle half of the width: red from about a quarter to three quarters
      val left = (covered.getRGB(0, 32) >> 16) & 0xff
      val right = (covered.getRGB(63, 32) >> 16) & 0xff
      assert(math.abs(left - 64) < 8, math.abs(right - 191) < 8)
    }
    test("compressed: the same size, close to the original") {
      val image = gradient(64, 64)
      val compressed = LtxPipeline.compressed(image)
      assert(compressed.getWidth == 64, compressed.getHeight == 64)
      val worst = (for {
        y <- 0 until 64
        x <- 0 until 64
      } yield {
        val (a, b) = (image.getRGB(x, y), compressed.getRGB(x, y))
        Seq(16, 8, 0)
          .map(s => math.abs(((a >> s) & 0xff) - ((b >> s) & 0xff)))
          .max
      }).max
      println(s"  worst channel difference: $worst levels")
      assert(worst < 24)
    }
  }
}
