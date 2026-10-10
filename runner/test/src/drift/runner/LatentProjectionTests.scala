package drift.runner

import scala.io.Source
import java.util.SplittableRandom

import utest.*

import drift.runner.diffusion.{Images, LatentProjection}

/** The step preview's latent-to-RGB projection (`specs/54-step-preview.md`):
  * the fit, the picture's layout, and the resources the families read.
  */
object LatentProjectionTests extends TestSuite {
  val tests = Tests {
    test("a token draws its block of pixels, row by row") {
      // 1 channel, 2 × 2 pixels a token: pixel p's red is (p + 1) / 4 × token
      val weights = (0 until 4).flatMap { pixel =>
        Seq((pixel + 1) / 4f, 0f, 0f, -1f, 0f, -1f)
      }.toArray
      val projection = new LatentProjection(1, 2, weights)
      // a row of two tokens: 1 and −1
      val picture = projection.picture(Array(1f, -1f), 1, 2)
      def red(x: Int, y: Int) = (picture.getRGB(x, y) >> 16) & 0xff
      assert(
        picture.getWidth == 4,
        picture.getHeight == 2,
        // 0.25, 0.5 over 0.75, 1 for the first token
        Seq(red(0, 0), red(1, 0), red(0, 1), red(1, 1)) ==
          Seq(159, 191, 223, 255),
        red(2, 0) == 96,
        red(3, 1) == 0,
        (picture.getRGB(0, 0) & 0xffff) == 0
      )
    }
    test("the estimate is x − σ × v") {
      val projection =
        new LatentProjection(1, 1, Array(1f, 0f, 1f, 0f, 1f, 0f))
      val picture =
        projection.estimate(Array(0.5f), Array(1f), 0.5f, 1, 1)
      assert((picture.getRGB(0, 0) & 0xff) == 128)
    }
    test("the fit finds the projection the samples were made with") {
      val random = new SplittableRandom(7)
      val (channels, block, height, width) = (5, 2, 6, 4)
      val truth = new LatentProjection(
        channels,
        block,
        Array.fill(block * block * 3 * (channels + 1))(
          (random.nextDouble() - 0.5).toFloat / 4
        )
      )
      val samples = Seq.fill(3) {
        val tokens =
          Array.fill(height * width * channels)(Images.gaussian(random) / 2)
        val pixels = new Array[Float](height * block * width * block * 3)
        for {
          gridY <- 0 until height
          gridX <- 0 until width
          output <- 0 until block * block * 3
        } {
          val row = output * (channels + 1)
          val at = (gridY * width + gridX) * channels
          val pixel = output / 3
          pixels(
            ((gridY * block + pixel / block) * width * block +
              gridX * block + pixel % block) * 3 + output % 3
          ) = truth.weights(row + channels) + (0 until channels)
            .map(c => truth.weights(row + c) * tokens(at + c))
            .sum
        }
        (tokens, pixels, height, width)
      }
      val fitted = LatentProjection.fit(channels, block, samples)
      val worst = fitted.weights
        .zip(truth.weights)
        .map((a, b) => math.abs(a - b))
        .max
      assert(worst < 1e-3)
      val read = LatentProjection.parse(Source.fromString(fitted.text))
      assert(
        read.channels == channels,
        read.block == block,
        read.weights.sameElements(fitted.weights)
      )
    }
    test("the families' projections are the tokens' width") {
      import LatentProjection.*
      assert(
        named(Wan21, 64).isDefined,
        named(Flux2, 128).isDefined,
        named(QwenImage21, 64).isDefined,
        named(MageFlow, 128).isDefined,
        // another model on the same VAE's name, with other tokens
        named(Flux2, 16).isEmpty,
        named("none", 64).isEmpty
      )
    }
  }
}
