package drift.backend.postprocess

import drift.shared.Tiling
import utest.*

import java.awt.image.BufferedImage

/** What an edit keeps of a tile (`specs/39-seamless-edit.md`), and how its
  * tiles are painted into the picture the next one is cut from.
  *
  * Worth testing because both fail silently: a composite that lets the model's
  * re-rendered background through looks like a normal edit until the grain and
  * the colour cast are compared with the source, and a paint that ramps the
  * wrong side leaves seams only visible once a real job has run.
  */
object EditCompositeTests extends TestSuite {

  private val Side = 160

  /** A smooth gradient, so a colour shift is not a flat offset of one value. */
  private def gradient: BufferedImage = {
    val image = BufferedImage(Side, Side, BufferedImage.TYPE_INT_RGB)
    for {
      y <- 0 until Side
      x <- 0 until Side
    } image.setRGB(x, y, rgb(60 + x / 2, 90 + y / 3, 120))
    image
  }

  private def rgb(red: Int, green: Int, blue: Int): Int =
    0xff000000 | red << 16 | green << 8 | blue

  private def channels(pixel: Int): (Int, Int, Int) =
    ((pixel >> 16) & 0xff, (pixel >> 8) & 0xff, pixel & 0xff)

  /** `source` as an edit model returns it: every pixel a little darker and with
    * a grain of its own, and the square at 60..100 turned red.
    */
  private def edited(source: BufferedImage): BufferedImage = {
    val random = new scala.util.Random(7)
    val image = BufferedImage(Side, Side, BufferedImage.TYPE_INT_RGB)
    for {
      y <- 0 until Side
      x <- 0 until Side
    } {
      val (red, green, blue) = channels(source.getRGB(x, y))
      def shifted(value: Int) =
        (value * 0.92 + random.nextGaussian() * 3).round.toInt.max(0).min(255)
      image.setRGB(
        x,
        y,
        if (x >= 60 && x < 100 && y >= 60 && y < 100) rgb(220, 30, 30)
        else rgb(shifted(red), shifted(green), shifted(blue))
      )
    }
    image
  }

  val tests = Tests {

    test("an edit that changed nothing gives the source back") {
      val source = gradient
      val result = EditComposite(source, gradient)
      assert(result.changedShare == 0.0)
      for {
        y <- 0 until Side by 17
        x <- 0 until Side by 17
      } assert(result.image.getRGB(x, y) == source.getRGB(x, y))
    }

    test("the untouched part is the source's own pixels, grain and all") {
      val source = gradient
      val result = EditComposite(source, edited(source))
      // Farther from the square than the means, the growth and the feather
      // reach (about 34 px), the mask is zero: exactly the source, not the
      // darker, grainy re-render.
      for {
        (x, y) <- List((5, 5), (150, 10), (10, 150), (150, 150), (15, 80))
      } assert(result.image.getRGB(x, y) == source.getRGB(x, y))
    }

    test("the change is kept, and only about as much as it covers") {
      val source = gradient
      val result = EditComposite(source, edited(source))
      val (red, green, _) = channels(result.image.getRGB(80, 80))
      assert(red > 180, green < 80)
      // The square is 1/16 of the tile; the means blur it, the mask is
      // grown and feathered — more than the square, far less than the tile.
      assert(result.changedShare > 0.05, result.changedShare < 0.3)
    }

    test("a painted tile ramps in over its overlap and covers the rest") {
      val picture = BufferedImage(8, 4, BufferedImage.TYPE_INT_RGB)
      val black = rgb(0, 0, 0)
      for {
        y <- 0 until 4
        x <- 0 until 8
      } picture.setRGB(x, y, black)
      val tile = BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until 4
        x <- 0 until 4
      } tile.setRGB(x, y, rgb(200, 200, 200))
      TileBlending.paint(
        picture,
        Tiling.Tile(4, 0, 4, 4),
        tile,
        left = 2,
        top = 0
      )
      // Left of the tile: untouched.
      assert(picture.getRGB(3, 1) == black)
      // In the overlap, a quarter then three quarters of the way to the tile.
      assert(channels(picture.getRGB(4, 1))._1 == 50)
      assert(channels(picture.getRGB(5, 1))._1 == 150)
      // Past it, the tile.
      assert(channels(picture.getRGB(6, 1))._1 == 200)
    }
  }
}
