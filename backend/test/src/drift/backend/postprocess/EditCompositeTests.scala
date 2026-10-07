package drift.backend.postprocess

import drift.shared.{ImageRegion, Tiling}
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

    test("a selection keeps the changed parts that touch it, whole") {
      // Two things changed: a bar that runs from inside the selection far out
      // of it, and a square nowhere near it — further than holes are closed
      // over, or the two would be one.
      val side = 320
      val source = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      val model = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until side
        x <- 0 until side
      } {
        source.setRGB(x, y, rgb(60 + x / 4, 90 + y / 6, 120))
        val bar = x >= 20 && x < 40 && y >= 10 && y < 150
        val square = x >= 240 && x < 290 && y >= 240 && y < 290
        model.setRGB(
          x,
          y,
          if (bar || square) rgb(220, 30, 30) else source.getRGB(x, y)
        )
      }
      val selection = ImageRegion(10, 10, 40, 30)
      val result = EditComposite(source, model, Some(selection))
      // the bar, down to its end, well outside the selection
      assert(result.image.getRGB(30, 140) == model.getRGB(30, 140))
      // the square does not touch the selection: the source's pixels
      assert(result.image.getRGB(265, 265) == source.getRGB(265, 265))
      // without a selection both are kept
      val all = EditComposite(source, model)
      assert(all.image.getRGB(265, 265) == model.getRGB(265, 265))
    }

    test("a new texture at nearly the same colour is kept whole") {
      // Cream wool over a beige shirt: the colour barely moves, the texture
      // does, and a patch in the middle happens to look like the source. A
      // mask read off the colour alone keeps half of it.
      val side = 256
      val source = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      val knitted = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until side
        x <- 0 until side
      } {
        val base = rgb(150 + x / 8, 140 + y / 8, 120)
        source.setRGB(x, y, base)
        val inside = x >= 70 && x < 190 && y >= 70 && y < 190
        val plain = x >= 120 && x < 140 && y >= 120 && y < 140
        knitted.setRGB(
          x,
          y,
          if (inside && !plain) {
            val stitch = if ((x / 3 + y / 3) % 2 == 0) 60 else -60
            rgb(160 + x / 8 + stitch, 150 + y / 8 + stitch, 130 + stitch)
          } else base
        )
      }
      val result = EditComposite(source, knitted)
      def mask(x: Int, y: Int) = channels(result.mask.getRGB(x, y))._1
      // the garment, the patch in its middle included
      assert(mask(90, 90) > 230, mask(130, 130) > 230, mask(180, 100) > 230)
      // and nothing far from it
      assert(mask(10, 10) == 0, mask(245, 128) == 0)
      assert(result.image.getRGB(10, 10) == source.getRGB(10, 10))
    }

    test("a picture the model moved is put back before it is compared") {
      val side = 256
      val random = new scala.util.Random(11)
      val source = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until side
        x <- 0 until side
      } {
        val grain = random.nextInt(60)
        source.setRGB(x, y, rgb(80 + x / 4 + grain, 90 + grain, 70 + y / 4))
      }
      // the same picture three pixels to the right and one down, a red square on it
      val returned = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until side
        x <- 0 until side
      } returned.setRGB(
        x,
        y,
        if (x >= 100 && x < 150 && y >= 100 && y < 150) rgb(220, 30, 30)
        else source.getRGB((x - 3).max(0), (y - 1).max(0))
      )
      val result = EditComposite(source, returned)
      assert(result.shift == (-3, -1))
      // Compared where it lies, every grain of the picture would be a change.
      assert(result.changedShare < 0.15)
      assert(result.image.getRGB(30, 200) == source.getRGB(30, 200))
      val (red, green, _) = channels(result.image.getRGB(122, 124))
      assert(red > 180, green < 80)
    }

    test("a carried-up edit leaves the band its mask is wide of") {
      // The mask read where the edit was made is some pixels wide of the
      // object; enlarged four times it is forty wide, and what the model drew
      // of the wall there would come along with the garment.
      val side = 320
      val source = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      val upscaled = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      val changed = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
      val random = new scala.util.Random(3)
      for {
        y <- 0 until side
        x <- 0 until side
      } {
        val base = rgb(100 + x / 4, 110 + y / 4, 130)
        source.setRGB(x, y, base)
        val (red, green, blue) = channels(base)
        def grainy(value: Int) =
          (value + random.nextGaussian() * 6).round.toInt.max(0).min(255)
        val object_ = x >= 120 && x < 200 && y >= 120 && y < 200
        upscaled.setRGB(
          x,
          y,
          if (object_) rgb(220, 30, 30)
          else rgb(grainy(red), grainy(green), grainy(blue))
        )
        val marked = x >= 80 && x < 240 && y >= 80 && y < 240
        changed.setRGB(x, y, if (marked) rgb(255, 255, 255) else rgb(0, 0, 0))
      }
      val result = EditComposite.carried(source, upscaled, changed, by = 4)
      val (red, green, _) = channels(result.image.getRGB(160, 160))
      assert(red > 180, green < 80)
      // inside the mask, off the object: the source, not the grainy wall
      assert(result.image.getRGB(95, 160) == source.getRGB(95, 160))
      assert(result.image.getRGB(160, 225) == source.getRGB(160, 225))
      assert(result.image.getRGB(10, 10) == source.getRGB(10, 10))
    }

    test("a paste kept within reach changes nothing past it") {
      val base = BufferedImage(100, 4, BufferedImage.TYPE_INT_RGB)
      val patch = BufferedImage(100, 4, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until 4
        x <- 0 until 100
      } {
        base.setRGB(x, y, rgb(0, 0, 0))
        patch.setRGB(x, y, rgb(200, 200, 200))
      }
      val window = drift.shared.ImageRegion(0, 0, 100, 4)
      val region = drift.shared.ImageRegion(40, 0, 20, 4)
      val wide = TileBlending.paste(base, patch, window, region)
      val near = TileBlending.paste(base, patch, window, region, Some(10))
      // to the window's edge, the ramp reaches a pixel 30 px from the region
      assert(channels(wide.getRGB(10, 1))._1 > 0)
      // within reach it does not, and is half way 5 px from it
      assert(channels(near.getRGB(10, 1))._1 == 0)
      assert(channels(near.getRGB(29, 1))._1 == 0)
      assert(channels(near.getRGB(35, 1))._1 == 110)
      assert(channels(near.getRGB(50, 1))._1 == 200)
      assert(channels(near.getRGB(64, 1))._1 == 110)
      assert(channels(near.getRGB(75, 1))._1 == 0)
    }

    test(
      "the box of what changed is grown, on multiples of 16, inside the pass"
    ) {
      val mask = BufferedImage(320, 240, BufferedImage.TYPE_INT_RGB)
      assert(EditCarry.changedBox(mask).isEmpty)
      for {
        y <- 100 until 150
        x <- 250 until 300
      } mask.setRGB(x, y, rgb(255, 255, 255))
      val box = EditCarry.changedBox(mask).get
      // 250 - 48 = 202 → 192; 300 + 48 = 348 → the pass's edge, 320
      assert(box.x == 192, box.width == 128)
      // 100 - 48 = 52 → 48; 150 + 48 = 198 → 208
      assert(box.y == 48, box.height == 160)
    }

    test("the pass an edit is reduced to, and how far it is carried back") {
      import drift.shared.EditRequest.{Pass, passOf}
      // what the model takes at once stays as it is
      assert(passOf(1024, 1536, 16) == Pass(1024, 1536, 1))
      assert(passOf(1000, 1500, 16) == Pass(1008, 1504, 1))
      // a 2k picture is halved, a 4k one quartered
      assert(passOf(2048, 3072, 16) == Pass(1024, 1536, 2))
      assert(passOf(4096, 6144, 16) == Pass(1024, 1536, 4))
      assert(passOf(4096, 4096, 16) == Pass(1536, 1536, 2))
      // an 8k one is brought within the pass and comes back ×4
      assert(passOf(8192, 8192, 16) == Pass(1536, 1536, 4))
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
