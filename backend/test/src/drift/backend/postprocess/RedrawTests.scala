package drift.backend.postprocess

import drift.shared.{ImageRegion, RedrawRequest, TileSettings, Tiling}
import utest.*

import java.awt.image.BufferedImage

/** What a redraw hands the model and what it keeps (`specs/45`).
  *
  * Worth testing because each failure is silent: a reference cut from the wrong
  * place still looks like a picture, a step count off by one still draws
  * something, and a colour correction that shifts the detail still returns an
  * image of the right size.
  */
object RedrawTests extends TestSuite {

  private def filled(width: Int, height: Int)(pixel: (Int, Int) => Int) = {
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    for {
      x <- 0 until width
      y <- 0 until height
    }
      image.setRGB(x, y, pixel(x, y))
    image
  }

  private def rgb(red: Int, green: Int, blue: Int) =
    red << 16 | green << 8 | blue

  val tests = Tests {

    test("a tile's reference is the 3×3 block of tiles centred on it") {
      val block = Redraw.neighbourhood(
        Tiling.Tile(8192, 8192, 1280, 1280),
        (16384, 16384)
      )
      assert(block == Tiling.Tile(7168, 7168, 3328, 3328))
    }

    test("at the picture's edge the block shifts to stay inside it") {
      val block =
        Redraw.neighbourhood(Tiling.Tile(0, 15104, 1280, 1280), (16384, 16384))
      assert(block == Tiling.Tile(0, 13056, 3328, 3328))
    }

    test("a picture smaller than the block is its own block") {
      val block =
        Redraw.neighbourhood(Tiling.Tile(0, 768, 1024, 1280), (1024, 2048))
      assert(block == Tiling.Tile(0, 0, 1024, 2048))
    }

    test("enough steps are scheduled for the asked ones to run") {
      // ⌊10 × 0.4⌋ = 4; ⌊14 × 0.7⌋ = 9, ⌊15 × 0.7⌋ = 10.
      assert(Redraw.scheduledSteps(4, 0.4) == 10)
      assert(Redraw.scheduledSteps(10, 0.7) == 15)
      assert(Redraw.scheduledSteps(8, 1.0) == 8)
      for {
        steps <- 1 to 30
        strength <- List(0.15, 0.3, 0.45, 0.55, 0.8)
      } {
        val scheduled = Redraw.scheduledSteps(steps, strength)
        assert((scheduled * strength.toFloat).toInt == steps)
      }
    }

    test("a redrawn tile takes the source's colour and keeps its own detail") {
      // The source a flat colour; the redraw darker by 20 and striped.
      val source = filled(200, 200)((_, _) => rgb(150, 120, 100))
      val stripe = (x: Int) => if (x % 2 == 0) 10 else -10
      val redrawn = filled(200, 200)((x, _) =>
        rgb(130 + stripe(x), 100 + stripe(x), 80 + stripe(x))
      )
      val matched = PostProcessImages.colourMatched(redrawn, source)
      val red = (x: Int) => (matched.getRGB(x, 100) >> 16) & 0xff
      // the mean is the source's, the stripes are the redraw's
      val mean = (90 until 110).map(red).sum / 20.0
      assert(math.abs(mean - 150) <= 1)
      assert(math.abs(red(100) - red(101)) >= 18)
    }

    test(
      "a tile's own settings are found by its place, or the job is refused"
    ) {
      val tiles =
        List(Tiling.Tile(0, 0, 1280, 1280), Tiling.Tile(1024, 0, 1280, 1280))
      def settings(strengths: Double*) = tiles
        .zip(strengths)
        .map((tile, strength) =>
          TileSettings(
            ImageRegion(tile.x, tile.y, tile.width, tile.height),
            "sand",
            strength
          )
        )
      val request = RedrawRequest("configuration")
      assert(Redraw.settingsFor(request, None, tiles) == Right(Map.empty))
      val Right(found) =
        Redraw.settingsFor(
          request.copy(tiles = settings(0.0, 0.5)),
          None,
          tiles
        ): @unchecked
      assert(found(tiles(1)).strength == 0.5)
      // read for another grid: refused, not painted with a neighbour's prompt
      val moved = tiles.map(tile => tile.copy(x = tile.x + 16))
      assert(
        Redraw
          .settingsFor(request.copy(tiles = settings(0.2, 0.5)), None, moved)
          .isLeft
      )
      assert(
        Redraw
          .settingsFor(request.copy(tiles = settings(0.2)), None, tiles)
          .isLeft
      )
      // a reading is of the whole picture
      assert(
        Redraw
          .settingsFor(
            request.copy(tiles = settings(0.2, 0.5)),
            Some(ImageRegion(0, 0, 512, 512)),
            tiles
          )
          .isLeft
      )
      // nothing to do
      assert(
        Redraw
          .settingsFor(request.copy(tiles = settings(0.0, 0.0)), None, tiles)
          .isLeft
      )
    }

    test("a selection's mask is white over it and black away from it") {
      val mask =
        Redraw.selectionMask(ImageRegion(400, 300, 200, 100), 1024, 1024, 64)
      assert(mask.getWidth == 1024, mask.getHeight == 1024)
      assert((mask.getRGB(500, 350) & 0xff) == 255)
      assert((mask.getRGB(100, 100) & 0xff) == 0)
      assert((mask.getRGB(900, 900) & 0xff) == 0)
      // grown by half the margin: just outside the selection is still repainted
      assert((mask.getRGB(390, 350) & 0xff) > 128)
    }
  }
}
