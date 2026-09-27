package drift.backend.postprocess

import drift.shared.Tiling
import utest.*

/** The window a redraw with context hands the model (`specs/27-redraw.md`).
  *
  * Worth testing because a window off the model's multiple is not refused:
  * sd-cpp aligns it up and answers with a bigger image, and the tile is then
  * cut from the wrong place — a shift nobody sees until the seams do.
  */
object TileContextTests extends TestSuite {

  private val context = TileWindow.TileContext(margin = 128, sizeMultiple = 16)

  val tests = Tests {

    test("inside the picture the tile gets the whole margin on every side") {
      val window = TileWindow.windowOf(
        Tiling.Tile(1024, 1024, 1280, 1280),
        4096,
        4096,
        context
      )
      assert(window == Tiling.Tile(896, 896, 1536, 1536))
    }

    test("at the picture's edge the margin is cut short") {
      val window =
        TileWindow.windowOf(Tiling.Tile(0, 0, 1280, 1280), 4096, 1280, context)
      assert(window == Tiling.Tile(0, 0, 1408, 1280))
    }

    test("a margin that would leave the multiple is trimmed after the tile") {
      // 40 px of room after the tile: 1280 + 128 + 40 is 1448, eight over the
      // multiple, so the eight come off the far side.
      val window =
        TileWindow.windowOf(
          Tiling.Tile(200, 0, 1280, 1280),
          1520,
          1280,
          context
        )
      assert(window.x == 72, window.width == 1440)
      assert(window.width % 16 == 0)
      assert(window.x <= 200, window.x + window.width >= 1480)
    }

    test("the mask is white over the tile and black around it") {
      val tile = Tiling.Tile(1024, 1024, 1280, 1280)
      val window = TileWindow.windowOf(tile, 4096, 4096, context)
      val mask = TileWindow.maskOf(tile, window)
      assert(mask.getWidth == 1536, mask.getHeight == 1536)
      assert((mask.getRGB(0, 0) & 0xff) == 0)
      assert((mask.getRGB(127, 700) & 0xff) == 0)
      assert((mask.getRGB(128, 128) & 0xff) == 255)
      assert((mask.getRGB(1407, 1407) & 0xff) == 255)
      assert((mask.getRGB(1408, 700) & 0xff) == 0)
    }
  }
}
