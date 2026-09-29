package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage

/** Putting decoded tiles back into one image (`specs/26-tiled-pid.md`): the
  * feather-blend of a tiled job, the same ramps painted tile by tile into a
  * picture an edit keeps cutting from (`specs/39-seamless-edit.md`), and the
  * feathered paste a partial redraw ends with (`specs/27-redraw.md`). The
  * geometry these work on — the tiles and the window — is
  * `drift.shared.Tiling`, which the browser shares; nothing here knows what
  * produced a tile, so a redraw blends the way PiD does.
  */
object TileBlending {

  /** Blends decoded tiles into one `width`×`height` image: a row's tiles left
    * to right into a strip, the strip under the rows above it, every overlap a
    * linear ramp from what is already painted to the new tile — no seam is a
    * hard edge. `load` must return an image of the tile's size; one row of
    * tiles is held at a time.
    */
  def blend(
      rows: List[List[Tiling.Tile]],
      width: Int,
      height: Int,
      load: Tiling.Tile => BufferedImage
  ): BufferedImage = {
    val image = new Array[Int](width * height)
    var paintedRows = 0
    rows.foreach { row =>
      val top = row.head.y
      val stripHeight = row.head.height
      val strip = new Array[Int](width * stripHeight)
      var paintedColumns = 0
      row.foreach { tile =>
        val source = load(tile)
          .getRGB(0, 0, tile.width, tile.height, null, 0, tile.width)
        val ramp = paintedColumns - tile.x
        for {
          y <- 0 until stripHeight
          u <- 0 until tile.width
        } {
          val index = y * width + tile.x + u
          val pixel = source(y * tile.width + u)
          strip(index) =
            if (u >= ramp) pixel else mix(strip(index), pixel, (u + 0.5) / ramp)
        }
        paintedColumns = tile.x + tile.width
      }
      val ramp = paintedRows - top
      for (v <- 0 until stripHeight) {
        val weight = (v + 0.5) / ramp
        val offset = (top + v) * width
        for (x <- 0 until width) {
          val pixel = strip(v * width + x)
          image(offset + x) =
            if (v >= ramp) pixel else mix(image(offset + x), pixel, weight)
        }
      }
      paintedRows = top + stripHeight
    }
    val result = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    result.setRGB(0, 0, width, height, image, 0, width)
    result
  }

  /** Paints one tile into `picture` where it lies, in place: a linear ramp from
    * what is already there across the `left` columns and `top` rows the tile
    * shares with tiles painted before it — its neighbour in the row and the row
    * above — full weight everywhere else. Where both overlaps meet, the smaller
    * weight holds. Painted in order, the tiles give what `blend` gives, but the
    * picture is whole after every tile, so the next one can be cut from it.
    */
  def paint(
      picture: BufferedImage,
      tile: Tiling.Tile,
      painted: BufferedImage,
      left: Int,
      top: Int
  ): Unit = {
    val below = picture.getRGB(
      tile.x,
      tile.y,
      tile.width,
      tile.height,
      null,
      0,
      tile.width
    )
    val above =
      painted.getRGB(0, 0, tile.width, tile.height, null, 0, tile.width)
    def weight(position: Int, overlap: Int): Double =
      if (position >= overlap) 1.0 else (position + 0.5) / overlap
    for {
      v <- 0 until tile.height
      u <- 0 until tile.width
    } {
      val index = v * tile.width + u
      below(index) =
        mix(below(index), above(index), weight(u, left) min weight(v, top))
    }
    picture.setRGB(
      tile.x,
      tile.y,
      tile.width,
      tile.height,
      below,
      0,
      tile.width
    )
  }

  /** `base` with `patch` — the `window` of it a partial redraw repainted — put
    * back in, at full weight inside `region` and ramping to nothing at the
    * window's edges. So the model works on a window wide enough to paint well,
    * only what the user selected changes, and the change has no box edge: the
    * margin between the region and the window is the ramp. A side with no
    * margin (the window met the image's edge there) keeps full weight, since
    * there is nothing beyond it to blend into.
    */
  def paste(
      base: BufferedImage,
      patch: BufferedImage,
      window: ImageRegion,
      region: ImageRegion
  ): BufferedImage = {
    val result = PostProcessImages.copyOf(base)
    val whole = ImageRegion(0, 0, window.width, window.height)
    result.setRGB(
      window.x,
      window.y,
      window.width,
      window.height,
      pastedPixels(base, patch, window, region, whole),
      0,
      window.width
    )
    result
  }

  /** What `paste` makes of `part` of the window alone — in the window's own
    * pixels, row by row — so a picture shown while the job runs can be brought
    * up to date one tile at a time instead of pasting the whole window again.
    */
  def pastedPixels(
      base: BufferedImage,
      patch: BufferedImage,
      window: ImageRegion,
      region: ImageRegion,
      part: ImageRegion
  ): Array[Int] = {
    val pixels = base.getRGB(
      window.x + part.x,
      window.y + part.y,
      part.width,
      part.height,
      null,
      0,
      part.width
    )
    val painted =
      patch.getRGB(part.x, part.y, part.width, part.height, null, 0, part.width)
    for (v <- 0 until part.height) {
      val y = window.y + part.y + v
      val weightY = ramp(y, region.y, region.height, window.y, window.height)
      for (u <- 0 until part.width) {
        val x = window.x + part.x + u
        val weightX = ramp(x, region.x, region.width, window.x, window.width)
        val index = v * part.width + u
        pixels(index) = mix(pixels(index), painted(index), weightX min weightY)
      }
    }
    pixels
  }

  /** How much of the repainted window a pixel at `position` on one axis keeps:
    * all of it within the region, then down to nothing across the margin that
    * separates the region from the window's edge.
    */
  private def ramp(
      position: Int,
      regionStart: Int,
      regionLength: Int,
      windowStart: Int,
      windowLength: Int
  ): Double = {
    val weight =
      if (position < regionStart) {
        val room = regionStart - windowStart
        if (room <= 0) 1.0 else (position - windowStart + 0.5) / room
      } else if (position >= regionStart + regionLength) {
        val room = windowStart + windowLength - regionStart - regionLength
        if (room <= 0) 1.0
        else (windowStart + windowLength - position - 0.5) / room
      } else 1.0
    weight.max(0.0).min(1.0)
  }

  /** `painted` moved `weight` of the way towards `tile`, per RGB channel. */
  private def mix(painted: Int, tile: Int, weight: Double): Int = {
    def channel(shift: Int) =
      math
        .round(
          ((painted >> shift) & 0xff) * (1 - weight) +
            ((tile >> shift) & 0xff) * weight
        )
        .toInt
    0xff000000 | channel(16) << 16 | channel(8) << 8 | channel(0)
  }
}
