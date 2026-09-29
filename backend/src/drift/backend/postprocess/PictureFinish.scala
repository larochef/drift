package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage

/** What becomes of the picture a tiled job paints before it is written: kept
  * as painted, or — a partial redraw or edit — pasted back into the source it
  * was cut from. Known by what it is rather than as a function, so the picture
  * shown while the job runs (`LivePicture`) can be finished one tile at a time.
  */
sealed private[postprocess] trait PictureFinish {

  /** The result, from the whole `painted` picture (cropped to what the job
    * makes).
    */
  def apply(painted: BufferedImage): BufferedImage

  /** The result's size, for a painted picture of `painted` px. */
  def size(painted: (Int, Int)): (Int, Int)

  /** Where `part` of the painted picture lands in the result. */
  def placed(part: ImageRegion): ImageRegion

  /** The result's pixels over `part` of it, row by row — `part` in the
    * result's own pixels.
    */
  def pixels(painted: BufferedImage, part: ImageRegion): Array[Int]
}

private[postprocess] object PictureFinish {

  /** The painted picture is the result. */
  case object AsPainted extends PictureFinish {
    def apply(painted: BufferedImage): BufferedImage = painted

    def size(painted: (Int, Int)): (Int, Int) = painted

    def placed(part: ImageRegion): ImageRegion = part

    def pixels(painted: BufferedImage, part: ImageRegion): Array[Int] =
      painted.getRGB(
        part.x,
        part.y,
        part.width,
        part.height,
        null,
        0,
        part.width
      )
  }

  /** The painted picture is the `window` of `source` a partial redraw or edit
    * repainted, feathered back in around `region` (`TileBlending.paste`).
    */
  final case class PastedInto(
      source: BufferedImage,
      window: ImageRegion,
      region: ImageRegion
  ) extends PictureFinish {
    def apply(painted: BufferedImage): BufferedImage =
      TileBlending.paste(source, painted, window, region)

    def size(painted: (Int, Int)): (Int, Int) =
      (source.getWidth, source.getHeight)

    def placed(part: ImageRegion): ImageRegion =
      part.copy(x = window.x + part.x, y = window.y + part.y)

    def pixels(painted: BufferedImage, part: ImageRegion): Array[Int] = {
      val result = source.getRGB(
        part.x,
        part.y,
        part.width,
        part.height,
        null,
        0,
        part.width
      )
      // Where the part meets the window, in the window's own pixels.
      val left = part.x.max(window.x)
      val top = part.y.max(window.y)
      val right = (part.x + part.width).min(window.x + window.width)
      val bottom = (part.y + part.height).min(window.y + window.height)
      if (right > left && bottom > top) {
        val inside = ImageRegion(
          left - window.x,
          top - window.y,
          right - left,
          bottom - top
        )
        val pasted =
          TileBlending.pastedPixels(source, painted, window, region, inside)
        for (v <- 0 until inside.height)
          System.arraycopy(
            pasted,
            v * inside.width,
            result,
            (top - part.y + v) * part.width + (left - part.x),
            inside.width
          )
      }
      result
    }
  }
}
