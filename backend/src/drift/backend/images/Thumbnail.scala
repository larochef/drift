package drift.backend.images

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

/** Downscaling to a JPEG, shared by everything that needs a smaller copy of a
  * generated image: the assistant, which must fit a vision model's input, and
  * the projects list, whose covers would otherwise be multi-megabyte PNGs
  * decoded at full size behind a 20rem tile.
  */
object Thumbnail {

  /** `image` fitted inside a `maxSide` box, never enlarged. */
  def jpeg(image: BufferedImage, maxSide: Int): Array[Byte] = {
    val longest = math.max(image.getWidth, image.getHeight).max(1)
    val scale = math.min(1.0, maxSide.toDouble / longest)
    val width = math.max(1, math.round(image.getWidth * scale).toInt)
    val height = math.max(1, math.round(image.getHeight * scale).toInt)
    // RGB, not the source type: JPEG has no alpha, and a paletted or ARGB
    // source would make the writer refuse.
    val scaled = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = scaled.createGraphics()
    try {
      graphics.setRenderingHint(
        RenderingHints.KEY_INTERPOLATION,
        RenderingHints.VALUE_INTERPOLATION_BILINEAR
      )
      graphics.drawImage(image, 0, 0, width, height, null)
    } finally graphics.dispose()
    val out = ByteArrayOutputStream()
    if (!ImageIO.write(scaled, "jpeg", out))
      throw IllegalStateException("no JPEG writer available")
    out.toByteArray
  }
}
