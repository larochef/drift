package drift.runner.diffusion

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.util.SplittableRandom

/** Pixels in and out of the pipelines: RGB or RGBA channels-last in [−1, 1],
  * and the starting noise.
  */
object Images {

  /** A standard normal value: Box–Muller on two uniforms in (0, 1]. */
  def gaussian(random: SplittableRandom): Float = {
    val u = 1 - random.nextDouble()
    val v = random.nextDouble()
    (math.sqrt(-2 * math.log(u)) * math.cos(2 * math.Pi * v)).toFloat
  }

  /** The log's line between the last step and the pixels: a decode prints
    * nothing until it is done, minutes later for a long video.
    */
  def decoding(what: String): Unit =
    println(s"sampling done, decoding $what (VAE)")

  /** RGB or RGBA (as `values` holds 3 or 4 per pixel) in [−1, 1],
    * channels-last, to 8 bits: `round((x + 1) / 2 × 255)`.
    */
  def toImage(values: Array[Float], width: Int, height: Int): BufferedImage = {
    val channels = values.length / (width * height)
    require(
      (channels == 3 || channels == 4) && channels * width * height == values.length,
      s"${values.length} values for a $width × $height image"
    )
    val image = new BufferedImage(
      width,
      height,
      if (channels == 4) BufferedImage.TYPE_INT_ARGB
      else BufferedImage.TYPE_INT_RGB
    )
    def byte(value: Float) =
      math.round((math.max(-1f, math.min(1f, value)) * 0.5f + 0.5f) * 255f)
    for {
      y <- 0 until height
      x <- 0 until width
    } {
      val at = (y * width + x) * channels
      val alpha = if (channels == 4) byte(values(at + 3)) else 255
      image.setRGB(
        x,
        y,
        (alpha << 24) | (byte(values(at)) << 16) | (byte(values(at + 1)) << 8) |
          byte(values(at + 2))
      )
    }
    image
  }

  /** An image's pixels in [−1, 1], channels-last: `x / 127.5 − 1`; RGB, or RGBA
    * with `alpha` (opaque when the image has none).
    */
  def pixels(image: BufferedImage, alpha: Boolean = false): Array[Float] = {
    val (width, height) = (image.getWidth, image.getHeight)
    val channels = if (alpha) 4 else 3
    val out = new Array[Float](width * height * channels)
    for {
      y <- 0 until height
      x <- 0 until width
    } {
      val argb = image.getRGB(x, y)
      val at = (y * width + x) * channels
      out(at) = ((argb >> 16) & 0xff) / 127.5f - 1
      out(at + 1) = ((argb >> 8) & 0xff) / 127.5f - 1
      out(at + 2) = (argb & 0xff) / 127.5f - 1
      if (alpha) out(at + 3) = ((argb >>> 24) & 0xff) / 127.5f - 1
    }
    out
  }

  /** `image` scaled to `width × height` (bicubic), its shape not kept. */
  def resized(image: BufferedImage, width: Int, height: Int): BufferedImage =
    if (image.getWidth == width && image.getHeight == height) image
    else {
      val out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      val graphics = out.createGraphics()
      try {
        graphics.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION,
          RenderingHints.VALUE_INTERPOLATION_BICUBIC
        )
        graphics.drawImage(image, 0, 0, width, height, null)
      } finally graphics.dispose()
      out
    }

  /** sd-cpp's `resize_before_vae` for a reference: the area of a megapixel,
    * never more than the output's, the reference's shape kept, each side
    * rounded to `multiple`.
    */
  def referenceSize(
      referenceWidth: Int,
      referenceHeight: Int,
      outputWidth: Int,
      outputHeight: Int,
      multiple: Int
  ): (Int, Int) = {
    val area = math.min(1024.0 * 1024, outputWidth.toDouble * outputHeight)
    val width = math.sqrt(area * referenceWidth / referenceHeight)
    val height = width * referenceHeight / referenceWidth
    def rounded(side: Double) =
      (math.round(side / multiple) * multiple).toInt.max(multiple)
    (rounded(width), rounded(height))
  }
}
