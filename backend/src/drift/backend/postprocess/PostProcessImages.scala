package drift.backend.postprocess

import drift.shared.*

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.nio.file.Path
import javax.imageio.ImageIO
import scala.util.Using
import scala.util.control.NonFatal

/** The image work post-processing needs, with JDK classes only: sizes, scaled
  * and padded copies, data URLs, and the checks a result goes through.
  */
object PostProcessImages {

  /** Image dimensions from the header alone — no full decode of what may be a
    * 4096² PNG.
    */
  def imageSize(file: Path): Option[(Int, Int)] =
    try
      Using.resource(ImageIO.createImageInputStream(file.toFile)) { stream =>
        val readers = ImageIO.getImageReaders(stream)
        if (!readers.hasNext) None
        else {
          val reader = readers.next()
          try {
            reader.setInput(stream)
            Some((reader.getWidth(0), reader.getHeight(0)))
          } finally reader.dispose()
        }
      }
    catch { case NonFatal(_) => None }

  /** `image` scaled down to fit within `side` px with its ratio kept — the
    * image itself when it already fits.
    */
  def fitWithin(image: BufferedImage, side: Int): BufferedImage = {
    val scale = side.toDouble / math.max(image.getWidth, image.getHeight)
    if (scale >= 1) image
    else
      scaledCopy(
        image,
        math.round(image.getWidth * scale).toInt.max(1),
        math.round(image.getHeight * scale).toInt.max(1)
      )
  }

  /** `image` centred on the smallest canvas of the `width`:`height` shape that
    * holds it, the rest neutral grey — so a resize to `width`×`height` scales
    * it evenly instead of stretching it. The image itself when it already has
    * that shape.
    */
  def letterboxed(
      image: BufferedImage,
      width: Int,
      height: Int
  ): BufferedImage = {
    val canvasWidth =
      math.max(
        image.getWidth,
        math.round(image.getHeight.toDouble * width / height).toInt
      )
    val canvasHeight =
      math.max(
        image.getHeight,
        math.round(image.getWidth.toDouble * height / width).toInt
      )
    if (canvasWidth == image.getWidth && canvasHeight == image.getHeight) image
    else {
      val canvas =
        BufferedImage(canvasWidth, canvasHeight, BufferedImage.TYPE_INT_RGB)
      val graphics = canvas.createGraphics()
      try {
        graphics.setColor(java.awt.Color(128, 128, 128))
        graphics.fillRect(0, 0, canvasWidth, canvasHeight)
        graphics.drawImage(
          image,
          (canvasWidth - image.getWidth) / 2,
          (canvasHeight - image.getHeight) / 2,
          null
        )
      } finally graphics.dispose()
      canvas
    }
  }

  /** Where `tile` sits in the reference image of `size`, as the sentence a
    * redraw *with a reference* starts each tile's prompt with: the third it is
    * centred in, and the span it covers. A redraw without one says something
    * else entirely (`Redraw.framing`).
    */
  def position(tile: Tiling.Tile, size: (Int, Int)): String = {
    val (width, height) = size
    def percent(value: Int, total: Int) =
      math.round(100.0 * (value min total) / total)
    def third(start: Int, length: Int, total: Int, names: List[String]) =
      names(((start + length / 2.0) / total * 3).toInt.min(2))
    val area = (
      third(tile.y, tile.height, height, List("top", "middle", "bottom")),
      third(tile.x, tile.width, width, List("left", "center", "right"))
    ) match {
      case ("middle", "center")   => "center"
      case (vertical, horizontal) => s"$vertical $horizontal"
    }
    s"This image is the $area part of the reference image, from " +
      s"${percent(tile.x, width)}% to ${percent(tile.x + tile.width, width)}% " +
      s"of its width and from ${percent(tile.y, height)}% to " +
      s"${percent(tile.y + tile.height, height)}% of its height."
  }

  /** `image` as a PNG data URL, the form sd-server's API takes images in. */
  def dataUrl(image: BufferedImage): String = {
    val bytes = java.io.ByteArrayOutputStream()
    ImageIO.write(image, "png", bytes)
    "data:image/png;base64," +
      java.util.Base64.getEncoder.encodeToString(bytes.toByteArray)
  }

  /** Whether every pixel of the image is black. PiD clamps `(x+1)/2` on decode,
    * so a NaN result saves as a well-formed, entirely black PNG that sd-cli
    * reports as a success.
    */
  def isBlack(file: Path): Boolean =
    Option(ImageIO.read(file.toFile)).exists { image =>
      val width = image.getWidth
      val row = new Array[Int](width)
      (0 until image.getHeight).forall { y =>
        image.getRGB(0, y, width, 1, row, 0, width)
        row.forall(pixel => (pixel & 0xffffff) == 0)
      }
    }

  /** Scales with JDK classes only: halves with bilinear filtering (a 2:1
    * bilinear pass is a box filter) until within 2× of the target, then one
    * bicubic pass. A single bicubic draw samples 16 pixels wherever it lands,
    * so a direct 4:1 shrink aliases; this approximates area averaging.
    */
  def scaledCopy(
      image: BufferedImage,
      width: Int,
      height: Int
  ): BufferedImage = {
    val kind =
      if (image.getColorModel.hasAlpha) BufferedImage.TYPE_INT_ARGB
      else BufferedImage.TYPE_INT_RGB
    def draw(source: BufferedImage, w: Int, h: Int, hint: AnyRef) = {
      val target = BufferedImage(w, h, kind)
      val graphics = target.createGraphics()
      try {
        graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, hint)
        graphics.setRenderingHint(
          RenderingHints.KEY_RENDERING,
          RenderingHints.VALUE_RENDER_QUALITY
        )
        graphics.drawImage(source, 0, 0, w, h, null)
      } finally graphics.dispose()
      target
    }
    var current = image
    while (current.getWidth / 2 >= width && current.getHeight / 2 >= height)
      current = draw(
        current,
        current.getWidth / 2,
        current.getHeight / 2,
        RenderingHints.VALUE_INTERPOLATION_BILINEAR
      )
    if (current.getWidth == width && current.getHeight == height) current
    else
      draw(current, width, height, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
  }

  /** `image` softened by a box blur of `radius` px, run twice per axis — two
    * box passes approximate a Gaussian closely enough here, and cost
    * milliseconds on a tile.
    *
    * What it is for (`specs/27-redraw.md`): an upscaler leaves its own texture
    * behind — waxy skin, doubled pores, ringing along edges — and that texture
    * is fine *structure*, exactly what a low-strength img2img pass is built to
    * preserve. The model then sharpens the artifacts instead of replacing them.
    * Taking the structure out before the pass leaves nothing to build on, so
    * the model paints its own texture there instead. A radius of zero or less
    * is the image itself.
    */
  def softened(image: BufferedImage, radius: Double): BufferedImage = {
    val r = math.round(radius).toInt
    if (r <= 0) image
    else {
      val width = image.getWidth
      val height = image.getHeight
      var pixels = image.getRGB(0, 0, width, height, null, 0, width)
      for (_ <- 0 until 2) {
        pixels = blurAxis(pixels, width, height, r, horizontal = true)
        pixels = blurAxis(pixels, width, height, r, horizontal = false)
      }
      val result = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      result.setRGB(0, 0, width, height, pixels, 0, width)
      result
    }
  }

  /** One box-blur pass along one axis, sampling clamped at the edges. */
  private def blurAxis(
      pixels: Array[Int],
      width: Int,
      height: Int,
      radius: Int,
      horizontal: Boolean
  ): Array[Int] = {
    val out = new Array[Int](pixels.length)
    val length = if (horizontal) width else height
    val lines = if (horizontal) height else width
    val window = 2 * radius + 1
    for (line <- 0 until lines) {
      def at(index: Int): Int = {
        val clamped = index.max(0).min(length - 1)
        if (horizontal) pixels(line * width + clamped)
        else pixels(clamped * width + line)
      }
      var red = 0
      var green = 0
      var blue = 0
      for (offset <- -radius to radius) {
        val pixel = at(offset)
        red += (pixel >> 16) & 0xff
        green += (pixel >> 8) & 0xff
        blue += pixel & 0xff
      }
      for (index <- 0 until length) {
        val value = 0xff000000 | (red / window) << 16 |
          (green / window) << 8 | (blue / window)
        if (horizontal) out(line * width + index) = value
        else out(index * width + line) = value
        val leaving = at(index - radius)
        val entering = at(index + radius + 1)
        red += ((entering >> 16) & 0xff) - ((leaving >> 16) & 0xff)
        green += ((entering >> 8) & 0xff) - ((leaving >> 8) & 0xff)
        blue += (entering & 0xff) - (leaving & 0xff)
      }
    }
    out
  }

  /** `image` as an RGB image with pixels of its own — a sub-image shares its
    * parent's, and painting into one changes the other.
    */
  def copyOf(image: BufferedImage): BufferedImage = {
    val copy =
      BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_RGB)
    val graphics = copy.createGraphics()
    try graphics.drawImage(image, 0, 0, null)
    finally graphics.dispose()
    copy
  }

  /** `image` grown to `width`×`height` by repeating its last column and row —
    * the image itself when it already is that size.
    */
  def padded(image: BufferedImage, width: Int, height: Int): BufferedImage =
    if (image.getWidth == width && image.getHeight == height) image
    else {
      val result = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      for {
        y <- 0 until height
        x <- 0 until width
      }
        result.setRGB(
          x,
          y,
          image.getRGB(x min (image.getWidth - 1), y min (image.getHeight - 1))
        )
      result
    }

  /** `image` scaled to cover `width`×`height` with its aspect kept, the
    * overflow cropped around the centre.
    */
  def fill(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val scale = math.max(
      width.toDouble / image.getWidth,
      height.toDouble / image.getHeight
    )
    val scaledWidth = math.round(image.getWidth * scale).toInt.max(width)
    val scaledHeight = math.round(image.getHeight * scale).toInt.max(height)
    val result = BufferedImage(
      width,
      height,
      if (image.getColorModel.hasAlpha) BufferedImage.TYPE_INT_ARGB
      else BufferedImage.TYPE_INT_RGB
    )
    val graphics = result.createGraphics()
    try
      graphics.drawImage(
        scaledCopy(image, scaledWidth, scaledHeight),
        (width - scaledWidth) / 2,
        (height - scaledHeight) / 2,
        null
      )
    finally graphics.dispose()
    result
  }
}
