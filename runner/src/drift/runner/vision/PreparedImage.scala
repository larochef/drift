package drift.runner.vision

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.security.MessageDigest
import java.util.Base64
import javax.imageio.ImageIO

/** An image as the Qwen vision tower reads it: `patches` F32 `[gridHeight ×
  * gridWidth, 3 × frames × patch²]`, each patch's pixels channel by channel
  * then frame by frame (`(c × frames + t) × patch² + y × patch + x`),
  * normalized to [−1, 1], the patches in merge-window order (each `merge ×
  * merge` window's patches together, row by row). A still is one frame (its own
  * second one in the tower's two-frame patches); a video's pair of frames two.
  * `key` identifies the picture, so a cached prompt knows it again.
  */
final case class PreparedImage(
    patches: Array[Float],
    gridHeight: Int,
    gridWidth: Int,
    merge: Int,
    key: String,
    frames: Int = 1
) {

  /** The language model's tokens for it, one per merge window. */
  def tokens: Int = gridHeight * gridWidth / (merge * merge)

  /** Its grid in those tokens. */
  def tokenRows: Int = gridHeight / merge
  def tokenColumns: Int = gridWidth / merge
}

/** Pixels in the shape the tower reads (transformers' Qwen2-VL image
  * processor): the sides resized to multiples of `patch × merge` (Qwen's
  * `smart_resize`, keeping the aspect and the pixel count between
  * `minimumPixels` and `maximumPixels`) with PIL's antialiased bicubic, alpha
  * over white, `x / 127.5 − 1`.
  */
final case class ImageSizing(
    patch: Int,
    merge: Int,
    minimumPixels: Int,
    maximumPixels: Int
) {
  def factor: Int = patch * merge

  /** Qwen's `smart_resize`: the nearest multiples of the factor, scaled down or
    * up to fit the pixel bounds.
    */
  def size(height: Int, width: Int): (Int, Int) = {
    if (math.max(height, width).toDouble / math.min(height, width) > 200)
      throw new IllegalArgumentException(
        s"an image of ${width}×$height is too narrow (aspect over 200)"
      )
    def rounded(side: Int) =
      math.max(factor, math.rint(side.toDouble / factor).toInt * factor)
    val (h, w) = (rounded(height), rounded(width))
    if (h.toLong * w > maximumPixels) {
      val beta = math.sqrt(height.toDouble * width / maximumPixels)
      (
        math.max(factor, math.floor(height / beta / factor).toInt * factor),
        math.max(factor, math.floor(width / beta / factor).toInt * factor)
      )
    } else if (h.toLong * w < minimumPixels) {
      val beta = math.sqrt(minimumPixels.toDouble / (height.toLong * width))
      (
        math.ceil(height * beta / factor).toInt * factor,
        math.ceil(width * beta / factor).toInt * factor
      )
    } else (h, w)
  }

  def prepare(image: BufferedImage, key: String): PreparedImage = {
    val (height, width) = size(image.getHeight, image.getWidth)
    val pixels = Resampling.bicubic(
      Resampling.rgbOverWhite(image),
      image.getWidth,
      image.getHeight,
      width,
      height
    )
    PreparedImage(
      patches(pixels.map(v => v / 127.5f - 1f), height, width),
      height / patch,
      width / patch,
      merge,
      key
    )
  }

  /** Normalized pixels (`[height, width, 3]`) into merge-window-ordered
    * patches.
    */
  def patches(pixels: Array[Float], height: Int, width: Int): Array[Float] =
    framePatches(Seq(pixels), height, width)

  /** Normalized frames (each `[height, width, 3]`) into merge-window-ordered
    * patches of all of them (transformers' video patches for a pair).
    */
  def framePatches(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int
  ): Array[Float] = {
    val (gridHeight, gridWidth) = (height / patch, width / patch)
    val area = patch * patch
    val depth = frames.size
    val out = new Array[Float](gridHeight * gridWidth * 3 * depth * area)
    var row = 0
    for {
      by <- 0 until gridHeight / merge
      bx <- 0 until gridWidth / merge
      iy <- 0 until merge
      ix <- 0 until merge
    } {
      val (top, left) = ((by * merge + iy) * patch, (bx * merge + ix) * patch)
      for {
        c <- 0 until 3
        (pixels, t) <- frames.zipWithIndex
        y <- 0 until patch
        x <- 0 until patch
      }
        out((row * 3 + c) * depth * area + t * area + y * patch + x) = pixels(
          ((top + y) * width + left + x) * 3 + c
        )
      row += 1
    }
    out
  }
}

object ImageSizing {

  /** Qwen3-VL's and Qwen 3.5's: patches of 16, windows of 2 × 2, at least 64
    * tokens' pixels (transformers' default) and at most 4096 tokens' (llama.cpp
    * caps the tokens too; transformers' own cap, 16 384 tokens, costs minutes
    * on a photo).
    */
  def qwen(patch: Int, merge: Int): ImageSizing = {
    val token = patch * merge * patch * merge
    ImageSizing(patch, merge, 64 * token, 4096 * token)
  }
}

object ImageInput {

  /** An image a request carries as a `data:` URL: its picture and a key of its
    * bytes.
    */
  def fromDataUrl(url: String): Either[String, (BufferedImage, String)] =
    if (!url.startsWith("data:"))
      Left("images must come as data: URLs")
    else {
      val comma = url.indexOf(',')
      val header = url.substring(5, math.max(comma, 5))
      if (comma < 0 || !header.endsWith(";base64"))
        Left("an image data: URL must be base64")
      else {
        val bytes =
          try Right(Base64.getDecoder.decode(url.substring(comma + 1)))
          catch {
            case _: IllegalArgumentException =>
              Left("an image's base64 is invalid")
          }
        bytes.flatMap { raw =>
          Option(ImageIO.read(new ByteArrayInputStream(raw)))
            .toRight(
              s"cannot decode a ${header.stripSuffix(";base64")} image (PNG, JPEG, GIF and BMP are read)"
            )
            .map(_ -> key(raw))
        }
      }
    }

  private def key(bytes: Array[Byte]): String =
    MessageDigest
      .getInstance("SHA-256")
      .digest(bytes)
      .map(b => f"${b & 0xff}%02x")
      .mkString
}

/** PIL's resampling, which transformers' processors use: separable, the filter
  * widened by the scale when shrinking (antialiased), each pass rounded to 8
  * bits.
  */
object Resampling {

  /** `[height, width, 3]` values in 0–255, transparent pixels over white. */
  def rgbOverWhite(image: BufferedImage): Array[Float] = {
    val (w, h) = (image.getWidth, image.getHeight)
    val out = new Array[Float](w * h * 3)
    val alpha = image.getColorModel.hasAlpha
    for {
      y <- 0 until h
      x <- 0 until w
    } {
      val argb = image.getRGB(x, y)
      val a = if (alpha) ((argb >>> 24) & 0xff) / 255f else 1f
      Seq(16, 8, 0).zipWithIndex.foreach { (shift, c) =>
        val value = (argb >> shift) & 0xff
        out((y * w + x) * 3 + c) = value * a + 255f * (1 - a)
      }
    }
    out
  }

  /** PIL's resampling filters: the kernel and its support. */
  enum Filter(val support: Double) {
    case Bicubic extends Filter(2.0)
    case Lanczos extends Filter(3.0)

    def apply(x: Double): Double = this match {
      case Bicubic => cubic(x)
      case Lanczos =>
        if (math.abs(x) < 3) sinc(x) * sinc(x / 3) else 0
    }
  }

  private def sinc(x: Double): Double =
    if (x == 0) 1 else math.sin(math.Pi * x) / (math.Pi * x)

  private def cubic(x: Double): Double = {
    val a = -0.5
    val t = math.abs(x)
    if (t < 1) ((a + 2) * t - (a + 3)) * t * t + 1
    else if (t < 2) (((t - 5) * t + 8) * t - 4) * a
    else 0
  }

  /** Each output coordinate's first input and weights along one axis. */
  private def taps(
      from: Int,
      to: Int,
      filter: Filter
  ): Array[(Int, Array[Double])] = {
    val scale = from.toDouble / to
    val support = filter.support * math.max(scale, 1.0)
    val stretch = math.max(scale, 1.0)
    Array.tabulate(to) { i =>
      val center = (i + 0.5) * scale
      val first = math.max(0, (center - support + 0.5).toInt)
      val last = math.min(from, (center + support + 0.5).toInt)
      val weights =
        Array.tabulate(last - first)(j =>
          filter((first + j - center + 0.5) / stretch)
        )
      val total = weights.sum
      (first, if (total != 0) weights.map(_ / total) else weights)
    }
  }

  private def clip8(v: Double): Float =
    math.min(255.0, math.max(0.0, math.rint(v))).toFloat

  /** `[height, width, 3]` from `[fromHeight, fromWidth, 3]`. */
  def bicubic(
      pixels: Array[Float],
      fromWidth: Int,
      fromHeight: Int,
      width: Int,
      height: Int
  ): Array[Float] =
    resampled(pixels, fromWidth, fromHeight, width, height, Filter.Bicubic)

  /** `[height, width, 3]` from `[fromHeight, fromWidth, 3]` through `filter`.
    */
  def resampled(
      pixels: Array[Float],
      fromWidth: Int,
      fromHeight: Int,
      width: Int,
      height: Int,
      filter: Filter
  ): Array[Float] = {
    // horizontal, then vertical, as PIL
    val across = taps(fromWidth, width, filter)
    val wide = new Array[Float](fromHeight * width * 3)
    for {
      y <- 0 until fromHeight
      x <- 0 until width
      c <- 0 until 3
    } {
      val (first, weights) = across(x)
      var sum = 0.0
      var j = 0
      while (j < weights.length) {
        sum += weights(j) * pixels((y * fromWidth + first + j) * 3 + c)
        j += 1
      }
      wide((y * width + x) * 3 + c) = clip8(sum)
    }
    val down = taps(fromHeight, height, filter)
    val out = new Array[Float](height * width * 3)
    for {
      y <- 0 until height
      x <- 0 until width
      c <- 0 until 3
    } {
      val (first, weights) = down(y)
      var sum = 0.0
      var j = 0
      while (j < weights.length) {
        sum += weights(j) * wide(((first + j) * width + x) * 3 + c)
        j += 1
      }
      out((y * width + x) * 3 + c) = clip8(sum)
    }
    out
  }
}
