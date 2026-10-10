package drift.runner.diffusion

import java.awt.image.BufferedImage
import scala.io.Source
import scala.util.Using

/** A VAE family's latents read as a small picture without its decoder
  * (`specs/54-step-preview.md`): each token (a row of `channels`, as the
  * family's transformer takes it, packed and normalized) gives `block × block`
  * RGB pixels by one matrix and a bias, so a grid of `h × w` tokens is a
  * picture of `h × block` by `w × block`. `weights` is `[block² × 3, channels +
  * 1]` row-major, row `(py × block + px) × 3 + colour`, the bias last in each
  * row; colours are in [−1, 1].
  *
  * The matrices are fitted by least squares on pictures and their latents
  * (`fit`) and kept as resources (`named`).
  */
final class LatentProjection(
    val channels: Int,
    val block: Int,
    val weights: Array[Float]
) {
  private val outputs = block * block * 3
  require(
    weights.length == outputs * (channels + 1),
    s"${weights.length} weights for $outputs outputs of ${channels + 1}"
  )

  /** The picture of `tokens` (`[gridHeight × gridWidth, channels]`, row-major).
    */
  def picture(
      tokens: Array[Float],
      gridHeight: Int,
      gridWidth: Int
  ): BufferedImage = estimate(tokens, tokens, 0f, gridHeight, gridWidth)

  /** The picture of the clean latents a velocity model's step is heading for:
    * `x − sigma × velocity`, both `[gridHeight × gridWidth, channels]`.
    */
  def estimate(
      x: Array[Float],
      velocity: Array[Float],
      sigma: Float,
      gridHeight: Int,
      gridWidth: Int
  ): BufferedImage = {
    require(
      x.length == gridHeight * gridWidth * channels && velocity.length == x.length,
      s"${x.length} values for $gridHeight × $gridWidth tokens of $channels"
    )
    frame(x, velocity, sigma, 0, gridHeight, gridWidth)
  }

  /** A video's estimate while it is sampled, a picture a latent frame: `x` and
    * `velocity` hold `frames` grids of `gridHeight × gridWidth` tokens one
    * after the other (and may hold more rows past them), spread over the clip's
    * `seconds`. The first `clean` latent frames are read as `x` holds them: the
    * ones a model keeps clean through the steps.
    */
  def clip(
      x: Array[Float],
      velocity: Array[Float],
      sigma: Float,
      frames: Int,
      gridHeight: Int,
      gridWidth: Int,
      seconds: Double,
      clean: Int = 0
  ): PreviewClip =
    PreviewClip(
      frames,
      seconds,
      index =>
        frame(
          x,
          velocity,
          if (index < clean) 0f else sigma,
          index,
          gridHeight,
          gridWidth
        )
    )

  /** `estimate` on the grid of latent frame `index`, the grids one after the
    * other in `x` and `velocity`.
    */
  private def frame(
      x: Array[Float],
      velocity: Array[Float],
      sigma: Float,
      index: Int,
      gridHeight: Int,
      gridWidth: Int
  ): BufferedImage = {
    val first = index * gridHeight * gridWidth * channels
    require(
      index >= 0 && first + gridHeight * gridWidth * channels <= math
        .min(x.length, velocity.length),
      s"${x.length} values hold no frame $index of $gridHeight × $gridWidth tokens of $channels"
    )
    val width = gridWidth * block
    val image =
      new BufferedImage(width, gridHeight * block, BufferedImage.TYPE_INT_RGB)
    val token = new Array[Float](channels)
    val colours = new Array[Int](3)
    var at = first
    var gridY = 0
    while (gridY < gridHeight) {
      var gridX = 0
      while (gridX < gridWidth) {
        var c = 0
        while (c < channels) {
          token(c) = x(at + c) - sigma * velocity(at + c)
          c += 1
        }
        at += channels
        var row = 0
        var pixel = 0
        while (pixel < block * block) {
          var colour = 0
          while (colour < 3) {
            var sum = weights(row + channels)
            c = 0
            while (c < channels) {
              sum += weights(row + c) * token(c)
              c += 1
            }
            colours(colour) = math.round(
              (math.max(-1f, math.min(1f, sum)) * 0.5f + 0.5f) * 255f
            )
            row += channels + 1
            colour += 1
          }
          image.setRGB(
            gridX * block + pixel % block,
            gridY * block + pixel / block,
            (colours(0) << 16) | (colours(1) << 8) | colours(2)
          )
          pixel += 1
        }
        gridX += 1
      }
      gridY += 1
    }
    image
  }

  /** The projection as `named` reads it: `channels block`, then a row a line.
    */
  def text: String =
    s"$channels $block\n" + weights
      .grouped(channels + 1)
      .map(_.mkString(" "))
      .mkString("", "\n", "\n")
}

object LatentProjection {

  /** The Wan 2.1 VAE's latents packed 2 × 2, 64 values a token (Krea 2,
    * Nucleus-Image, and each latent frame of Wan 2.2 A14B).
    */
  val Wan21 = "wan-2.1"

  /** The FLUX.2 VAE's packed latents, 128 values a token. */
  val Flux2 = "flux.2"

  /** The Qwen Image 2.1 VAE's latents, 64 values a token. */
  val QwenImage21 = "qwen-image-2.1"

  /** The Mage-Flow VAE's latents. */
  val MageFlow = "mage-flow"

  /** LTX 2.5's video VAE's latents, 128 values a token of 32 × 32 pixels. */
  val Ltx25 = "ltx-2.5"

  /** MiniMax H3's video VAE's latents packed 2 × 2, 96 values a token. */
  val MiniMaxH3 = "minimax-h3"

  /** The fitted projection of a VAE family, for tokens of `channels`; none
    * where the runner carries none, and its pictures then have no preview.
    */
  def named(name: String, channels: Int): Option[LatentProjection] =
    Option(getClass.getResourceAsStream(s"/previews/$name.txt"))
      .map(stream => Using.resource(Source.fromInputStream(stream))(parse))
      .filter(_.channels == channels)

  /** `LatentProjection.text`'s inverse. */
  def parse(source: Source): LatentProjection = {
    val lines = source.getLines().filter(_.trim.nonEmpty)
    val Array(channels, block) = lines.next().trim.split("\\s+").map(_.toInt)
    new LatentProjection(
      channels,
      block,
      lines.flatMap(_.trim.split("\\s+").map(_.toFloat)).toArray
    )
  }

  /** The least-squares projection of `samples`: each a picture's tokens
    * (`[gridHeight × gridWidth, channels]`) and the picture itself at `block`
    * pixels a token (`[gridHeight × block, gridWidth × block, 3]`, channels
    * last, in [−1, 1]). Solves the normal equations, the same for every output.
    */
  def fit(
      channels: Int,
      block: Int,
      samples: Seq[(Array[Float], Array[Float], Int, Int)]
  ): LatentProjection = {
    val inputs = channels + 1
    val outputs = block * block * 3
    val gram = Array.ofDim[Double](inputs, inputs)
    val moments = Array.ofDim[Double](outputs, inputs)
    val token = new Array[Double](inputs)
    token(channels) = 1.0
    samples.foreach { (tokens, pixels, gridHeight, gridWidth) =>
      val width = gridWidth * block
      for {
        gridY <- 0 until gridHeight
        gridX <- 0 until gridWidth
      } {
        val at = (gridY * gridWidth + gridX) * channels
        (0 until channels).foreach(c => token(c) = tokens(at + c).toDouble)
        var i = 0
        while (i < inputs) {
          var j = 0
          while (j <= i) {
            gram(i)(j) += token(i) * token(j)
            j += 1
          }
          i += 1
        }
        (0 until outputs).foreach { output =>
          val pixel = output / 3
          val value = pixels(
            ((gridY * block + pixel / block) * width +
              gridX * block + pixel % block) * 3 + output % 3
          )
          i = 0
          while (i < inputs) {
            moments(output)(i) += value * token(i)
            i += 1
          }
        }
      }
    }
    for {
      i <- 0 until inputs
      j <- i + 1 until inputs
    } gram(i)(j) = gram(j)(i)
    // a little ridge: channels a VAE barely uses must not blow the solve up
    val ridge = 1e-6 * (0 until inputs).map(i => gram(i)(i)).sum / inputs
    (0 until inputs).foreach(i => gram(i)(i) += ridge)
    val solved = solve(gram, moments)
    new LatentProjection(channels, block, solved.flatten.map(_.toFloat))
  }

  /** `x` of `a × xᵀ = b` for each row of `b`: Gauss-Jordan with partial
    * pivoting, on copies.
    */
  private def solve(
      a: Array[Array[Double]],
      b: Array[Array[Double]]
  ): Array[Array[Double]] = {
    val size = a.length
    val left = a.map(_.clone)
    // right(i) is row i of the right-hand sides, one column an output
    val right = Array.tabulate(size, b.length)((i, output) => b(output)(i))
    (0 until size).foreach { column =>
      val pivot = (column until size).maxBy(row => math.abs(left(row)(column)))
      val (swappedLeft, swappedRight) = (left(pivot), right(pivot))
      left(pivot) = left(column)
      right(pivot) = right(column)
      left(column) = swappedLeft
      right(column) = swappedRight
      val scale = 1.0 / left(column)(column)
      (0 until size).filter(_ != column).foreach { row =>
        val factor = left(row)(column) * scale
        if (factor != 0) {
          var k = column
          while (k < size) {
            left(row)(k) -= factor * left(column)(k)
            k += 1
          }
          k = 0
          while (k < b.length) {
            right(row)(k) -= factor * right(column)(k)
            k += 1
          }
        }
      }
    }
    Array.tabulate(b.length, size)((output, i) => right(i)(output) / left(i)(i))
  }
}
