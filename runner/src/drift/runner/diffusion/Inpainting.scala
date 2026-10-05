package drift.runner.diffusion

import drift.runner.ops.Ops
import drift.runner.tensor.*

import java.awt.image.BufferedImage

/** A mask on an img2img run of a flow-matching sampler (sd-cpp's `mask_image`):
  * white is repainted, black is kept. After every step the kept part of the
  * latent is put back at the noise level the step arrived at — the init image's
  * latent under that much of the run's own noise — so the model paints the
  * white part in the presence of the rest, and the rest comes out as the init
  * image went in. A repair at a high strength otherwise moves and relights
  * everything the tile shows (`specs/52-auto-redraw.md`).
  *
  * `weights` and `noise` are one value per latent value, as the sampler's `x`
  * (`shape`).
  */
final class Inpainting(
    ops: Ops,
    shape: Shape,
    weights: Array[Float],
    noise: Array[Float]
) {
  private val repainted = ops.fromFloats(shape, weights)
  private val kept = ops.fromFloats(shape, weights.map(1f - _))
  private val noised = ops.fromFloats(shape, noise)
  private val clean = ops.allocate(DType.F32, shape)
  private val known = ops.allocate(DType.F32, shape)
  private val part = ops.allocate(DType.F32, shape)

  /** The init image's latent, before any noise is added to it. */
  def remember(init: Tensor): Unit = ops.copy(init, clean)

  /** Puts the kept part of `x` back, at noise level `sigma`. */
  def restore(x: Tensor, sigma: Float): Unit = {
    ops.scale(noised, sigma, known)
    ops.scale(clean, 1 - sigma, part)
    ops.add(known, part, known)
    ops.mul(known, kept, known)
    ops.mul(x, repainted, x)
    ops.add(x, known, x)
  }

  def release(): Unit =
    Seq(repainted, kept, noised, clean, known, part).foreach(ops.release)
}

object Inpainting {

  /** How much of each latent token `mask` has repainted, 0 to 1: the mean of
    * its pixels over the token's cell, once per channel, tokens row by row.
    */
  def weights(
      mask: BufferedImage,
      gridHeight: Int,
      gridWidth: Int,
      channels: Int
  ): Array[Float] = {
    val cellHeight = mask.getHeight.toDouble / gridHeight
    val cellWidth = mask.getWidth.toDouble / gridWidth
    val result = new Array[Float](gridHeight * gridWidth * channels)
    for {
      row <- 0 until gridHeight
      column <- 0 until gridWidth
    } {
      val top = (row * cellHeight).toInt
      val bottom =
        (((row + 1) * cellHeight).toInt).max(top + 1).min(mask.getHeight)
      val left = (column * cellWidth).toInt
      val right =
        (((column + 1) * cellWidth).toInt).max(left + 1).min(mask.getWidth)
      var sum = 0L
      for {
        y <- top until bottom
        x <- left until right
      } {
        val pixel = mask.getRGB(x, y)
        sum += ((pixel >> 16) & 0xff) + ((pixel >> 8) & 0xff) + (pixel & 0xff)
      }
      val weight = (sum / (3.0 * 255 * (bottom - top) * (right - left))).toFloat
      java.util.Arrays.fill(
        result,
        (row * gridWidth + column) * channels,
        (row * gridWidth + column + 1) * channels,
        weight
      )
    }
    result
  }
}
