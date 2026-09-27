package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.Ops
import drift.runner.tensor.Tensor

import java.nio.file.Path

/** The FLUX VAEs (diffusers' `AutoencoderKL` and `AutoencoderKLFlux2`), both
  * ways, from the original LDM names (`encoder.down.N.block.N`,
  * `decoder.up.N.upsample`). Channels-last throughout, group norms of 32
  * groups, every weight uploaded as BF16:
  *   - the encoder: a 3×3 in; per level two residual blocks and (but the last)
  *     a stride-2 3×3 downsampler; the middle (residual, attention, residual);
  *     norm, SiLU, a 3×3 to twice the latent channels and FLUX.2's 1×1
  *     `quant_conv`, whose first half is the mean (the latent taken, not
  *     sampled);
  *   - the decoder: FLUX.2's `post_quant_conv`, a 3×3 in, the middle; per level
  *     from the deepest three residual blocks and (but the last) nearest ×2
  *     then a 3×3; norm, SiLU, a 3×3 to RGB in [−1, 1].
  * The diffusion model's latents: FLUX.2's (32 channels at 1/8, with
  * `bn.running_mean`) packed 2×2 (128 features at 1/16, feature `c × 4 + dy × 2
  * + dx`) and normalized by the checkpoint's batch-norm statistics; FLUX.1's
  * (16 channels at 1/8, Z-Image's too) as they are, `0.3611 (z − 0.1159)`.
  */
final class FluxVae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  /** The batch norm's epsilon (diffusers' `batch_norm_eps`). */
  private val BatchNormEpsilon = 1e-4

  /** FLUX.1's latent scale and shift (its `scale_factor`, `shift_factor`). */
  private val Flux1Scale = 0.3611f
  private val Flux1Shift = 0.1159f

  private val weights = new VaeWeights(ops, source)

  private def norm(prefix: String) =
    ChannelNorm.Group(
      weights.floats(s"$prefix.weight"),
      weights.floats(s"$prefix.bias")
    )

  private def residual(prefix: String) =
    ResidualBlock(
      norm(s"$prefix.norm1"),
      weights.conv3x3(s"$prefix.conv1"),
      norm(s"$prefix.norm2"),
      weights.conv3x3(s"$prefix.conv2"),
      Option.when(weights.has(s"$prefix.nin_shortcut.weight"))(
        weights.conv1x1(s"$prefix.nin_shortcut")
      )
    )

  private def attention(prefix: String) =
    PixelAttention(
      norm(s"$prefix.norm"),
      weights.conv1x1(s"$prefix.q"),
      weights.conv1x1(s"$prefix.k"),
      weights.conv1x1(s"$prefix.v"),
      weights.conv1x1(s"$prefix.proj_out")
    )

  /** A level's residual blocks, and its resampling convolution if it has one.
    */
  final private case class Level(
      blocks: Seq[ResidualBlock],
      resample: Option[Convolution]
  )

  private def levels(prefix: String, resample: String): Seq[Level] =
    Iterator
      .from(0)
      .map(i => s"$prefix.$i")
      .takeWhile(p => weights.has(s"$p.block.0.conv1.weight"))
      .map(p =>
        Level(
          Iterator
            .from(0)
            .map(j => s"$p.block.$j")
            .takeWhile(b => weights.has(s"$b.conv1.weight"))
            .map(residual)
            .toSeq,
          Option.when(weights.has(s"$p.$resample.conv.weight"))(
            weights.conv3x3(s"$p.$resample.conv")
          )
        )
      )
      .toSeq

  private def middle(prefix: String) =
    (
      residual(s"$prefix.mid.block_1"),
      attention(s"$prefix.mid.attn_1"),
      residual(s"$prefix.mid.block_2")
    )

  if (!source.has("encoder.conv_in.weight"))
    throw new FormatException("no FLUX VAE (encoder.conv_in.weight)")

  /** FLUX.2's: packed 2×2, batch-norm normalized. */
  val packed: Boolean = source.has("bn.running_mean")

  private val encoderIn = weights.conv3x3("encoder.conv_in")
  private val downs = levels("encoder.down", "downsample")
  private val encoderMiddle = middle("encoder")
  private val encoderNorm = norm("encoder.norm_out")

  /** Latent channels: half of what the encoder's head gives (mean, log
    * variance).
    */
  val channels: Long =
    source("encoder.conv_out.weight").shape.dimensions.head / 2
  // the head, then FLUX.2's quant convolution: the mean's channels alone
  private val (encoderOut, quant) =
    if (weights.has("encoder.quant_conv.weight"))
      (
        weights.conv3x3("encoder.conv_out"),
        Some(weights.conv1x1("encoder.quant_conv").take(channels))
      )
    else (weights.conv3x3("encoder.conv_out").take(channels), None)

  private val postQuant =
    Option.when(weights.has("decoder.post_quant_conv.weight"))(
      weights.conv1x1("decoder.post_quant_conv")
    )
  private val decoderIn = weights.conv3x3("decoder.conv_in")
  private val decoderMiddle = middle("decoder")
  private val ups = levels("decoder.up", "upsample").reverse
  private val decoderNorm = norm("decoder.norm_out")
  private val decoderOut = weights.conv3x3("decoder.conv_out")

  /** Image pixels per latent row along each side: 16 packed over four levels, 8
    * unpacked.
    */
  val patch: Int = (if (packed) 2 else 1) << downs.count(_.resample.isDefined)

  /** Features per latent row: the batch norm's statistics' count, or the
    * channels.
    */
  val features: Long = if (packed) 4 * channels else channels

  // the latents are (z − mean) / deviation
  private val (mean, deviation) =
    if (packed) {
      val variance = weights.values("bn.running_var")
      (
        weights.values("bn.running_mean"),
        variance.map(v => math.sqrt(v + BatchNormEpsilon).toFloat)
      )
    } else
      (
        Array.fill(channels.toInt)(Flux1Shift),
        Array.fill(channels.toInt)(1 / Flux1Scale)
      )
  require(
    mean.length == features,
    s"${mean.length} latent statistics for $features features"
  )
  // (z − mean) / σ and back, as `modulate`'s x × (1 + scale) + shift
  private val normalizeScale =
    weights.floats(deviation.map(1 / _ - 1))
  private val normalizeShift =
    weights.floats(mean.indices.map(i => -mean(i) / deviation(i)).toArray)
  private val restoreScale = weights.floats(deviation.map(_ - 1))
  private val restoreShift = weights.floats(mean)

  /** The latents of `image` (`[H, W, 3]`, values in [−1, 1], H and W multiples
    * of `patch`) as the diffusion model takes them, `[H/patch × W/patch,
    * features]` row-major: FLUX.2's packed and normalized (`[H/16 × W/16,
    * 128]`), FLUX.1's normalized. The tensors it allocates are released but for
    * the result's.
    */
  def encode(image: Tensor): Tensor = {
    val Seq(height, width, rgb) = image.shape.dimensions
    require(
      rgb == 3 && height % patch == 0 && width % patch == 0,
      s"encode: image ${image.shape}"
    )
    val run = new VaeRun(ops)
    try {
      val copy = run.image(height, width, 3)
      ops.copy(image, copy)
      var x = run.conv3x3(copy, encoderIn)
      downs.foreach { level =>
        level.blocks.foreach(block => x = run.residual(x, block))
        level.resample.foreach(conv => x = run.conv3x3(x, conv, stride = 2))
      }
      x = run.residual(x, encoderMiddle._1)
      x = run.attend(x, encoderMiddle._2)
      x = run.residual(x, encoderMiddle._3)
      val normed = run.normSilu(x, encoderNorm)
      run.free(x)
      val head = run.conv3x3(normed, encoderOut)
      val latents = quant.fold(head)(run.conv1x1(head, _))
      val tokens = (height / patch) * (width / patch)
      val out =
        if (packed) {
          val rows = run.image(1, tokens, features)
          ops.packPatches(latents, 2, run.pixels(rows))
          run.free(latents)
          rows
        } else latents
      ops.modulate(
        run.pixels(out),
        normalizeScale,
        normalizeShift,
        run.pixels(out)
      )
      run.result(out).view(tokens, features)
    } finally run.release()
  }

  /** The image of `latents` (`[gridH × gridW, features]`, as `encode` gives
    * them): `[patch × gridH, patch × gridW, 3]`, values in [−1, 1]. The tensors
    * it allocates are released but for the result's.
    */
  def decode(latents: Tensor, gridHeight: Int): Tensor = {
    val Seq(tokens, width) = latents.shape.dimensions
    require(
      width == features && tokens % gridHeight == 0,
      s"decode: latents ${latents.shape} on $gridHeight rows"
    )
    val gridWidth = tokens / gridHeight
    val run = new VaeRun(ops)
    try {
      val restored = run.image(gridHeight, gridWidth, features)
      ops.modulate(latents, restoreScale, restoreShift, run.pixels(restored))
      val z =
        if (packed) {
          val unpacked = run.image(2L * gridHeight, 2 * gridWidth, channels)
          ops.unpackPatches(run.pixels(restored), gridHeight, 2, unpacked)
          run.free(restored)
          unpacked
        } else restored
      var x = run.conv3x3(postQuant.fold(z)(run.conv1x1(z, _)), decoderIn)
      x = run.residual(x, decoderMiddle._1)
      x = run.attend(x, decoderMiddle._2)
      x = run.residual(x, decoderMiddle._3)
      ups.foreach { level =>
        level.blocks.foreach(block => x = run.residual(x, block))
        level.resample.foreach(conv => x = run.upsample(x, conv))
      }
      val normed = run.normSilu(x, decoderNorm)
      run.free(x)
      run.result(run.conv3x3(normed, decoderOut))
    } finally run.release()
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object FluxVae {

  def open(ops: Ops, path: Path): FluxVae = {
    val source = WeightSource.open(ops, path)
    try new FluxVae(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
