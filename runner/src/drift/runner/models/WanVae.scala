package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.Ops
import drift.runner.tensor.Tensor

import java.nio.file.Path

/** The Wan 2.1 VAE on one image (Krea 2's, Qwen Image's and Wan's VAE;
  * diffusers' `AutoencoderKLQwenImage`, sd-cpp's `wan_vae.hpp`), from the
  * original checkpoint names (`encoder.*`, `conv1`, `decoder.*`, `conv2`). On a
  * single frame its causal 3-D convolutions are 2-D ones with their last
  * temporal slice, and its temporal resamplers skip their time convolution.
  * Channels-last throughout:
  *   - the encoder, when the checkpoint has one: `encoder.conv1` (3×3);
  *     `downsamples`: residual blocks, and stride-2 3×3 convolutions; the
  *     middle; the head to twice the latent channels, and `conv1` (1×1, the
  *     quant convolution) whose first half is the mean (taken, not sampled);
  *   - `conv2` (1×1, the post-quant convolution), `decoder.conv1` (3×3);
  *   - the middle: a residual block, single-head attention over the pixels, a
  *     residual block;
  *   - `upsamples`: residual blocks, and nearest ×2 upsampling followed by a
  *     3×3 convolution that halves the channels;
  *   - the head: norm, SiLU, a 3×3 convolution to RGB in [−1, 1]. Its norms are
  *     RMS over the channels (`F.normalize × √C × γ`). Every weight goes to the
  *     GPU as BF16 (the GEMMs' type).
  */
final class WanVae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  /** Wan 2.1's per-channel latent statistics: the diffusion model's latents are
    * `(z − mean) / std`.
    */
  val LatentMean: Array[Float] = Array(-0.7571f, -0.7089f, -0.9113f, 0.1075f,
    -0.1745f, 0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f,
    -0.1922f, -0.9497f, 0.2503f, -0.2921f)
  val LatentStd: Array[Float] = Array(2.8184f, 1.4541f, 2.3275f, 2.6558f,
    1.2196f, 1.7708f, 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
    1.6382f, 1.1253f, 2.8251f, 1.9160f)

  private val weights = new VaeWeights(ops, source)

  private val layers = new WanLayers(weights)
  import layers.{norm, residual}

  private enum Stage {
    case Block(residual: ResidualBlock)
    case Resample(conv: Convolution)
  }

  /** A side's stages: residual blocks, and its resampling convolutions (the
    * encoder's stride 2, the decoder's after nearest ×2).
    */
  private def stages(prefix: String) =
    Iterator
      .from(0)
      .map(i => s"$prefix.$i")
      .takeWhile(p =>
        source.has(s"$p.residual.0.gamma") || source.has(
          s"$p.resample.1.weight"
        )
      )
      .map(p =>
        if (source.has(s"$p.residual.0.gamma")) Stage.Block(residual(p))
        else Stage.Resample(weights.conv3x3(s"$p.resample.1"))
      )
      .toSeq

  /** The encoder: its input, stages, middle, head, and the quant convolution's
    * mean half.
    */
  final private case class Encoder(
      input: Convolution,
      stages: Seq[Stage],
      middle: (ResidualBlock, PixelAttention, ResidualBlock),
      headNorm: ChannelNorm,
      head: Convolution,
      quant: Convolution
  )
  private val encoder = Option.when(source.has("encoder.conv1.weight")) {
    val head = weights.conv3x3("encoder.head.2")
    Encoder(
      weights.conv3x3("encoder.conv1"),
      stages("encoder.downsamples"),
      layers.middle("encoder"),
      norm("encoder.head.0"),
      head,
      weights.conv1x1("conv1").take(head.outChannels / 2)
    )
  }

  if (!source.has("decoder.conv1.weight"))
    throw new FormatException("no Wan VAE decoder (decoder.conv1.weight)")
  private val postQuant = weights.conv1x1("conv2")
  private val input = weights.conv3x3("decoder.conv1")
  private val middle = layers.middle("decoder")
  private val upsamples = stages("decoder.upsamples")
  private val headNorm = norm("decoder.head.0")
  private val head = weights.conv3x3("decoder.head.2")
  private val latentScale = weights.floats(LatentStd.map(_ - 1f))
  private val latentShift = weights.floats(LatentMean)
  // (z − mean) / std, as `modulate`'s x × (1 + scale) + shift
  private val normalizeScale = weights.floats(LatentStd.map(1 / _ - 1))
  private val normalizeShift =
    weights.floats(
      LatentMean.indices.map(i => -LatentMean(i) / LatentStd(i)).toArray
    )

  /** Image pixels per latent pixel along each side: 8. */
  val scale: Int = 1 << upsamples.count(_.isInstanceOf[Stage.Resample])

  /** The latents of `image` (`[H, W, 3]`, values in [−1, 1], H and W multiples
    * of `scale`), normalized as the diffusion models take them: `[H/8, W/8,
    * 16]`. The tensors it allocates are released but for the result's.
    */
  def encode(image: Tensor): Tensor = {
    val parts = encoder.getOrElse(
      throw new FormatException("no Wan VAE encoder (encoder.conv1.weight)")
    )
    val Seq(height, width, rgb) = image.shape.dimensions
    require(
      rgb == 3 && height % scale == 0 && width % scale == 0,
      s"encode: image ${image.shape}"
    )
    val run = new VaeRun(ops)
    try {
      val copy = run.image(height, width, 3)
      ops.copy(image, copy)
      var x = run.conv3x3(copy, parts.input)
      parts.stages.foreach {
        case Stage.Block(block)   => x = run.residual(x, block)
        case Stage.Resample(conv) => x = run.conv3x3(x, conv, stride = 2)
      }
      x = run.middle(x, parts.middle)
      val normed = run.normSilu(x, parts.headNorm)
      run.free(x)
      val latents =
        run.conv1x1(run.conv3x3(normed, parts.head), parts.quant)
      ops.modulate(
        run.pixels(latents),
        normalizeScale,
        normalizeShift,
        run.pixels(latents)
      )
      run.result(latents)
    } finally run.release()
  }

  /** The decoded image of `latents` (the diffusion model's, channels-last
    * `[h, w, 16]`): `[8h, 8w, 3]`, values in [−1, 1]. The tensors it allocates
    * are released but for the result's.
    */
  def decode(latents: Tensor): Tensor = {
    val Seq(h, w, channels) = latents.shape.dimensions
    require(channels == 16, s"decode: latents ${latents.shape}")
    val run = new VaeRun(ops)
    import run.pixels
    try {
      // the latents un-normalized, then the post-quant 1×1 and the input 3×3
      val z = run.image(h, w, 16)
      ops.modulate(pixels(latents), latentScale, latentShift, pixels(z))
      var x = run.conv3x3(run.conv1x1(z, postQuant), input)
      x = run.middle(x, middle)
      upsamples.foreach {
        case Stage.Block(block)   => x = run.residual(x, block)
        case Stage.Resample(conv) => x = run.upsample(x, conv)
      }
      val normed = run.normSilu(x, headNorm)
      run.free(x)
      run.result(run.conv3x3(normed, head))
    } finally run.release()
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object WanVae {

  def open(ops: Ops, path: Path): WanVae = {
    val source = WeightSource.open(ops, path)
    try new WanVae(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
