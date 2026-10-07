package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.Ops
import drift.runner.tensor.Tensor

import java.nio.file.Path

/** The Qwen Image 2.1 VAE both ways (diffusers' `AutoencoderKLQwenImage21`,
  * sd-cpp's `wan_vae.hpp` in its Wan 2.2 form) from the original Wan names
  * (`encoder.downsamples.N.downsamples.M`, `decoder.upsamples.N.upsamples.M`,
  * `conv1`/`conv2` for the quant convolutions). 64 latent channels at 1/16,
  * RGBA pixels. On a single frame its causal 3-D convolutions are 2-D ones and
  * its temporal resamplers skip their time convolution. Channels-last
  * throughout, RMS channel norms, every weight uploaded as BF16:
  *   - each level of the encoder: residual blocks, a stride-2 3×3 (but the
  *     last), plus a shortcut of its input averaged down onto its output's
  *     channels (`averageDown`; the last level's input as it is);
  *   - each level of the decoder: residual blocks, nearest ×2 then a 3×3 (but
  *     the last), plus a shortcut of its input duplicated up onto its output's
  *     channels (`duplicateUp`; none for the last);
  *   - the temporal levels (those with a `time_conv`) shuffle over two frames,
  *     of which the image is the last;
  *   - the latent is the mean (the first 64 of the encoder's 128 channels),
  *     normalized by the per-channel statistics.
  * GRN's HBQ tokenizer is the same network in RGB with its pixels shuffled 2 ×
  * 2 into 12 channels and three resampling levels; its decoder runs here
  * (`pixelPatch`), on latents taken as they are.
  */
final class QwenImage21Vae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  /** The diffusion model's latents are `(z − mean) / std`. */
  val LatentMean: Array[Float] = Array(0.5126f, 0.7721f, -0.0631f, 1.3506f,
    -0.7855f, -2.1025f, -0.3458f, 1.3722f, 1.8873f, -1.7177f, -0.651f, 0.2732f,
    0.7562f, -0.6163f, -1.0277f, 3.8363f, 2.021f, 0.0472f, 0.932f, 2.0087f,
    2.4954f, -0.1391f, -1.4249f, 1.8464f, -0.5236f, 1.2826f, 3.7046f, -1.3035f,
    2.7286f, -1.4518f, -1.9036f, -1.9955f, -0.0342f, -1.0265f, -0.7636f,
    3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f, -1.0914f, -1.9435f, -0.2784f,
    -1.368f, 0.4809f, -0.4433f, 0.3764f, 0.5729f, -2.0595f, 1.096f, -1.326f,
    -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f, 0.3026f, 1.9373f, 1.4937f,
    0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f)
  val LatentStd: Array[Float] = Array(3.2001f, 3.2936f, 3.4321f, 3.0091f,
    3.1061f, 4.0379f, 4.0705f, 3.791f, 3.0785f, 3.65f, 3.9308f, 3.0904f,
    2.8778f, 3.7675f, 3.732f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f,
    2.9249f, 3.9454f, 3.0988f, 4.2489f, 3.4896f, 3.8513f, 3.9323f, 3.4719f,
    3.7498f, 4.283f, 3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.077f, 3.5075f, 3.27f,
    3.4767f, 2.8063f, 5.1125f, 3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f,
    3.8312f, 3.5605f, 4.3875f, 3.9624f, 4.0168f, 3.5643f, 4.055f, 5.5614f,
    4.2963f, 4.408f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.149f, 3.7662f,
    3.6746f, 3.4563f, 3.8161f)

  private val weights = new VaeWeights(ops, source)
  private val layers = new WanLayers(weights)

  /** A level: its residual blocks, its resampling 3×3 if it has one, and the
    * frames its shortcut shuffles over.
    */
  final private case class Level(
      blocks: Seq[ResidualBlock],
      resample: Option[Convolution],
      frames: Int
  )

  private def levels(side: String, kind: String): Seq[Level] =
    Iterator
      .from(0)
      .map(i => s"$side.$kind.$i.$kind")
      .takeWhile(p => source.has(s"$p.0.residual.0.gamma"))
      .map { p =>
        val blocks = Iterator
          .from(0)
          .takeWhile(j => source.has(s"$p.$j.residual.0.gamma"))
          .map(j => layers.residual(s"$p.$j"))
          .toSeq
        val resample = s"$p.${blocks.size}"
        Level(
          blocks,
          Option.when(source.has(s"$resample.resample.1.weight"))(
            weights.conv3x3(s"$resample.resample.1")
          ),
          if (source.has(s"$resample.time_conv.weight")) 2 else 1
        )
      }
      .toSeq

  if (
    !source.has("decoder.upsamples.0.upsamples.0.residual.0.gamma") ||
    !source.has("encoder.conv1.weight")
  )
    throw new FormatException(
      "no Qwen Image 2.1 VAE (decoder.upsamples.0.upsamples.0, encoder.conv1)"
    )

  private val encoderIn = weights.conv3x3("encoder.conv1")
  private val downs = levels("encoder", "downsamples")
  private val encoderMiddle = layers.middle("encoder")
  private val encoderNorm = layers.norm("encoder.head.0")
  private val encoderOut = weights.conv3x3("encoder.head.2")

  /** Latent channels: half of what the encoder's head gives (mean, log
    * variance).
    */
  val channels: Long = encoderOut.outChannels / 2
  private val quant = weights.conv1x1("conv1").take(channels)

  private val postQuant = weights.conv1x1("conv2")
  private val decoderIn = weights.conv3x3("decoder.conv1")
  private val decoderMiddle = layers.middle("decoder")
  private val ups = levels("decoder", "upsamples")
  private val decoderNorm = layers.norm("decoder.head.0")
  private val storedOut = weights.conv3x3("decoder.head.2")

  /** Pixels shuffled into the channels along each side: 2 for GRN's HBQ
    * tokenizer (the same network in RGB, 12 channels in and out, three
    * resampling levels), whose latents are also taken as they are, with no
    * statistics; 1 for Qwen Image 2.1's.
    */
  val pixelPatch: Int = if (storedOut.outChannels == 12) 2 else 1
  private val normalized = pixelPatch == 1
  require(
    !normalized || channels == LatentMean.length,
    s"$channels latent channels, not ${LatentMean.length}"
  )

  // the tokenizer's channels are `c × 4 + dx × 2 + dy`; `unpackPatches` takes
  // `c × 4 + dy × 2 + dx`
  private val decoderOut =
    if (pixelPatch == 1) storedOut
    else {
      val weight =
        weights.keep(
          ops.allocate(storedOut.weight.dtype, storedOut.weight.shape)
        )
      val bias =
        weights.keep(ops.allocate(storedOut.bias.dtype, storedOut.bias.shape))
      (0L until storedOut.outChannels).foreach { row =>
        val (c, dy, dx) = (row / 4, row % 4 / 2, row % 2)
        val stored = c * 4 + dx * 2 + dy
        ops.copy(storedOut.weight.rows(stored, 1), weight.rows(row, 1))
        ops.copy(storedOut.bias.rows(stored, 1), bias.rows(row, 1))
      }
      Convolution(weight, bias)
    }

  /** Image pixels per latent pixel along each side: 16. */
  val scale: Int = (1 << downs.count(_.resample.isDefined)) * pixelPatch

  /** Image channels: RGBA, or the tokenizer's RGB. */
  val imageChannels: Long =
    encoderIn.weight.shape.last / 9 / (pixelPatch * pixelPatch)

  // (z − mean) / std and back, as `modulate`'s x × (1 + scale) + shift
  private val normalizeScale = weights.floats(LatentStd.map(1 / _ - 1))
  private val normalizeShift =
    weights.floats(
      LatentMean.indices.map(i => -LatentMean(i) / LatentStd(i)).toArray
    )
  private val restoreScale = weights.floats(LatentStd.map(_ - 1))
  private val restoreShift = weights.floats(LatentMean)

  /** The latents of `image` (`[H, W, 4]`, values in [−1, 1], H and W multiples
    * of `scale`), normalized as the diffusion model takes them: `[H/16, W/16,
    * 64]`. The tensors it allocates are released but for the result's.
    */
  def encode(image: Tensor): Tensor = {
    val Seq(height, width, pixelChannels) = image.shape.dimensions
    require(
      pixelChannels == imageChannels && height % scale == 0 && width % scale == 0,
      s"encode: image ${image.shape}"
    )
    require(pixelPatch == 1, "the HBQ tokenizer's encoder is not run here")
    val run = new VaeRun(ops)
    import run.{dimensions, pixels}
    try {
      val copy = run.image(height, width, pixelChannels)
      ops.copy(image, copy)
      var x = run.conv3x3(copy, encoderIn)
      downs.foreach { level =>
        val (h, w, c) = dimensions(x)
        val out = level.blocks.last.conv2.outChannels
        val shortcut = level.resample match {
          case Some(_) =>
            val averaged = run.image(h / 2, w / 2, out)
            ops.averageDown(x, level.frames, averaged)
            averaged
          case None =>
            require(c == out, s"an encoder level from $c to $out channels")
            val same = run.image(h, w, c)
            ops.copy(x, same)
            same
        }
        level.blocks.foreach(block => x = run.residual(x, block))
        level.resample.foreach(conv => x = run.conv3x3(x, conv, stride = 2))
        ops.add(x, shortcut, x)
        run.free(shortcut)
      }
      x = run.middle(x, encoderMiddle)
      val normed = run.normSilu(x, encoderNorm)
      run.free(x)
      val latents = run.conv1x1(run.conv3x3(normed, encoderOut), quant)
      ops.modulate(
        pixels(latents),
        normalizeScale,
        normalizeShift,
        pixels(latents)
      )
      run.result(latents)
    } finally run.release()
  }

  /** The image of `latents` (`[h, w, 64]`, as `encode` gives them; the HBQ
    * tokenizer's as they are): `[16h, 16w, imageChannels]`, values in [−1, 1].
    * The tensors it allocates are released but for the result's.
    */
  def decode(latents: Tensor): Tensor = {
    val Seq(h, w, c) = latents.shape.dimensions
    require(c == channels, s"decode: latents ${latents.shape}")
    val run = new VaeRun(ops)
    import run.{dimensions, pixels}
    try {
      val z = run.image(h, w, c)
      if (normalized)
        ops.modulate(pixels(latents), restoreScale, restoreShift, pixels(z))
      else ops.copy(latents, z)
      var x = run.conv3x3(run.conv1x1(z, postQuant), decoderIn)
      x = run.middle(x, decoderMiddle)
      ups.foreach { level =>
        val shortcut = level.resample.map { _ =>
          val (height, width, _) = dimensions(x)
          val duplicated = run.image(
            2 * height,
            2 * width,
            level.blocks.last.conv2.outChannels
          )
          ops.duplicateUp(x, level.frames, duplicated)
          duplicated
        }
        level.blocks.foreach(block => x = run.residual(x, block))
        level.resample.foreach(conv => x = run.upsample(x, conv))
        shortcut.foreach { s =>
          ops.add(x, s, x)
          run.free(s)
        }
      }
      val normed = run.normSilu(x, decoderNorm)
      run.free(x)
      val out = run.conv3x3(normed, decoderOut)
      if (pixelPatch == 1) run.result(out)
      else {
        val (height, width, packed) = dimensions(out)
        val image = run.image(
          height * pixelPatch,
          width * pixelPatch,
          packed / (pixelPatch * pixelPatch)
        )
        ops.unpackPatches(pixels(out), height.toInt, pixelPatch, image)
        run.result(image)
      }
    } finally run.release()
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object QwenImage21Vae {

  def open(ops: Ops, path: Path): QwenImage21Vae = {
    val source = WeightSource.open(ops, path)
    try new QwenImage21Vae(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
