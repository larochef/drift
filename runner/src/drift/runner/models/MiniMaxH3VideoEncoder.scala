package drift.runner.models

import drift.runner.diffusion.TorchRandom
import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** How an encoder's posterior becomes latents: its mean (ComfyUI's), or a draw
  * from it under its own generator, rounded through F16 (diffusers'
  * `encode_vae_condition`, the released recipe for keyframes and references).
  */
enum Posterior {
  case Mean
  case Sample(seed: Long)
}

/** MiniMax H3's video VAE encoder (diffusers' `MiniMaxH3VideoEncoder3d` and
  * `AutoencoderKLMiniMaxH3._encode`, ComfyUI's `EncoderFCN3D`), from the
  * original names (`encoder.down.N.block.N`, `nin_shortcut`, `downsample`): a
  * causal 3-D CNN, channels-last frame by frame.
  *   - Every convolution is 3 × 3 × 3 over the frame and the two before it
  *     (zeros before the first): three 3 × 3 ones summed, each reflect-padded
  *     in space. The downsamplers halve space (reflect-padded by one below and
  *     to the right) and, at `temporalLevels`, time: output frame `o` reads
  *     frames `2o − 2`, `2o − 1` and `2o`. So 17 frames make 5 latent frames,
  *     one makes one.
  *   - Group norms of 32 groups over each frame alone, SiLU.
  *   - `quant_conv` gives the posterior's 2 × 24 moments (mean, log-variance).
  *   - The released recipe: 256-pixel tiles overlapping by at least 64, blended
  *     in latent space (the decoder's tiling); a clip of several frames in
  *     chunks of 17 (the last frame repeated up to a whole chunk), the last 3
  *     latent frames dropped: `17n + 5` frames give `5n + 2`; one frame goes
  *     through alone.
  *   - Pixels in [−1, 1] → ImageNet-normalized; latents normalized by the
  *     file's `latents_mean` and `latents_std`.
  * F16 weights (as the file stores them) and patches, F32 sums: BF16 ones move
  * the latents by a few percent (the reference runs in F32).
  */
final class MiniMaxH3VideoEncoder private (
    ops: Ops,
    source: WeightSource,
    tilePixels: Int,
    tileOverlap: Int,
    temporalLevels: Set[Int],
    groups: Int
) extends AutoCloseable {

  private val weights = new VaeWeights(ops, source)
  private val Epsilon = 1e-6f
  private val ClipLength = 17
  private val TokenDrop = 3

  /** Pixels per latent's side. */
  val Scale = 16

  /** A causal 3 × 3 × 3 convolution's slices, F16 as the file stores them (BF16
    * patches and sums move the latents by a few percent).
    */
  private def causal(prefix: String): Seq[Convolution] =
    weights.conv3x3x3(prefix, half = true)

  /** A 1 × 1 × 1 convolution as an F16 linear (F16 products whatever
    * `wideProducts` says; its inputs whole blocks of 32 channels).
    */
  private def pointwise(prefix: String): Convolution = {
    val Seq(out, in) = weights.shape(s"$prefix.weight").dimensions.take(2)
    Convolution(
      weights.f16(Shape.of(out, in), weights.values(s"$prefix.weight")),
      weights.floats(s"$prefix.bias")
    )
  }

  final private class Norm(prefix: String) {
    val weight: Tensor = weights.floats(s"$prefix.weight")
    val bias: Tensor = weights.floats(s"$prefix.bias")
  }

  final private class Residual(prefix: String) {
    val norm1 = new Norm(s"$prefix.norm1")
    val conv1: Seq[Convolution] = causal(s"$prefix.conv1")
    val norm2 = new Norm(s"$prefix.norm2")
    val conv2: Seq[Convolution] = causal(s"$prefix.conv2")
    val shortcut: Option[Convolution] =
      Option.when(weights.has(s"$prefix.nin_shortcut.weight"))(
        pointwise(s"$prefix.nin_shortcut")
      )
  }

  final private class Level(index: Int) {
    private val prefix = s"encoder.down.$index"
    val blocks: Seq[Residual] = Iterator
      .from(0)
      .takeWhile(i => weights.has(s"$prefix.block.$i.conv1.weight"))
      .map(i => new Residual(s"$prefix.block.$i"))
      .toSeq
    val downsample: Option[Seq[Convolution]] =
      Option.when(weights.has(s"$prefix.downsample.conv.weight"))(
        causal(s"$prefix.downsample.conv")
      )
    val temporal: Boolean = temporalLevels(index)
  }

  if (!weights.has("encoder.conv_in.weight"))
    throw new FormatException("no MiniMax H3 video encoder (encoder.conv_in)")
  private val convIn = causal("encoder.conv_in")
  private val levels = Iterator
    .from(0)
    .takeWhile(i => weights.has(s"encoder.down.$i.block.0.conv1.weight"))
    .map(new Level(_))
    .toSeq
  private val normOut = new Norm("encoder.norm_out")
  private val convOut = causal("encoder.conv_out")
  // quant_conv on the host, in F32: the moments take no rounding
  private val quantWeight = weights.values("quant_conv.weight")
  private val quantBias = weights.values("quant_conv.bias")
  private val moments = quantBias.length

  /** The latent channels (half the moments). */
  val latentChannels: Int = moments / 2
  private val latentMean = weights.values("latents_mean")
  private val latentStd = weights.values("latents_std")
  private val pixelNormalization = weights.f32(
    Shape.of(2, 3),
    Array.tabulate(6) { i =>
      val c = i % 3
      // (x + 1) / 2 → (· − mean) / std as x × scale + shift
      if (i < 3) 0.5f / MiniMaxH3Vae.PixelStd(c) - 1
      else (0.5f - MiniMaxH3Vae.PixelMean(c)) / MiniMaxH3Vae.PixelStd(c)
    }
  )

  // ---- the network, on frames [h, w, C] ------------------------------------------

  private def sizes(x: Tensor): (Long, Long, Long) = {
    val Seq(h, w, c) = x.shape.dimensions
    (h, w, c)
  }

  /** The causal convolution over `frames` (they stay the caller's), `time` and
    * `space` its strides.
    */
  private def causal(
      frames: Seq[Tensor],
      slices: Seq[Convolution],
      time: Int,
      space: Int
  ): Seq[Tensor] = {
    val (h, w, _) = sizes(frames.head)
    val outChannels = slices.last.outChannels
    val count = (frames.size - 1) / time + 1
    (0 until count).map { o =>
      val out =
        ops.allocate(DType.F32, Shape.of(h / space, w / space, outChannels))
      val newest = slices(2)
      ops.conv3x3(
        frames(time * o),
        newest.weight,
        newest.bias,
        out,
        space,
        reflect = true
      )
      Seq(0, 1).filter(j => time * o - 2 + j >= 0).foreach { j =>
        val part = ops.allocate(DType.F32, out.shape)
        try {
          ops.conv3x3(
            frames(time * o - 2 + j),
            slices(j).weight,
            slices(j).bias,
            part,
            space,
            reflect = true
          )
          ops.add(out, part, out)
        } finally ops.release(part)
      }
      out
    }
  }

  private def normSilu(x: Tensor, norm: Norm): Tensor = {
    val (h, w, c) = sizes(x)
    val out = ops.allocate(DType.F32, x.shape)
    ops.groupNorm(
      x.view(h * w, c),
      groups,
      norm.weight,
      norm.bias,
      Epsilon,
      out.view(h * w, c)
    )
    ops.activation(Activation.Silu, out, out)
    out
  }

  private def linear(x: Tensor, conv: Convolution): Tensor = {
    val (h, w, c) = sizes(x)
    val out = ops.allocate(DType.F32, Shape.of(h, w, conv.outChannels))
    ops.linear(x.view(h * w, c), conv.weight, out.view(h * w, conv.outChannels))
    ops.addRow(
      out.view(h * w, conv.outChannels),
      conv.bias,
      out.view(h * w, conv.outChannels)
    )
    out
  }

  /** `frames` through `stage`, released. */
  private def through(
      frames: Seq[Tensor],
      stage: Seq[Tensor] => Seq[Tensor]
  ): Seq[Tensor] =
    try stage(frames)
    finally frames.foreach(ops.release)

  private def residual(frames: Seq[Tensor], block: Residual): Seq[Tensor] = {
    val first = through(
      frames.map(normSilu(_, block.norm1)),
      causal(_, block.conv1, 1, 1)
    )
    val second = through(
      through(first, _.map(normSilu(_, block.norm2))),
      causal(_, block.conv2, 1, 1)
    )
    second.zip(frames).foreach { (y, x) =>
      block.shortcut match {
        case Some(conv) =>
          val shortcut = linear(x, conv)
          ops.add(y, shortcut, y)
          ops.release(shortcut)
        case None => ops.add(y, x, y)
      }
    }
    frames.foreach(ops.release)
    second
  }

  /** One tile of one clip: frames `[h, w, 3]` normalized (released) → moments
    * `[frames', h / 16, w / 16, 2C]` on the host.
    */
  private def encodeTile(frames: Seq[Tensor]): Array[Float] = {
    var x = through(frames, causal(_, convIn, 1, 1))
    levels.foreach { level =>
      level.blocks.foreach(block => x = residual(x, block))
      level.downsample.foreach(slices =>
        x = through(x, causal(_, slices, if (level.temporal) 2 else 1, 2))
      )
    }
    x =
      through(through(x, _.map(normSilu(_, normOut))), causal(_, convOut, 1, 1))
    val features =
      try x.flatMap(ops.toFloats).toArray
      finally x.foreach(ops.release)
    Array.tabulate(features.length) { i =>
      val (voxel, o) = (i / moments, i % moments)
      var sum = quantBias(o).toDouble
      var j = 0
      while (j < moments) {
        sum += quantWeight(o * moments + j) * features(voxel * moments + j)
        j += 1
      }
      sum.toFloat
    }
  }

  /** One clip (`frames` of `[height, width, 3]` in [−1, 1]) spatially tiled and
    * stitched: moments `[frames', height / 16, width / 16, 2C]`.
    */
  private def encodeClip(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int
  ): Array[Float] = {
    val (rowStarts, tileHeight, rowOverlaps) =
      VaeTiles.split(height, tilePixels, tileOverlap, 16)
    val (columnStarts, tileWidth, columnOverlaps) =
      VaeTiles.split(width, tilePixels, tileOverlap, 16)
    val tiles = rowStarts.map { top =>
      columnStarts.map { left =>
        encodeTile(frames.map { pixels =>
          val tile = new Array[Float](tileHeight * tileWidth * 3)
          (0 until tileHeight).foreach(y =>
            System.arraycopy(
              pixels,
              ((top + y) * width + left) * 3,
              tile,
              y * tileWidth * 3,
              tileWidth * 3
            )
          )
          val uploaded =
            ops.fromFloats(Shape.of(tileHeight, tileWidth, 3), tile)
          val normalized = ops.allocate(DType.F32, uploaded.shape)
          try
            ops.modulate(
              uploaded.view(tileHeight.toLong * tileWidth, 3),
              pixelNormalization.rows(0, 1).view(3),
              pixelNormalization.rows(1, 1).view(3),
              normalized.view(tileHeight.toLong * tileWidth, 3)
            )
          finally ops.release(uploaded)
          normalized
        })
      }
    }
    val latentFrames =
      Iterator
        .iterate(frames.size)(n => (n - 1) / 2 + 1)
        .drop(levels.count(_.temporal))
        .next()
    VaeTiles.stitch(
      tiles,
      latentFrames,
      tileHeight / Scale,
      tileWidth / Scale,
      rowOverlaps.map(_ / Scale),
      columnOverlaps.map(_ / Scale),
      height / Scale,
      width / Scale,
      2 * latentChannels
    )
  }

  /** The posterior's moments of `frames` (each `[height, width, 3]` in [−1, 1],
    * sides multiples of 16): `[T, height / 16, width / 16, 2C]` (mean then
    * log-variance), T = 1 for one frame, else `5n + 2` for `17n + 5` frames.
    */
  def moments(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int
  ): Array[Float] = {
    require(
      frames.nonEmpty && height % Scale == 0 && width % Scale == 0,
      s"encode: ${frames.size} frames of $width × $height"
    )
    if (frames.size == 1) encodeClip(frames, height, width)
    else {
      val padded = frames ++ Seq.fill(
        (ClipLength - frames.size % ClipLength) % ClipLength
      )(frames.last)
      val all = padded
        .grouped(ClipLength)
        .flatMap(clip => encodeClip(clip, height, width))
        .toArray
      val voxel = (height / Scale) * (width / Scale) * 2 * latentChannels
      java.util.Arrays.copyOf(all, all.length - TokenDrop * voxel)
    }
  }

  /** Normalized latents of `frames` (as `moments` takes them), `[T, height /
    * 16, width / 16, C]`, by `posterior`.
    */
  def encode(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int,
      posterior: Posterior
  ): Array[Float] = {
    val moments = this.moments(frames, height, width)
    val c = latentChannels
    val voxels = moments.length / (2 * c)
    // the draw is `randn(mean.shape)` of `[1, C, T, h, w]`: channel-major
    val noise = posterior match {
      case Posterior.Mean         => None
      case Posterior.Sample(seed) =>
        Some(new TorchRandom(seed).normal(voxels * c))
    }
    Array.tabulate(voxels * c) { i =>
      val (voxel, channel) = (i / c, i % c)
      val mean = moments(voxel * 2 * c + channel)
      val latent = noise match {
        case None        => mean
        case Some(drawn) =>
          val logVariance =
            math.max(-30f, math.min(20f, moments(voxel * 2 * c + c + channel)))
          val sample = mean + math.exp(0.5 * logVariance).toFloat * drawn(
            channel * voxels + voxel
          )
          MiniMaxH3VideoEncoder.roundedToHalf(sample)
      }
      (latent - latentMean(channel)) / latentStd(channel)
    }
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object MiniMaxH3VideoEncoder {

  /** `value` rounded to the nearest F16 (to even), back as a float. */
  def roundedToHalf(value: Float): Float =
    java.lang.Float.float16ToFloat(java.lang.Float.floatToFloat16(value))

  /** The encoder of the video VAE `path` (its `encoder.*`, `quant_conv` and
    * latent statistics), tiled as released unless told otherwise; time halves
    * at `temporalLevels` (the released config's second and third levels: the
    * file does not record it), its group norms of `groups`.
    */
  def open(
      ops: Ops,
      path: Path,
      tilePixels: Int = 256,
      tileOverlap: Int = 64,
      temporalLevels: Set[Int] = Set(1, 2),
      groups: Int = 32
  ): MiniMaxH3VideoEncoder = {
    val source = WeightSource.open(ops, path)
    try
      new MiniMaxH3VideoEncoder(
        ops,
        source,
        tilePixels,
        tileOverlap,
        temporalLevels,
        groups
      )
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
