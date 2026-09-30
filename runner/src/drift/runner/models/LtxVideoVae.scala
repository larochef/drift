package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** LTX 2's convolutional video VAE decoder (the official "conv" file, LTX's
  * `CausalVideoAutoencoder` with a non-causal decoder; diffusers'
  * `LTX2VideoResnetBlock3d` and `LTX2VideoUpsampler3d`), channels-last frame by
  * frame, its structure read off the weights: `decoder.up_blocks.N` are
  * residual stages (`res_blocks`: pixel norm, SiLU, 3×3×3 convolution, twice,
  * plus the input) or depth-to-space upsamplers (`conv`: a 3×3×3 convolution to
  * `stride × out` channels, rearranged to 2× frames, 2× rows and columns or
  * both, the first new frame dropped), then a pixel norm, SiLU and the 3×3×3
  * convolution to 4 × 4 patches of RGB. The convolutions see the first and last
  * frames repeated past the ends (non-causal); the upsamplers' strides follow
  * from their channels and the next stage's.
  */
final class LtxVideoVae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  private val weights = new VaeWeights(ops, source)
  private def dimensions(name: String) =
    source.shape(name).dimensions.map(_.toInt)

  private val mean = weights.values("per_channel_statistics.mean-of-means")
  private val std = weights.values("per_channel_statistics.std-of-means")
  val latentChannels: Int = mean.length
  private val Patch = 4

  private enum Stage {
    case Residual(blocks: Seq[(Seq[Convolution], Seq[Convolution])])

    /** `time` × 2 in frames, `space` × 2 in rows and columns. */
    case Up(
        conv: Seq[Convolution],
        time: Boolean,
        space: Boolean,
        channels: Int
    )
  }

  private def residuals(prefix: String) =
    Iterator
      .from(0)
      .takeWhile(i => source.has(s"$prefix.res_blocks.$i.conv1.conv.weight"))
      .map(i =>
        (
          weights.conv3x3x3(s"$prefix.res_blocks.$i.conv1.conv"),
          weights.conv3x3x3(s"$prefix.res_blocks.$i.conv2.conv")
        )
      )
      .toSeq

  private val stages: Seq[Stage] = {
    val count = Iterator
      .from(0)
      .takeWhile(i =>
        source.has(s"decoder.up_blocks.$i.res_blocks.0.conv1.conv.weight") ||
          source.has(s"decoder.up_blocks.$i.conv.conv.weight")
      )
      .size
    (0 until count).map { i =>
      val prefix = s"decoder.up_blocks.$i"
      if (source.has(s"$prefix.res_blocks.0.conv1.conv.weight"))
        Stage.Residual(residuals(prefix))
      else {
        val outputs = dimensions(s"$prefix.conv.conv.weight").head
        // the channels after it: the next stage's, else the output convolution's
        val next =
          if (
            source.has(
              s"decoder.up_blocks.${i + 1}.res_blocks.0.conv1.conv.weight"
            )
          )
            dimensions(
              s"decoder.up_blocks.${i + 1}.res_blocks.0.conv1.conv.weight"
            )(1)
          else dimensions("decoder.conv_out.conv.weight")(1)
        val (time, space) = outputs / next match {
          case 8     => (true, true)
          case 4     => (false, true)
          case 2     => (true, false)
          case other =>
            throw new FormatException(s"$prefix: an upsampler of stride $other")
        }
        val (s0, s12) = (if (time) 2 else 1, if (space) 4 else 1)
        // stored channel ((c × s0 + t) × s1 + y) × s2 + x → (t, c, y, x): each
        // new frame's channels contiguous, then packed 2 × 2 as unpackPatches takes them
        val rows: Int => Int = r => {
          val (t, rest) = (r / (next * s12), r % (next * s12))
          val (ch, sub) = (rest / s12, rest % s12)
          (ch * s0 + t) * s12 + sub
        }
        Stage.Up(
          weights.conv3x3x3(s"$prefix.conv.conv", rows),
          time,
          space,
          next
        )
      }
    }
  }
  private val input = weights.conv3x3x3("decoder.conv_in.conv")
  // stored channel c × 16 + x × 4 + y → c × 16 + y × 4 + x (unpackPatches')
  private val output = weights.conv3x3x3(
    "decoder.conv_out.conv",
    r => (r / 16) * 16 + (r % 4) * 4 + (r / 4) % 4
  )

  private def sizes(x: Tensor) = {
    val Seq(h, w, c) = x.shape.dimensions
    (h, w, c)
  }

  private def pixels(x: Tensor) = {
    val (h, w, c) = sizes(x)
    x.view(h * w, c)
  }

  private val onesFor = scala.collection.mutable.Map.empty[Long, Tensor]
  private def ones(channels: Long) =
    onesFor.getOrElseUpdate(
      channels,
      weights.floats(Array.fill(channels.toInt)(1f))
    )

  /** Pixel norm (RMS over the channels, ε 10⁻⁸) then SiLU, into a new frame. */
  private def normSilu(x: Tensor): Tensor = {
    val (h, w, c) = sizes(x)
    val out = ops.allocate(DType.F32, Shape.of(h, w, c))
    ops.rmsNorm(pixels(x), ones(c), 1e-8f, 0f, pixels(out))
    ops.activation(Activation.Silu, out, out)
    out
  }

  /** A non-causal 3×3×3 convolution over `frames` (the ends repeated). */
  private def conv(frames: Seq[Tensor], slices: Seq[Convolution]): Seq[Tensor] =
    frames.indices.map { t =>
      val (h, w, _) = sizes(frames(t))
      val channels = slices.last.outChannels
      val out = ops.allocate(DType.F32, Shape.of(h, w, channels))
      val part = ops.allocate(DType.F32, Shape.of(h, w, channels))
      try
        slices.zipWithIndex.foreach { (slice, tau) =>
          val source =
            frames(math.min(math.max(t + tau - 1, 0), frames.size - 1))
          if (tau == 0) ops.conv3x3(source, slice.weight, slice.bias, out)
          else {
            ops.conv3x3(source, slice.weight, slice.bias, part)
            ops.add(out, part, out)
          }
        }
      finally ops.release(part)
      out
    }

  private def release(frames: Seq[Tensor]): Unit = frames.foreach(ops.release)

  /** Decodes normalized `latents` (`[h, w, channels]` per latent frame) into
    * frames `[32h, 32w, 3]` in [−1, 1] (unclamped): `8 (T − 1) + 1` of them,
    * handed to `emit` in order and released after.
    */
  def decode(latents: Seq[Tensor], emit: Tensor => Unit): Unit = {
    val shift = weights.floats(mean)
    val scale = weights.floats(std.map(_ - 1f))
    var frames = latents.map { latent =>
      val (h, w, c) = sizes(latent)
      require(c == latentChannels, s"decode: latents ${latent.shape}")
      val z = ops.allocate(DType.F32, Shape.of(h, w, c))
      ops.modulate(pixels(latent), scale, shift, pixels(z))
      z
    }
    def replace(next: Seq[Tensor]): Unit = { release(frames); frames = next }
    replace(conv(frames, input))
    stages.foreach {
      case Stage.Residual(blocks) =>
        blocks.foreach { (first, second) =>
          val normed = frames.map(normSilu)
          val inner = conv(normed, first)
          release(normed)
          val normedInner = inner.map(normSilu)
          release(inner)
          val out = conv(normedInner, second)
          release(normedInner)
          out.zip(frames).foreach((o, x) => ops.add(o, x, o))
          replace(out)
        }
      case Stage.Up(slices, time, space, channels) =>
        val convolved = conv(frames, slices)
        val (s0, s12) = (if (time) 2 else 1, if (space) 4 else 1)
        val expanded = convolved.flatMap { x =>
          val (h, w, _) = sizes(x)
          (0 until s0).map { t =>
            val part =
              ops.allocate(DType.F32, Shape.of(h * w, channels.toLong * s12))
            if (s0 == 1) ops.copy(pixels(x), part)
            else {
              val other = ops.allocate(DType.F32, part.shape)
              if (t == 0) ops.splitHalves(pixels(x), part, other)
              else ops.splitHalves(pixels(x), other, part)
              ops.release(other)
            }
            if (!space) part.view(h, w, channels.toLong)
            else {
              val image =
                ops.allocate(DType.F32, Shape.of(2 * h, 2 * w, channels.toLong))
              ops.unpackPatches(part, h.toInt, 2, image)
              ops.release(part)
              image
            }
          }
        }
        release(convolved)
        // the first new frame is dropped
        val kept = if (time) { ops.release(expanded.head); expanded.tail }
        else expanded
        replace(kept)
    }
    val normed = frames.map(normSilu)
    replace(Nil)
    val patches = conv(normed, output)
    release(normed)
    try
      patches.foreach { x =>
        val (h, w, _) = sizes(x)
        val rgb = ops.allocate(DType.F32, Shape.of(h * Patch, w * Patch, 3))
        try {
          ops.unpackPatches(pixels(x), h.toInt, Patch, rgb)
          emit(rgb)
        } finally ops.release(rgb)
      }
    finally release(patches)
  }

  /** How many frames `decode` gives for `frames` latent frames. */
  def frameCount(frames: Int): Int =
    stages.foldLeft(frames) {
      case (n, Stage.Up(_, true, _, _)) => 2 * n - 1
      case (n, _)                       => n
    }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object LtxVideoVae {

  def open(ops: Ops, path: Path): LtxVideoVae = {
    val source = WeightSource.open(ops, path)
    try {
      if (
        !source.has("decoder.conv_in.conv.weight") || !source.has(
          "per_channel_statistics.mean-of-means"
        )
      )
        throw new FormatException(s"$path is no LTX 2 video VAE")
      new LtxVideoVae(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
