package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** LTX 2's convolutional video VAE (the official "conv" file, LTX's
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
  *
  * The encoder (diffusers' `LTX2VideoEncoder3d`, loaded on first use) is causal
  * (the first frame repeated before it): 4 × 4 patches of RGB, a 3×3×3
  * convolution, then `encoder.down_blocks.N`, residual stages as the decoder's
  * or space-to-depth downsamplers (`conv`: a 3×3×3 convolution to `out /
  * stride` channels, rearranged ×2 in time, space or both, plus the input
  * rearranged alike and averaged over groups of channels; in time, the first
  * frame repeated before the pairs), a pixel norm, SiLU and the convolution to
  * the latent means (and a log-variance, dropped: the mode).
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

  /** A 3×3×3 convolution over `frames`: non-causal (the ends repeated), or
    * `causal` (the first frame repeated twice before it).
    */
  private def conv(
      frames: Seq[Tensor],
      slices: Seq[Convolution],
      causal: Boolean = false
  ): Seq[Tensor] =
    frames.indices.map { t =>
      val (h, w, _) = sizes(frames(t))
      val channels = slices.last.outChannels
      val out = ops.allocate(DType.F32, Shape.of(h, w, channels))
      val part = ops.allocate(DType.F32, Shape.of(h, w, channels))
      try
        slices.zipWithIndex.foreach { (slice, tau) =>
          val source =
            if (causal) frames(math.max(t + tau - 2, 0))
            else frames(math.min(math.max(t + tau - 1, 0), frames.size - 1))
          if (tau == 0) ops.conv3x3(source, slice.weight, slice.bias, out)
          else {
            ops.conv3x3(source, slice.weight, slice.bias, part)
            ops.add(out, part, out)
          }
        }
      finally ops.release(part)
      counted(h.toDouble * w * pixelCost(slices))
      out
    }

  /** What a convolution costs a pixel, measured on gfx1151 (`bugs/52`): its
    * product, and gathering its inputs, which is as much as a product over a
    * thousand outputs — most of the time where the channels are few and the
    * frames large.
    */
  private def pixelCost(slices: Seq[Convolution]): Double =
    slices
      .map(slice =>
        slice.weight.shape.last.toDouble * (slice.outChannels + 1000)
      )
      .sum

  /** Told what each frame's convolution cost (`pixelCost` times its pixels):
    * what `decode` counts its progress in.
    */
  private var counted: Double => Unit = _ => ()

  /** What decoding `frames` latent frames of `height × width` costs, in
    * `counted`'s unit: every convolution `decode` runs.
    */
  private def decodeCost(frames: Int, height: Long, width: Long): Double = {
    var (t, h, w) = (frames.toLong, height, width)
    def cost(slices: Seq[Convolution]) =
      t.toDouble * h * w * pixelCost(slices)
    var total = cost(input)
    stages.foreach {
      case Stage.Residual(blocks) =>
        blocks.foreach((first, second) => total += cost(first) + cost(second))
      case Stage.Up(slices, time, space, _) =>
        total += cost(slices)
        if (time) t = 2 * t - 1
        if (space) { h *= 2; w *= 2 }
    }
    total + cost(output)
  }

  private def release(frames: Seq[Tensor]): Unit = frames.foreach(ops.release)

  /** `frames` through residual `blocks` (pixel norm, SiLU, convolution, twice,
    * plus the input), into new frames; `frames` are left to the caller.
    */
  private def residual(
      frames: Seq[Tensor],
      blocks: Seq[(Seq[Convolution], Seq[Convolution])],
      causal: Boolean
  ): Seq[Tensor] =
    blocks.foldLeft(frames) { case (current, (first, second)) =>
      val normed = current.map(normSilu)
      val inner = conv(normed, first, causal)
      release(normed)
      val normedInner = inner.map(normSilu)
      release(inner)
      val out = conv(normedInner, second, causal)
      release(normedInner)
      out.zip(current).foreach((o, x) => ops.add(o, x, o))
      if (current ne frames) release(current)
      out
    }

  /** Decodes normalized `latents` (`[h, w, channels]` per latent frame) into
    * frames `[32h, 32w, 3]` in [−1, 1] (unclamped): `8 (T − 1) + 1` of them,
    * handed to `emit` in order and released after. `progress` is told the
    * fraction of the convolutions done, frame by frame.
    */
  def decode(
      latents: Seq[Tensor],
      emit: Tensor => Unit,
      progress: Double => Unit
  ): Unit = {
    val (height, width, _) = sizes(latents.head)
    val total = decodeCost(latents.size, height, width)
    var done = 0.0
    counted = cost => { done += cost; progress(done / total) }
    try decoded(latents, emit)
    finally counted = _ => ()
  }

  private def decoded(latents: Seq[Tensor], emit: Tensor => Unit): Unit = {
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
        replace(residual(frames, blocks, causal = false))
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

  // ---- the encoder ----------------------------------------------------------------

  private enum Down {
    case Residual(blocks: Seq[(Seq[Convolution], Seq[Convolution])])

    /** A space-to-depth downsampler: `time` frames (1 or 2) and `space` ×
      * `space` pixels (1 or 2) into one, `channels` out, the residual the mean
      * of `group` consecutive channels of the input rearranged alike.
      */
    case Compress(
        conv: Seq[Convolution],
        time: Int,
        space: Int,
        channels: Int,
        group: Int
    )
  }

  final private class Encoder {
    private def at(i: Int) = s"encoder.down_blocks.$i"
    private def isResidual(i: Int) =
      source.has(s"${at(i)}.res_blocks.0.conv1.conv.weight")
    private def isCompress(i: Int) = source.has(s"${at(i)}.conv.conv.weight")
    private val count =
      Iterator.from(0).takeWhile(i => isResidual(i) || isCompress(i)).size

    /** The channels stage `i` takes in: its first convolution's inputs. */
    private def inputsOf(i: Int) =
      if (i == count) dimensions("encoder.conv_out.conv.weight")(1)
      else if (isResidual(i))
        dimensions(s"${at(i)}.res_blocks.0.conv1.conv.weight")(1)
      else dimensions(s"${at(i)}.conv.conv.weight")(1)

    val stages: Seq[Down] = (0 until count).map { i =>
      if (isResidual(i)) Down.Residual(residuals(at(i)))
      else {
        val Seq(outputs, inputs) =
          dimensions(s"${at(i)}.conv.conv.weight").take(2)
        val next = inputsOf(i + 1)
        val (time, space) = next / outputs match {
          case 8     => (2, 2)
          case 4     => (1, 2)
          case 2     => (2, 1)
          case other =>
            throw new FormatException(
              s"${at(i)}: a downsampler of stride $other"
            )
        }
        Down.Compress(
          weights.conv3x3x3(s"${at(i)}.conv.conv"),
          time,
          space,
          next,
          inputs * time * space * space / next
        )
      }
    }
    // packed channel c × 16 + y × 4 + x (packPatches') → stored c × 16 + x × 4 + y
    val input: Seq[Convolution] = weights.conv3x3x3(
      "encoder.conv_in.conv",
      inputs = i => (i / 16) * 16 + (i % 4) * 4 + (i / 4) % 4
    )
    val output: Seq[Convolution] =
      weights.conv3x3x3("encoder.conv_out.conv").map(_.take(latentChannels))
    val scale: Tensor = weights.floats(std.map(1f / _ - 1f))
    val shift: Tensor =
      weights.floats(mean.indices.map(i => -mean(i) / std(i)).toArray)
  }

  private lazy val encoder = new Encoder

  /** One space-to-depth downsampler over `frames` (released here). */
  private def compress(
      frames: Seq[Tensor],
      stage: Down.Compress
  ): Seq[Tensor] = {
    val Down.Compress(slices, time, space, channels, group) = stage
    // in time, the first frame repeated before the pairs
    val padded = if (time == 2) frames.head +: frames else frames
    val convolved = conv(padded, slices, causal = true)
    val (h, w, inputs) = sizes(frames.head)
    val convChannels = slices.last.outChannels.toInt
    val (oh, ow) = (h.toInt / space, w.toInt / space)
    val out = (0 until padded.size / time).map { j =>
      val sources = (0 until time).map(t => ops.toFloats(padded(j * time + t)))
      val results =
        (0 until time).map(t => ops.toFloats(convolved(j * time + t)))
      val values = new Array[Float](oh * ow * channels)
      val perPixel = time * space * space
      // channel ((c × time + t) × space + dy) × space + dx of (y, x) is
      // channel c of pixel (y × space + dy, x × space + dx) of frame t
      def pick(
          frames: IndexedSeq[Array[Float]],
          width: Int,
          k: Int,
          y: Int,
          x: Int
      ) = {
        val (c, rest) = (k / perPixel, k % perPixel)
        val (t, dy, dx) =
          (rest / (space * space), rest / space % space, rest % space)
        frames(t)(((y * space + dy) * w.toInt + x * space + dx) * width + c)
      }
      for {
        y <- 0 until oh
        x <- 0 until ow
        o <- 0 until channels
      } {
        var sum = 0.0
        (0 until group).foreach(g =>
          sum += pick(sources, inputs.toInt, o * group + g, y, x)
        )
        values((y * ow + x) * channels + o) =
          pick(results, convChannels, o, y, x) + (sum / group).toFloat
      }
      ops.fromFloats(Shape.of(oh, ow, channels), values)
    }
    release(convolved)
    release(frames)
    out
  }

  /** Encodes `frames` (`[H, W, 3]` in [−1, 1], `8n + 1` of them, sides
    * multiples of 32) into normalized latents `[H / 32, W / 32, channels]`,
    * `n + 1` of them, the caller's to release; `frames` are left to the caller.
    */
  def encode(frames: Seq[Tensor]): Seq[Tensor] = {
    val e = encoder
    var current = frames.map { rgb =>
      val (h, w, _) = sizes(rgb)
      val packed =
        ops.allocate(
          DType.F32,
          Shape.of(h / Patch, w / Patch, 3L * Patch * Patch)
        )
      ops.packPatches(rgb, Patch, pixels(packed))
      packed
    }
    def replace(next: Seq[Tensor]): Unit = { release(current); current = next }
    replace(conv(current, e.input, causal = true))
    e.stages.foreach {
      case Down.Residual(blocks) =>
        replace(residual(current, blocks, causal = true))
      case stage: Down.Compress =>
        current = compress(current, stage)
    }
    val normed = current.map(normSilu)
    replace(Nil)
    val latents = conv(normed, e.output, causal = true)
    release(normed)
    latents.foreach(z => ops.modulate(pixels(z), e.scale, e.shift, pixels(z)))
    latents
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
