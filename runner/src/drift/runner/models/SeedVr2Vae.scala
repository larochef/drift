package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** SeedVR2's video VAE (the reference's `VideoAutoencoderKLWrapper`,
  * `s8_c16_t4_inflation_sd3`: an SD3 VAE inflated to causal 3-D), both ways,
  * channels-last frame by frame. Its diffusers names are read as the LDM ones
  * `WeightNames` gives every diffusers `AutoencoderKL`. Group norms of 32
  * groups and the middle's attention work on each frame alone; time only passes
  * through the causal convolutions, which are streams:
  *   - a 3×3×3 one keeps the last two frames it was given, **the first frame
  *     standing for the ones before it**, and a frame's output is its three
  *     temporal slices over the frame and those two;
  *   - the encoder's temporal downsamplers (3×3×3, stride 2 every way) turn the
  *     first frame alone into one, then each pair of frames and the frame
  *     before them into one;
  *   - the decoder's upsamplers are a 1×1 to 4 or 8 times the channels,
  *     shuffled into 2×2 pixels of one frame or two (the stream's first frame
  *     gives one), then a 3×3×3.
  * `1 + 4n` frames make `1 + n` latent frames of 16 channels at 1/8. The
  * latents here are the VAE's own: the transformer's are `LatentScale` times
  * them.
  */
final class SeedVr2Vae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  private val weights = new VaeWeights(ops, source)

  private def norm(prefix: String) =
    ChannelNorm.Group(
      weights.floats(s"$prefix.weight"),
      weights.floats(s"$prefix.bias")
    )

  /** A causal 3×3×3 convolution (its first `keep` output channels) and the last
    * two frames it was given.
    */
  final private class Causal(prefix: String, keep: Option[Long] = None) {
    val slices: Seq[Convolution] = {
      val all = weights.conv3x3x3(prefix)
      keep.fold(all)(count => all.map(_.take(count)))
    }
    // the three slices as one, for a frame that stands for its predecessors
    val still: Convolution = {
      val all = weights.conv3x3x3Summed(prefix)
      keep.fold(all)(all.take)
    }
    var history: List[Tensor] = Nil // newest first
    def outChannels: Long = slices.last.outChannels
    def reset(): Unit = { history.foreach(ops.release); history = Nil }
  }

  final private class Residual(prefix: String) {
    val norm1: ChannelNorm = norm(s"$prefix.norm1")
    val conv1 = new Causal(s"$prefix.conv1")
    val norm2: ChannelNorm = norm(s"$prefix.norm2")
    val conv2 = new Causal(s"$prefix.conv2")
    val shortcut: Option[Convolution] =
      Option.when(weights.has(s"$prefix.nin_shortcut.weight"))(
        weights.conv1x1(s"$prefix.nin_shortcut")
      )
    def reset(): Unit = { conv1.reset(); conv2.reset() }
  }

  final private class Middle(prefix: String) {
    val first = new Residual(s"$prefix.block_1")
    val attention: PixelAttention = PixelAttention(
      norm(s"$prefix.attn_1.norm"),
      weights.conv1x1(s"$prefix.attn_1.q"),
      weights.conv1x1(s"$prefix.attn_1.k"),
      weights.conv1x1(s"$prefix.attn_1.v"),
      weights.conv1x1(s"$prefix.attn_1.proj_out")
    )
    val second = new Residual(s"$prefix.block_2")
    def reset(): Unit = { first.reset(); second.reset() }
  }

  /** A stride-2 downsampler: spatial alone (one 3×3) or temporal too (three
    * slices, and the last frame it was given).
    */
  final private class Down(prefix: String) {
    val slices: Seq[Convolution] =
      if (weights.shape(s"$prefix.weight").dimensions(2) == 3)
        weights.conv3x3x3(prefix)
      else Seq(weights.conv3x3(prefix))
    def temporal: Boolean = slices.size == 3
    val still: Option[Convolution] =
      Option.when(temporal)(weights.conv3x3x3Summed(prefix))
    var previous: Option[Tensor] = None
    def reset(): Unit = { previous.foreach(ops.release); previous = None }
  }

  /** An upsampler: its 1×1 as one linear per output frame (rows ordered as
    * `unpackPatches` takes them), then a causal convolution.
    */
  final private class Up(prefix: String) {
    val conv = new Causal(s"$prefix.conv")
    val shuffles: Seq[Convolution] = {
      val name = s"$prefix.upscale_conv"
      val dimensions = weights.shape(s"$name.weight").dimensions.map(_.toInt)
      val (out, channels) = (dimensions(0), dimensions(1))
      val frames = out / channels / 4
      val (weight, bias) =
        (weights.values(s"$name.weight"), weights.values(s"$name.bias"))
      // stored `(x y z c)`: row ((2 x + y) frames + z) channels + c
      def stored(row: Int, z: Int) = {
        val (c, pixel) = (row / 4, row % 4)
        (pixel * frames + z) * channels + c
      }
      (0 until frames).map { z =>
        Convolution(
          weights.bf16(
            Shape.of(4L * channels, channels),
            Array.tabulate(4 * channels * channels)(i =>
              weight(stored(i / channels, z) * channels + i % channels)
            )
          ),
          weights.floats(
            Array.tabulate(4 * channels)(row => bias(stored(row, z)))
          )
        )
      }
    }
    var started = false
    def reset(): Unit = { conv.reset(); started = false }
  }

  private def blocks(prefix: String): Seq[Residual] =
    Iterator
      .from(0)
      .map(i => s"$prefix.block.$i")
      .takeWhile(p => weights.has(s"$p.conv1.weight"))
      .map(new Residual(_))
      .toSeq

  private def levels(prefix: String): Seq[String] =
    Iterator
      .from(0)
      .map(i => s"$prefix.$i")
      .takeWhile(p => weights.has(s"$p.block.0.conv1.weight"))
      .toSeq

  final private class Decoder {
    val in = new Causal("decoder.conv_in")
    val middle = new Middle("decoder.mid")
    // the deepest level first
    val ups: Seq[(Seq[Residual], Option[Up])] =
      levels("decoder.up").reverse.map { p =>
        (
          blocks(p),
          Option.when(weights.has(s"$p.upsample.conv.weight"))(
            new Up(s"$p.upsample")
          )
        )
      }
    val normOut: ChannelNorm = norm("decoder.norm_out")
    val out = new Causal("decoder.conv_out")
    def reset(): Unit = {
      in.reset(); middle.reset(); out.reset()
      ups.foreach { (residuals, up) =>
        residuals.foreach(_.reset()); up.foreach(_.reset())
      }
    }
  }

  final private class Encoder {
    val in = new Causal("encoder.conv_in")
    val downs: Seq[(Seq[Residual], Option[Down])] = levels("encoder.down").map {
      p =>
        (
          blocks(p),
          Option.when(weights.has(s"$p.downsample.conv.weight"))(
            new Down(s"$p.downsample.conv")
          )
        )
    }
    val middle = new Middle("encoder.mid")
    val normOut: ChannelNorm = norm("encoder.norm_out")
    // mean and log-variance: the mean alone, the latent being the mode
    val out = new Causal("encoder.conv_out", Some(SeedVr2Vae.LatentChannels))
    def reset(): Unit = {
      in.reset(); middle.reset(); out.reset()
      downs.foreach { (residuals, down) =>
        residuals.foreach(_.reset()); down.foreach(_.reset())
      }
    }
  }

  private def part(name: String, weight: String) =
    if (!source.has(weight))
      throw new FormatException(s"no SeedVR2 VAE $name ($weight)")

  private var decoderMade: Option[Decoder] = None
  private def decoder: Decoder = decoderMade.getOrElse {
    part("decoder", "decoder.conv_in.weight")
    val made = new Decoder
    decoderMade = Some(made)
    made
  }
  private var encoderMade: Option[Encoder] = None
  private def encoder: Encoder = encoderMade.getOrElse {
    part("encoder", "encoder.conv_in.weight")
    val made = new Encoder
    encoderMade = Some(made)
    made
  }

  // ---- frames ----------------------------------------------------------------------

  private def frame(height: Long, width: Long, channels: Long): Tensor =
    ops.allocate(DType.F32, Shape.of(height, width, channels))

  private def pixels(x: Tensor): Tensor = {
    val Seq(h, w, c) = x.shape.dimensions
    x.view(h * w, c)
  }

  private def sizes(x: Tensor): (Long, Long, Long) = {
    val Seq(h, w, c) = x.shape.dimensions
    (h, w, c)
  }

  private def copy(x: Tensor): Tensor = {
    val (h, w, c) = sizes(x)
    val out = frame(h, w, c)
    ops.copy(x, out)
    out
  }

  /** The 3×3 `slices` over `inputs`, summed. */
  private def summed(
      inputs: Seq[Tensor],
      slices: Seq[Convolution],
      stride: Int
  ): Tensor = {
    val (h, w, _) = sizes(inputs.head)
    val out = frame(h / stride, w / stride, slices.last.outChannels)
    ops.conv3x3(inputs.last, slices.last.weight, slices.last.bias, out, stride)
    inputs.zip(slices).dropRight(1).foreach { (input, slice) =>
      val part = frame(h / stride, w / stride, slices.last.outChannels)
      try {
        ops.conv3x3(input, slice.weight, slice.bias, part, stride)
        ops.add(out, part, out)
      } finally ops.release(part)
    }
    out
  }

  /** Whether more frames follow the first: a lone frame keeps no history. */
  private var streaming = true

  /** `x` through `conv`, its history updated; `x` becomes history, or is
    * released when no frame follows (the caller must not release it).
    */
  private def causal(x: Tensor, conv: Causal): Tensor =
    if (conv.history.isEmpty) {
      val out = summed(Seq(x), Seq(conv.still), stride = 1)
      if (streaming) conv.history = List(x) else ops.release(x)
      out
    } else {
      val newer = conv.history.head
      val oldest = conv.history.lift(1).getOrElse(newer)
      val out = summed(Seq(oldest, newer, x), conv.slices, stride = 1)
      conv.history.drop(1).foreach(ops.release)
      conv.history = x :: conv.history.take(1)
      out
    }

  private def normSilu(x: Tensor, norm: ChannelNorm): Tensor = {
    val (h, w, c) = sizes(x)
    val out = frame(h, w, c)
    norm(ops, pixels(x), pixels(out))
    ops.activation(Activation.Silu, out, out)
    out
  }

  private def linear(x: Tensor, conv: Convolution): Tensor = {
    val (h, w, _) = sizes(x)
    val out = frame(h, w, conv.outChannels)
    ops.linear(pixels(x), conv.weight, pixels(out))
    ops.addRow(pixels(out), conv.bias, pixels(out))
    out
  }

  /** A residual block on one frame; `x` is released. */
  private def residual(x: Tensor, block: Residual): Tensor = {
    val first = causal(normSilu(x, block.norm1), block.conv1)
    val second = causal(normSilu(first, block.norm2), block.conv2)
    ops.release(first)
    block.shortcut match {
      case Some(conv) =>
        val shortcut = linear(x, conv)
        ops.add(second, shortcut, second)
        ops.release(shortcut)
      case None => ops.add(second, x, second)
    }
    ops.release(x)
    second
  }

  /** The middle on one frame; `x` is released. */
  private def middle(x: Tensor, parts: Middle): Tensor = {
    val run = new VaeRun(ops)
    val attended =
      try run.result(run.attend(residual(x, parts.first), parts.attention))
      finally run.release()
    residual(attended, parts.second)
  }

  /** A downsampler on `frames`, which are released or kept as its history. */
  private def halved(frames: Seq[Tensor], down: Down): Seq[Tensor] =
    if (!down.temporal)
      frames.map { x =>
        try summed(Seq(x), down.slices, stride = 2)
        finally ops.release(x)
      }
    else
      down.previous match {
        case None =>
          val Seq(first) = frames
          val out = summed(Seq(first), down.still.toSeq, stride = 2)
          if (streaming) down.previous = Some(first) else ops.release(first)
          Seq(out)
        case Some(previous) =>
          require(frames.size % 2 == 0, s"downsampling ${frames.size} frames")
          val stream = previous +: frames
          val out = (0 until frames.size / 2).map(j =>
            summed(stream.slice(2 * j, 2 * j + 3), down.slices, stride = 2)
          )
          stream.dropRight(1).foreach(ops.release)
          down.previous = Some(frames.last)
          out
      }

  /** An upsampler on `frames`, which are released. */
  private def doubled(frames: Seq[Tensor], up: Up): Seq[Tensor] =
    frames.flatMap { x =>
      val (h, w, c) = sizes(x)
      // the stream's first frame makes one frame, the others two
      val made =
        if (up.shuffles.size == 2 && !up.started) up.shuffles.take(1)
        else up.shuffles
      up.started = true
      try
        made.map { shuffle =>
          val packed = linear(x, shuffle)
          val large = frame(2 * h, 2 * w, c)
          try ops.unpackPatches(pixels(packed), h.toInt, 2, large)
          finally ops.release(packed)
          // the convolution keeps its input as history
          causal(large, up.conv)
        }
      finally ops.release(x)
    }

  // ---- decoding --------------------------------------------------------------------

  /** Decodes `latents` (the VAE's own: each `[h, w, 16]`, in order) into frames
    * `[8h, 8w, 3]` in [−1, 1] (unclamped), handed to `emit` as they come and
    * released after it returns: `1 + 4 (T − 1)` of them.
    */
  def decode(latents: Seq[Tensor], emit: Tensor => Unit): Unit = {
    val d = decoder
    d.reset()
    streaming = latents.size > 1
    try
      latents.foreach { latent =>
        val (_, _, c) = sizes(latent)
        require(c == SeedVr2Vae.LatentChannels, s"decode: ${latent.shape}")
        // conv_in keeps its input as history
        var frames = Seq(middle(causal(copy(latent), d.in), d.middle))
        d.ups.foreach { (residuals, up) =>
          residuals.foreach(block => frames = frames.map(residual(_, block)))
          up.foreach(u => frames = doubled(frames, u))
        }
        frames.foreach { x =>
          val normed = normSilu(x, d.normOut)
          ops.release(x)
          val rgb = causal(normed, d.out)
          try emit(rgb)
          finally ops.release(rgb)
        }
      }
    finally d.reset()
  }

  // ---- encoding --------------------------------------------------------------------

  /** The latents of `video` (frames `[H, W, 3]` in [−1, 1], H and W multiples
    * of 8, `1 + 4n` of them): the posterior's mean, `1 + n` of `[H/8, W/8,
    * 16]`, the caller's to release.
    */
  def encode(video: Seq[Tensor]): Seq[Tensor] = {
    require(
      (video.size - 1) % 4 == 0,
      s"encode: ${video.size} frames, not 1 + 4n"
    )
    val e = encoder
    e.reset()
    streaming = video.size > 1
    try {
      val chunks = video.take(1) +: video.drop(1).grouped(4).toSeq
      chunks.map { chunk =>
        var frames = chunk.map(x => causal(copy(x), e.in))
        e.downs.foreach { (residuals, down) =>
          residuals.foreach(block => frames = frames.map(residual(_, block)))
          down.foreach(d => frames = halved(frames, d))
        }
        val Seq(x) = frames
        val last = middle(x, e.middle)
        val normed = normSilu(last, e.normOut)
        ops.release(last)
        causal(normed, e.out)
      }
    } finally e.reset()
  }

  def close(): Unit = {
    decoderMade.foreach(_.reset())
    encoderMade.foreach(_.reset())
    weights.release()
    source.close()
  }
}

object SeedVr2Vae {

  val LatentChannels = 16L

  /** The transformer's latents are the VAE's times this (`scaling_factor`). */
  val LatentScale = 0.9152f

  def open(ops: Ops, path: Path): SeedVr2Vae = {
    val source = WeightSource.open(ops, path)
    try new SeedVr2Vae(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
