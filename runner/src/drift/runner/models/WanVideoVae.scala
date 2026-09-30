package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** The Wan 2.1 VAE on video (diffusers' `AutoencoderKLWan`, sd-cpp's
  * `wan_vae.hpp`), from the original names, channels-last frame by frame. Its
  * causal 3-D convolutions are streams: each keeps the last two frames it was
  * given (zeros before the first), and a frame's output is its three temporal
  * slices over the frame and those two, all 3×3 convolutions. The chunking of
  * the reference (the decoder one latent frame at a time, the encoder a first
  * frame then four at a time) gives the same streams. Temporal resampling:
  *   - the decoder's `upsample3d` doubles each frame by its time convolution
  *     (two frames of the output's channels), but for the very first frame,
  *     which passes alone, the stream starting after it;
  *   - the encoder's `downsample3d` passes the first frame alone, then turns
  *     each pair of frames and the frame before them into one.
  * One latent frame makes 4 frames, but the first which makes 1.
  */
final class WanVideoVae private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  private val weights = new VaeWeights(ops, source)
  private val layers = new WanLayers(weights)
  import layers.norm

  /** A causal 3×3×3 convolution and its two frames of history. */
  final private class Causal(prefix: String) {
    val slices: Seq[Convolution] = weights.conv3x3x3(prefix)
    var history: List[Tensor] = Nil // newest first
    def outChannels: Long = slices.last.outChannels
    def reset(): Unit = { history.foreach(ops.release); history = Nil }
  }

  /** A causal temporal (3×1×1) convolution and its history. */
  final private class Temporal(prefix: String) {
    val slices: Seq[Convolution] = weights.conv3x1x1(prefix)
    var history: List[Tensor] = Nil
    var started = false
    def reset(): Unit = {
      history.foreach(ops.release); history = Nil; started = false
    }
  }

  final private class Residual(prefix: String) {
    val norm1: ChannelNorm = norm(s"$prefix.residual.0")
    val conv1 = new Causal(s"$prefix.residual.2")
    val norm2: ChannelNorm = norm(s"$prefix.residual.3")
    val conv2 = new Causal(s"$prefix.residual.6")
    val shortcut: Option[Convolution] =
      Option.when(weights.has(s"$prefix.shortcut.weight"))(
        weights.conv1x1(s"$prefix.shortcut")
      )
    def causals: Seq[Causal] = Seq(conv1, conv2)
  }

  private enum Stage {
    case Block(residual: Residual)
    case Up(conv: Convolution, time: Option[Temporal])
    case Down(conv: Convolution, time: Option[Temporal])
  }

  private def stages(prefix: String, up: Boolean): Seq[Stage] =
    Iterator
      .from(0)
      .map(i => s"$prefix.$i")
      .takeWhile(p =>
        source.has(s"$p.residual.0.gamma") || source.has(
          s"$p.resample.1.weight"
        )
      )
      .map { p =>
        if (source.has(s"$p.residual.0.gamma")) Stage.Block(new Residual(p))
        else {
          val conv = weights.conv3x3(s"$p.resample.1")
          val time = Option.when(source.has(s"$p.time_conv.weight"))(
            new Temporal(s"$p.time_conv")
          )
          if (up) Stage.Up(conv, time) else Stage.Down(conv, time)
        }
      }
      .toSeq

  final private class Middle(prefix: String) {
    val first = new Residual(s"$prefix.middle.0")
    val attention: PixelAttention = layers.attention(s"$prefix.middle.1")
    val second = new Residual(s"$prefix.middle.2")
  }

  if (!source.has("decoder.conv1.weight"))
    throw new FormatException("no Wan VAE decoder (decoder.conv1.weight)")
  private val postQuant = weights.conv1x1("conv2")
  private val decoderIn = new Causal("decoder.conv1")
  private val decoderMiddle = new Middle("decoder")
  private val upsamples = stages("decoder.upsamples", up = true)
  private val decoderHeadNorm = norm("decoder.head.0")
  private val decoderHead = new Causal("decoder.head.2")

  final private class Encoder {
    val in = new Causal("encoder.conv1")
    val downsamples: Seq[Stage] = stages("encoder.downsamples", up = false)
    val middle = new Middle("encoder")
    val headNorm: ChannelNorm = norm("encoder.head.0")
    val head = new Causal("encoder.head.2")
    val quant: Convolution = {
      val conv = weights.conv1x1("conv1")
      conv.take(conv.outChannels / 2)
    }
  }
  private lazy val encoder = {
    if (!source.has("encoder.conv1.weight"))
      throw new FormatException("no Wan VAE encoder (encoder.conv1.weight)")
    new Encoder
  }

  private val latentScale = weights.floats(WanVideoVae.LatentStd.map(_ - 1f))
  private val latentShift = weights.floats(WanVideoVae.LatentMean)
  // (z − mean) / std, as `modulate`'s x × (1 + scale) + shift
  private val normalizeScale =
    weights.floats(WanVideoVae.LatentStd.map(1 / _ - 1))
  private val normalizeShift = weights.floats(
    WanVideoVae.LatentMean.indices
      .map(i => -WanVideoVae.LatentMean(i) / WanVideoVae.LatentStd(i))
      .toArray
  )

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

  /** `x` through `conv` (stride 1), its history updated; `x` becomes history
    * (the caller must not release it).
    */
  private def causal(x: Tensor, conv: Causal): Tensor = {
    val (h, w, _) = sizes(x)
    val out = frame(h, w, conv.outChannels)
    ops.conv3x3(x, conv.slices(2).weight, conv.slices(2).bias, out)
    val older = conv.history.zip(Seq(conv.slices(1), conv.slices(0)))
    older.foreach { (previous, slice) =>
      val part = frame(h, w, conv.outChannels)
      try {
        ops.conv3x3(previous, slice.weight, slice.bias, part)
        ops.add(out, part, out)
      } finally ops.release(part)
    }
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

  /** A temporal convolution's output over `window` (oldest first, a missing
    * frame being zeros): the 1×1 `slices` summed.
    */
  private def temporal(
      window: Seq[Option[Tensor]],
      slices: Seq[Convolution]
  ): Tensor = {
    val (h, w, _) = sizes(window.flatten.head)
    val out = frame(h, w, slices.last.outChannels)
    ops.zero(out)
    window.zip(slices).foreach {
      case (Some(input), slice) =>
        val part = linear(input, slice)
        ops.add(out, part, out)
        ops.release(part)
      case (None, slice) =>
        // a zero frame gives its slice's bias alone
        ops.addRow(pixels(out), slice.bias, pixels(out))
    }
    out
  }

  /** Output channels `[from, from + count)` of convolutions. */
  private def part(slices: Seq[Convolution], from: Long, count: Long) =
    slices.map(s =>
      Convolution(s.weight.rows(from, count), s.bias.rows(from, count))
    )

  /** `upsample3d`'s time convolution on `frames`: each frame two, but for the
    * stream's first frame, which passes alone.
    */
  private def doubled(frames: Seq[Tensor], time: Temporal): Seq[Tensor] =
    if (!time.started) {
      time.started = true
      frames
    } else
      frames.flatMap { x =>
        val window = Seq(time.history.lift(1), time.history.headOption, Some(x))
        val channels = time.slices.last.outChannels / 2
        val pair = Seq(0L, channels)
          .map(from => temporal(window, part(time.slices, from, channels)))
        time.history.drop(1).foreach(ops.release)
        time.history = x :: time.history.take(1)
        pair
      }

  /** `downsample3d`'s time convolution on `frames`: the stream's first frame
    * alone, then each pair with the frame before it into one (stride 2).
    */
  private def halved(frames: Seq[Tensor], time: Temporal): Seq[Tensor] =
    if (!time.started) {
      time.started = true
      time.history = List(copy(frames.last))
      frames
    } else {
      require(frames.size % 2 == 0, s"downsample3d over ${frames.size} frames")
      val stream = time.history.head +: frames
      val out = (0 until frames.size / 2).map(j =>
        temporal(
          Seq(
            Some(stream(2 * j)),
            Some(stream(2 * j + 1)),
            Some(stream(2 * j + 2))
          ),
          time.slices
        )
      )
      time.history.foreach(ops.release)
      time.history = List(copy(frames.last))
      frames.foreach(ops.release)
      out
    }

  private def copy(x: Tensor): Tensor = {
    val (h, w, c) = sizes(x)
    val out = frame(h, w, c)
    ops.copy(x, out)
    out
  }

  /** A 2-D resampling convolution on one frame (after nearest ×2 when `up`,
    * else stride 2); `x` is released.
    */
  private def resample(x: Tensor, conv: Convolution, up: Boolean): Tensor = {
    val run = new VaeRun(ops)
    try
      run.result(
        if (up) run.upsample(x, conv) else run.conv3x3(x, conv, stride = 2)
      )
    finally run.release()
  }

  // ---- decoding --------------------------------------------------------------------

  private def resetDecoder(): Unit = {
    decoderIn.reset()
    decoderHead.reset()
    (Seq(decoderMiddle.first, decoderMiddle.second) ++ upsamples.collect {
      case Stage.Block(r) => r
    }).flatMap(_.causals).foreach(_.reset())
    upsamples.collect { case Stage.Up(_, Some(t)) => t }.foreach(_.reset())
  }

  /** Decodes `latents` (the diffusion model's, normalized: each `[h, w, 16]`,
    * in order) into frames `[8h, 8w, 3]` in [−1, 1] (unclamped), handed to
    * `emit` as they come and released after it returns: `1 + 4 (T − 1)` of
    * them.
    */
  def decode(latents: Seq[Tensor], emit: Tensor => Unit): Unit = {
    resetDecoder()
    try
      latents.foreach { latent =>
        val (h, w, c) = sizes(latent)
        require(c == 16, s"decode: latents ${latent.shape}")
        val z = frame(h, w, 16)
        ops.modulate(pixels(latent), latentScale, latentShift, pixels(z))
        val quant = linear(z, postQuant)
        ops.release(z)
        // conv1 keeps its input as history
        var frames = Seq(middle(causal(quant, decoderIn), decoderMiddle))
        upsamples.foreach {
          case Stage.Block(block)   => frames = frames.map(residual(_, block))
          case Stage.Up(conv, time) =>
            // a doubled frame's input stays as the time convolution's history
            frames = time
              .fold(frames)(doubled(frames, _))
              .map(resample(_, conv, up = true))
          case Stage.Down(_, _) => ()
        }
        frames.foreach { x =>
          val normed = normSilu(x, decoderHeadNorm)
          ops.release(x)
          val rgb = causal(normed, decoderHead)
          try emit(rgb)
          finally ops.release(rgb)
        }
      }
    finally resetDecoder()
  }

  // ---- encoding --------------------------------------------------------------------

  private def resetEncoder(): Unit = {
    val e = encoder
    e.in.reset()
    e.head.reset()
    (Seq(e.middle.first, e.middle.second) ++ e.downsamples.collect {
      case Stage.Block(r) => r
    }).flatMap(_.causals).foreach(_.reset())
    e.downsamples
      .collect { case Stage.Down(_, Some(t)) => t }
      .foreach(_.reset())
  }

  /** The latents of `video` (frames `[H, W, 3]` in [−1, 1], H and W multiples
    * of 8, `1 + 4n` of them), normalized as the diffusion models take them:
    * `1 + n` of `[H/8, W/8, 16]`, the caller's to release.
    */
  def encode(video: Seq[Tensor]): Seq[Tensor] = {
    require(
      (video.size - 1) % 4 == 0,
      s"encode: ${video.size} frames, not 1 + 4n"
    )
    val e = encoder
    resetEncoder()
    try {
      val chunks = video.take(1) +: video.drop(1).grouped(4).toSeq
      chunks.map { chunk =>
        var frames = chunk.map(x => causal(copy(x), e.in))
        e.downsamples.foreach {
          case Stage.Block(block)     => frames = frames.map(residual(_, block))
          case Stage.Down(conv, time) =>
            val spatial = frames.map(resample(_, conv, up = false))
            frames = time.fold(spatial)(halved(spatial, _))
          case Stage.Up(_, _) => ()
        }
        val Seq(x) = frames
        val normed = normSilu(middle(x, e.middle), e.headNorm)
        val head = causal(normed, e.head)
        val latent = linear(head, e.quant)
        ops.release(head)
        ops.modulate(
          pixels(latent),
          normalizeScale,
          normalizeShift,
          pixels(latent)
        )
        latent
      }
    } finally resetEncoder()
  }

  def close(): Unit = {
    resetDecoder()
    if (source.has("encoder.conv1.weight")) resetEncoder()
    weights.release()
    source.close()
  }
}

object WanVideoVae {

  /** Wan 2.1's per-channel latent statistics: the diffusion model's latents are
    * `(z − mean) / std`.
    */
  val LatentMean: Array[Float] = Array(-0.7571f, -0.7089f, -0.9113f, 0.1075f,
    -0.1745f, 0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f,
    -0.1922f, -0.9497f, 0.2503f, -0.2921f)
  val LatentStd: Array[Float] = Array(2.8184f, 1.4541f, 2.3275f, 2.6558f,
    1.2196f, 1.7708f, 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
    1.6382f, 1.1253f, 2.8251f, 1.9160f)

  def open(ops: Ops, path: Path): WanVideoVae = {
    val source = WeightSource.open(ops, path)
    try new WanVideoVae(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
