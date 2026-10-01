package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path

/** What LTX 2's audio file keeps no tensor for: the two vocoders' upsampling
  * rates, the bandwidth extension's STFT hop, and the two sample rates.
  */
final case class LtxAudioConfig(
    vocoderStrides: Seq[Int],
    extensionStrides: Seq[Int],
    hop: Int,
    inputRate: Int,
    outputRate: Int
)

object LtxAudioConfig {

  /** LTX 2.5's released `vocoder` config. */
  val Released: LtxAudioConfig = LtxAudioConfig(
    vocoderStrides = Seq(5, 2, 2, 2, 2, 2),
    extensionStrides = Seq(6, 5, 2, 2, 2),
    hop = 80,
    inputRate = 16000,
    outputRate = 48000
  )
}

/** LTX 2's soundtrack from its audio latents (diffusers'
  * `AutoencoderKLLTX2Audio.decode`, then `LTX2VocoderWithBWE`), from the
  * official audio file (`audio_vae.*` and `vocoder.*`). The packed latents
  * `[L, C × M]` (column `c × M + m`) denormalized by the per-channel
  * statistics, as an image of `L` rows (time) by `M` (latent mel bins) of `C`
  * channels; the mel decoder causal in time (3×3 convolutions padded by two
  * rows before and none after, pixel norms, SiLU, residual stages, nearest ×2
  * upsamplers dropping their first row) into a stereo mel spectrogram of `4L −
  * 3` frames; the vocoder (BigVGAN) to a 16 kHz waveform; then the bandwidth
  * extension: each channel's causal STFT (the file's basis), its log-mel (the
  * file's filter bank), a second BigVGAN to the 48 kHz residual, added to the
  * waveform resampled ×3 (a Hann-windowed sinc). Not clamped (diffusers clamps
  * to [−1, 1]; `Soundtrack.fitted` scales the track instead).
  */
final class LtxAudio private (
    ops: Ops,
    source: WeightSource,
    config: LtxAudioConfig
) {

  private val weights = new VaeWeights(ops, source)
  private val Decoder = "audio_vae.decoder"

  private val (mean, deviation) = (
    weights.values("audio_vae.per_channel_statistics.mean-of-means"),
    weights.values("audio_vae.per_channel_statistics.std-of-means")
  )
  private val latentChannels =
    weights.shape(s"$Decoder.conv_in.conv.weight").dimensions(1).toInt
  private val latentMelBins = mean.length / latentChannels

  final private case class Residual(
      conv1: Convolution,
      conv2: Convolution,
      shortcut: Option[Convolution]
  )

  private def residual(prefix: String) = Residual(
    weights.conv3x3(s"$prefix.conv1.conv"),
    weights.conv3x3(s"$prefix.conv2.conv"),
    Option.when(weights.has(s"$prefix.nin_shortcut.conv.weight"))(
      weights.conv1x1(s"$prefix.nin_shortcut.conv")
    )
  )

  private def counted(name: Int => String) =
    Iterator.from(0).takeWhile(i => weights.has(name(i))).size

  private val convIn = weights.conv3x3(s"$Decoder.conv_in.conv")
  private val middle =
    Seq(residual(s"$Decoder.mid.block_1"), residual(s"$Decoder.mid.block_2"))

  /** From the deepest level up: its residual blocks, then its upsampler. */
  private val levels = {
    val count = counted(l => s"$Decoder.up.$l.block.0.conv1.conv.weight")
    (count - 1 to 0 by -1).map { level =>
      val prefix = s"$Decoder.up.$level"
      val blocks = (0 until counted(b => s"$prefix.block.$b.conv1.conv.weight"))
        .map(b => residual(s"$prefix.block.$b"))
      val upsampler =
        Option.when(weights.has(s"$prefix.upsample.conv.conv.weight"))(
          weights.conv3x3(s"$prefix.upsample.conv.conv")
        )
      (blocks, upsampler)
    }
  }
  private val convOut = weights.conv3x3(s"$Decoder.conv_out.conv")

  private val vocoder =
    new BigVgan(ops, weights, "vocoder.vocoder", config.vocoderStrides)
  private val extension =
    new BigVgan(ops, weights, "vocoder.bwe_generator", config.extensionStrides)

  private val (stftBasis, filterLength) = {
    val name = "vocoder.mel_stft.stft_fn.forward_basis"
    val Seq(rows, _, taps) = weights.shape(name).dimensions
    (weights.f32(Shape.of(rows, taps), weights.values(name)), taps.toInt)
  }
  private val melBasis = weights.values("vocoder.mel_stft.mel_basis")
  private val extensionMelBins =
    weights.shape("vocoder.mel_stft.mel_basis").dimensions.head.toInt
  private val ratio = config.outputRate / config.inputRate
  private val resampler = LtxAudio.hannFilter(ratio)

  val sampleRate: Int = config.outputRate

  /** The waveform's channels (stereo). */
  val channels: Int = vocoder.outChannels

  private val onesFor = scala.collection.mutable.Map.empty[Long, Tensor]
  private def ones(channels: Long) =
    onesFor.getOrElseUpdate(
      channels,
      weights.floats(Array.fill(channels.toInt)(1f))
    )

  private def sizes(x: Tensor): (Long, Long, Long) = {
    val Seq(h, w, c) = x.shape.dimensions
    (h, w, c)
  }

  /** Pixel norm (RMS over the channels, ε 10⁻⁶) then SiLU, into a new image. */
  private def normSilu(x: Tensor): Tensor = {
    val (h, w, c) = sizes(x)
    val out = ops.allocate(DType.F32, x.shape)
    ops.rmsNorm(x.view(h * w, c), ones(c), 1e-6f, 0f, out.view(h * w, c))
    ops.activation(Activation.Silu, out, out)
    out
  }

  /** A 3×3 convolution causal in time: a zero row before `x`, the usual
    * padding, the rows from `drop` up to `x`'s count.
    */
  private def causal(x: Tensor, conv: Convolution, drop: Int = 0): Tensor = {
    val (h, w, c) = sizes(x)
    val padded = ops.allocate(DType.F32, Shape.of(h + 1, w, c))
    val full = ops.allocate(DType.F32, Shape.of(h + 1, w, conv.outChannels))
    try {
      ops.zero(padded.rows(0, 1))
      ops.copy(x, padded.rows(1, h))
      ops.conv3x3(padded, conv.weight, conv.bias, full)
      val out = ops.allocate(DType.F32, Shape.of(h - drop, w, conv.outChannels))
      ops.copy(full.rows(drop, h - drop), out)
      out
    } finally Seq(padded, full).foreach(ops.release)
  }

  private def residual(x: Tensor, block: Residual): Tensor = {
    val (h, w, c) = sizes(x)
    val first = normSilu(x)
    val convolved = causal(first, block.conv1)
    ops.release(first)
    val second = normSilu(convolved)
    ops.release(convolved)
    val out = causal(second, block.conv2)
    ops.release(second)
    block.shortcut match {
      case Some(conv) =>
        val shortcut =
          ops.allocate(DType.F32, Shape.of(h * w, conv.outChannels))
        ops.linear(x.view(h * w, c), conv.weight, shortcut)
        ops.addRow(shortcut, conv.bias, shortcut)
        ops.add(out, shortcut.view(h, w, conv.outChannels), out)
        ops.release(shortcut)
      case None => ops.add(out, x, out)
    }
    out
  }

  /** Nearest ×2, then the causal convolution without its first row. */
  private def upsample(x: Tensor, conv: Convolution): Tensor = {
    val (h, w, c) = sizes(x)
    val doubled = ops.allocate(DType.F32, Shape.of(2 * h, 2 * w, c))
    try {
      ops.upsample2x(x, doubled)
      causal(doubled, conv, drop = 1)
    } finally ops.release(doubled)
  }

  /** The stereo mel spectrogram `[T, mel bins, channels]` of `frames` packed
    * latent rows, and its `(T, mel bins)`.
    */
  private[runner] def spectrogram(latents: Array[Float], frames: Int) = {
    val width = mean.length
    val image = Array.tabulate(frames * width) { i =>
      val (t, rest) = (i / width, i % width)
      val (m, c) = (rest / latentChannels, rest % latentChannels)
      val j = c * latentMelBins + m
      latents(t * width + j) * deviation(j) + mean(j)
    }
    var x = ops.fromFloats(
      Shape.of(frames, latentMelBins, latentChannels),
      image
    )
    def step(next: Tensor): Unit = {
      ops.release(x)
      x = next
    }
    try {
      step(causal(x, convIn))
      middle.foreach(block => step(residual(x, block)))
      levels.foreach { (blocks, upsampler) =>
        blocks.foreach(block => step(residual(x, block)))
        upsampler.foreach(conv => step(upsample(x, conv)))
      }
      step(normSilu(x))
      step(causal(x, convOut))
      val (time, bins, _) = sizes(x)
      (ops.toFloats(x), time.toInt, bins.toInt)
    } finally ops.release(x)
  }

  /** One channel's log-mel frames `[samples / hop, mel bins]`: the causal STFT
    * (zeros before, `filter length − hop`), its magnitudes through the filter
    * bank, their log above 10⁻⁵.
    */
  private def logMel(samples: Array[Float]): Array[Float] = {
    val hop = config.hop
    val frames = samples.length / hop
    val bins = stftBasis.shape.dimensions.head.toInt / 2
    val x = ops.fromFloats(Shape.of(samples.length, 1), samples)
    val spectrum = ops.allocate(DType.F32, Shape.of(frames, 2L * bins))
    val values =
      try {
        ops.conv1d(
          x,
          stftBasis,
          None,
          filterLength,
          1,
          hop,
          filterLength - hop,
          0,
          spectrum
        )
        ops.toFloats(spectrum)
      } finally Seq(x, spectrum).foreach(ops.release)
    val out = new Array[Float](frames * extensionMelBins)
    (0 until frames).foreach { frame =>
      val row = frame * 2 * bins
      val magnitudes = Array.tabulate(bins) { f =>
        val (re, im) =
          (values(row + f).toDouble, values(row + bins + f).toDouble)
        math.sqrt(re * re + im * im)
      }
      (0 until extensionMelBins).foreach { b =>
        var sum = 0.0
        (0 until bins).foreach(f =>
          sum += melBasis(b * bins + f) * magnitudes(f)
        )
        out(frame * extensionMelBins + b) =
          math.log(math.max(sum, 1e-5)).toFloat
      }
    }
    out
  }

  /** `samples` ×`ratio` through the Hann-windowed sinc (a transposed
    * convolution of the replicate-padded signal, times the ratio, cropped).
    */
  private def resampled(samples: Array[Float]): Array[Float] = {
    val (width, taps) = (LtxAudio.HannWidth, resampler.length)
    val cropLeft = 2 * width * ratio
    def padded(i: Int) = samples(
      math.min(math.max(i - width, 0), samples.length - 1)
    )
    Array.tabulate(samples.length * ratio) { n =>
      val q = n + cropLeft
      var sum = 0.0
      var k = q % ratio
      while (k < taps) {
        sum += resampler(k) * padded((q - k) / ratio)
        k += ratio
      }
      (ratio * sum).toFloat
    }
  }

  /** The soundtrack of `latents` (packed rows `[L, C × M]`), interleaved over
    * `channels` at `sampleRate`.
    */
  def decode(latents: Array[Float]): Array[Float] = {
    val frames = latents.length / mean.length
    val (mel, time, bins) = spectrogram(latents, frames)
    // [T, bins, C] → rows of channel-major bins (column c × bins + m)
    val rows = Array.tabulate(mel.length) { i =>
      val (t, j) = (i / (channels * bins), i % (channels * bins))
      mel((t * bins + j % bins) * channels + j / bins)
    }
    val low = {
      val x = ops.fromFloats(Shape.of(time, channels.toLong * bins), rows)
      try {
        val wave = vocoder(x)
        try ops.toFloats(wave)
        finally ops.release(wave)
      } finally ops.release(x)
    }
    val samples = low.length / channels
    // whole STFT hops, zeros after
    val hops = (samples + config.hop - 1) / config.hop
    val perChannel = (0 until channels).map { c =>
      Array.tabulate(hops * config.hop)(i =>
        if (i < samples) low(i * channels + c) else 0f
      )
    }
    val mels = perChannel.map(logMel)
    val extensionInput = Array.tabulate(hops * channels * extensionMelBins) {
      i =>
        val (frame, j) =
          (i / (channels * extensionMelBins), i % (channels * extensionMelBins))
        mels(j / extensionMelBins)(
          frame * extensionMelBins + j % extensionMelBins
        )
    }
    val high = {
      val x = ops.fromFloats(
        Shape.of(hops, channels.toLong * extensionMelBins),
        extensionInput
      )
      try {
        val wave = extension(x)
        try ops.toFloats(wave)
        finally ops.release(wave)
      } finally ops.release(x)
    }
    val skips = perChannel.map(resampled)
    val length = samples * ratio
    require(
      high.length / channels >= length,
      s"the bandwidth extension made ${high.length / channels} samples for $length"
    )
    Array.tabulate(length * channels) { i =>
      val (n, c) = (i / channels, i % channels)
      high(i) + skips(c)(n)
    }
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object LtxAudio {

  /** The Hann resampler's half-width in input samples (`ceil(6 / 0.99)`). */
  private val HannWidth = 7

  /** diffusers' `UpSample1d(window_type="hann")` filter for `ratio`: a sinc
    * rolled off to 0.99 under a Hann window 6 zero crossings wide, `2 × 7 ×
    * ratio + 1` taps.
    */
  private def hannFilter(ratio: Int): Array[Float] = {
    val (rolloff, zeros) = (0.99, 6.0)
    Array.tabulate(2 * HannWidth * ratio + 1) { k =>
      val t = (k.toDouble / ratio - HannWidth) * rolloff
      val clamped = math.min(math.max(t, -zeros), zeros)
      val window = math.pow(math.cos(clamped * math.Pi / zeros / 2), 2)
      val sinc = if (t == 0) 1.0 else math.sin(math.Pi * t) / (math.Pi * t)
      (sinc * window * rolloff / ratio).toFloat
    }
  }

  def open(
      ops: Ops,
      path: Path,
      config: LtxAudioConfig = LtxAudioConfig.Released
  ): LtxAudio = {
    val source = WeightSource.open(ops, path)
    try {
      if (
        !source.has("audio_vae.decoder.conv_in.conv.weight") ||
        !source.has("vocoder.vocoder.conv_pre.weight")
      )
        throw new FormatException(
          s"$path is no LTX 2 audio VAE with its vocoder"
        )
      new LtxAudio(ops, source, config)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
