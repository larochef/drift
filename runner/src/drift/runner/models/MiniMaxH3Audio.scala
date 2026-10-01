package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.Ops
import drift.runner.tensor.*

import java.nio.file.Path

/** MiniMax H3's audio decoder (diffusers' `AutoencoderKLMiniMaxH3Audio.decode`,
  * the audio VAE's `dec_in_proj` and `decoder`): mono latents `[L, 32]`
  * denormalized by the file's `latents_mean` and `latents_std`, a 1×1
  * convolution to the trunk's width, then BigVGAN up to `L × 800` samples at 32
  * kHz, not clamped (diffusers clamps to [−1, 1]; `Soundtrack.fitted` scales
  * the stereo track instead). The model is mono: a stereo soundtrack is two
  * decodes. `strides` are the file's `decoder_rates`.
  */
final class MiniMaxH3Audio private (
    ops: Ops,
    source: WeightSource,
    strides: Seq[Int]
) {

  private val weights = new VaeWeights(ops, source)
  private val (mean, deviation) =
    (weights.values("latents_mean"), weights.values("latents_std"))
  private val inputProjection =
    TimeConvolution.load(weights, "dec_in_proj")
  private val vocoder = new BigVgan(ops, weights, "decoder", strides)

  val sampleRate: Int = MiniMaxH3Audio.SampleRate

  /** The latent channels a frame holds. */
  val channels: Int = mean.length

  /** One channel's samples from its normalized latents `[L, channels]`
    * (row-major).
    */
  def decode(latents: Array[Float]): Array[Float] = {
    val frames = latents.length / channels
    val denormalized = Array.tabulate(latents.length) { i =>
      latents(i) * deviation(i % channels) + mean(i % channels)
    }
    val x = ops.fromFloats(Shape.of(frames, channels), denormalized)
    try {
      val trunk = inputProjection(ops, x)
      try {
        val waveform = vocoder(trunk)
        try ops.toFloats(waveform)
        finally ops.release(waveform)
      } finally ops.release(trunk)
    } finally ops.release(x)
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object MiniMaxH3Audio {

  val SampleRate = 32000

  /** The released decoder's upsampling rates (800 samples a latent). */
  val Strides: Seq[Int] = Seq(5, 5, 2, 2, 2, 2, 2)

  def open(
      ops: Ops,
      path: Path,
      strides: Seq[Int] = Strides
  ): MiniMaxH3Audio = {
    val source = WeightSource.open(ops, path)
    try {
      if (!source.has("dec_in_proj.weight") || !source.has("latents_mean"))
        throw new FormatException(s"$path is no MiniMax H3 audio VAE")
      new MiniMaxH3Audio(ops, source, strides)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
