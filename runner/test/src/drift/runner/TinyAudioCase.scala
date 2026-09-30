package drift.runner

import drift.runner.models.{LtxAudio, LtxAudioConfig, MiniMaxH3Audio}
import drift.runner.ops.Ops

/** The golden tiny soundtrack decoders (`fixtures/tiny_diffusion.py`) on a
  * backend, against diffusers'.
  */
object TinyAudioCase {

  private def relative(actual: Array[Float], expected: Array[Float]): Double = {
    assert(
      actual.length == expected.length,
      s"${actual.length} samples for ${expected.length}"
    )
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** MiniMax H3's audio decoder, two channels of 6 latents into 60 samples
    * each: the worse channel's error.
    */
  def minimaxH3Error(ops: Ops): Double = {
    val decoder = MiniMaxH3Audio.open(
      ops,
      Fixtures.path("tiny/minimax_h3_audio/model.safetensors"),
      strides = Seq(5, 2)
    )
    try
      Fixtures.withSafetensors("tiny/minimax_h3_audio/expected.safetensors") {
        golden =>
          val (latents, waveform) =
            (golden("latents").decode(), golden("waveform").decode())
          val (rows, samples) = (latents.length / 2, waveform.length / 2)
          (0 until 2)
            .map(c =>
              relative(
                decoder.decode(latents.slice(c * rows, (c + 1) * rows)),
                waveform.slice(c * samples, (c + 1) * samples)
              )
            )
            .max
      }
    finally decoder.close()
  }

  /** LTX 2's audio decoder, vocoder and bandwidth extension: 5 latent frames
    * into 510 stereo samples at 48 kHz.
    */
  def ltxError(ops: Ops): Double = {
    val decoder = LtxAudio.open(
      ops,
      Fixtures.path("tiny/ltx_audio/model.safetensors"),
      LtxAudioConfig(
        vocoderStrides = Seq(5, 2),
        extensionStrides = Seq(6, 5),
        hop = 10,
        inputRate = 16000,
        outputRate = 48000
      )
    )
    try
      Fixtures.withSafetensors("tiny/ltx_audio/expected.safetensors") {
        golden =>
          relative(
            decoder.decode(golden("latents").decode()),
            golden("waveform").decode()
          )
      }
    finally decoder.close()
  }
}
