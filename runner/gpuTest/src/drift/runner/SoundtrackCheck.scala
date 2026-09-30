package drift.runner

import drift.runner.models.{LtxAudio, MiniMaxH3Audio}
import drift.runner.ops.{HipOps, MatVecInputs}

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Paths}

/** Decodes latents with a released audio file, for a comparison with diffusers
  * on the same file: `ltx|h3 AUDIO_FILE LATENTS OUT`, the latents and the
  * samples raw little-endian F32 (LTX's packed rows `[L, 128]`, interleaved
  * stereo out; one H3 channel's rows `[L, 32]`, mono out).
  */
object SoundtrackCheck {

  private def floats(bytes: Array[Byte]): Array[Float] = {
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.fill(bytes.length / 4)(buffer.getFloat)
  }

  def main(arguments: Array[String]): Unit = {
    val Array(family, file, latentsFile, outFile) = arguments
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      val latents = floats(Files.readAllBytes(Paths.get(latentsFile)))
      val started = System.nanoTime()
      val samples = family match {
        case "ltx" =>
          val decoder = LtxAudio.open(ops, Paths.get(file))
          try decoder.decode(latents)
          finally decoder.close()
        case "ltx-mel" =>
          val decoder = LtxAudio.open(ops, Paths.get(file))
          try decoder.spectrogram(latents, latents.length / 128)._1
          finally decoder.close()
        case "h3" =>
          val decoder = MiniMaxH3Audio.open(ops, Paths.get(file))
          try decoder.decode(latents)
          finally decoder.close()
      }
      println(
        f"${samples.length} samples in ${(System.nanoTime() - started) / 1e9}%.1f s"
      )
      val out = ByteBuffer
        .allocate(4 * samples.length)
        .order(ByteOrder.LITTLE_ENDIAN)
      samples.foreach(out.putFloat)
      Files.write(Paths.get(outFile), out.array())
    } finally ops.close()
  }
}
