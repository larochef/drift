package drift.runner

import drift.runner.models.{Ltx2, LtxTextFeatures, WeightSource}
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.tensor.Shape

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}

/** LTX 2's text features and connectors on raw hidden states, for a
  * comparison with diffusers on the released files: `TEXT_ENCODER
  * TRANSFORMER DIR`, `DIR` holding little-endian F32 `states.f32` (the valid
  * tokens' 49 hidden states, state-major `[49 × tokens, 3840]`); the
  * connectors' 1024 rows are written beside it as `video_rows.f32` and
  * `audio_rows.f32`.
  */
object LtxTextCheck {

  private def floats(path: Path): Array[Float] = {
    val buffer =
      ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN)
    Array.fill(buffer.remaining / 4)(buffer.getFloat)
  }

  private def write(path: Path, values: Array[Float]): Unit = {
    val buffer =
      ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN)
    values.foreach(buffer.putFloat)
    Files.write(path, buffer.array())
  }

  def main(arguments: Array[String]): Unit = {
    val Array(textEncoder, transformer, folder) = arguments
    val dir = Paths.get(folder)
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      val textSource = WeightSource.open(ops, Paths.get(textEncoder))
      val features = new LtxTextFeatures(ops, textSource, 49, 3840)
      val (source, video, audio) = Ltx2.connectors(ops, Paths.get(transformer))
      try {
        val values = floats(dir.resolve("states.f32"))
        val states = ops.fromFloats(Shape.of(values.length / 3840, 3840), values)
        val (v, a) = features(states)
        val (videoRows, audioRows) = (video(v, 1024), audio(a, 1024))
        write(dir.resolve("video_rows.f32"), ops.toFloats(videoRows))
        write(dir.resolve("audio_rows.f32"), ops.toFloats(audioRows))
        println(s"connector rows ${videoRows.shape} and ${audioRows.shape}")
      } finally {
        audio.release()
        video.release()
        source.close()
        features.release()
        textSource.close()
      }
    } finally ops.close()
  }
}
