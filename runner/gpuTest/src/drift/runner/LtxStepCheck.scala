package drift.runner

import drift.runner.models.{Ltx2, Ltx2Layout}
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.tensor.{DType, Shape}

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}

/** One step of a released LTX 2 transformer on raw inputs, for a comparison
  * with diffusers on the same file: `TRANSFORMER DIR FRAMES HEIGHT WIDTH
  * AUDIO_FRAMES FPS TIMESTEP`, `DIR` holding little-endian F32 `video.f32`
  * (`[tokens, 128]`), `audio.f32` (`[audio frames, 128]`), `text.f32` and
  * `audio_text.f32` (the connectors' rows); the velocities are written beside
  * them as `video_out.f32` and `audio_out.f32`.
  */
object LtxStepCheck {

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
    val Array(file, folder, frames, height, width, audioFrames, fps, timestep) =
      arguments
    val dir = Paths.get(folder)
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      val model = Ltx2.open(ops, Paths.get(file))
      try {
        val c = model.config
        val layout = Ltx2Layout(
          frames.toInt,
          height.toInt,
          width.toInt,
          audioFrames.toInt,
          fps.toDouble
        )
        def input(name: String, columns: Long) = {
          val values = floats(dir.resolve(name))
          ops.fromFloats(Shape.of(values.length / columns, columns), values)
        }
        val video = input("video.f32", c.videoChannels)
        val audio = input("audio.f32", c.audioChannels)
        val text = input("text.f32", c.hidden)
        val audioText = input("audio_text.f32", c.audioHidden)
        val (videoOut, audioOut) = (
          ops.allocate(DType.F32, video.shape),
          ops.allocate(DType.F32, audio.shape)
        )
        model.velocity(
          layout,
          video,
          audio,
          text,
          audioText,
          timestep.toFloat,
          videoOut,
          audioOut
        )
        write(dir.resolve("video_out.f32"), ops.toFloats(videoOut))
        write(dir.resolve("audio_out.f32"), ops.toFloats(audioOut))
        println(s"velocities of ${layout.videoTokens} video and ${layout.audioFrames} audio rows")
      } finally model.close()
    } finally ops.close()
  }
}
