package drift.runner

import drift.runner.models.{
  MiniMaxH3AudioEncoder,
  MiniMaxH3VideoEncoder,
  Posterior
}
import drift.runner.ops.{CpuOps, HipOps, MatVecInputs}

import java.nio.file.Paths

/** Loads the released VAEs' encoders and runs them on small inputs, for their
  * names and shapes: `VIDEO_VAE AUDIO_VAE`. A 64 × 96 gradient frame and a
  * 22-frame clip of it through the video encoder, one second of a 440 Hz sine
  * through the audio encoder; prints each result's shape and range, and with an
  * `OUT` folder writes them there (raw little-endian F32: `frame`, `clip`,
  * `sound`) for a comparison with diffusers; `cpu` after it runs the frame
  * alone on the reference backend.
  */
object MiniMaxH3EncoderCheck {

  def main(arguments: Array[String]): Unit = {
    val Seq(videoVae, audioVae) = arguments.toSeq.take(2)
    val folder = arguments.lift(2).map(Paths.get(_))
    def write(name: String, values: Array[Float]) = folder.foreach { f =>
      val buffer = java.nio.ByteBuffer
        .allocate(4 * values.length)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
      buffer.asFloatBuffer().put(values)
      java.nio.file.Files.write(f.resolve(name), buffer.array())
    }
    // `cpu` as a fourth argument: the reference backend, the frame alone
    val reference = arguments.lift(3).contains("cpu")
    val ops =
      if (reference) new CpuOps else new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      def range(values: Array[Float]) =
        f"[${values.min}%.3f, ${values.max}%.3f], mean ${values.sum / values.length}%.3f"
      val video = MiniMaxH3VideoEncoder.open(ops, Paths.get(videoVae))
      try {
        val frame = Array.tabulate(64 * 96 * 3)(i =>
          (i % (96 * 3)).toFloat / (96 * 3) * 2 - 1
        )
        val latents = video.encode(Seq(frame), 64, 96, Posterior.Sample(42))
        println(
          s"frame: ${latents.length / (4 * 6 * 24)} latent frame(s), ${range(latents)}"
        )
        write("frame", latents)
        if (!reference) {
          val clip = video.encode(Seq.fill(22)(frame), 64, 96, Posterior.Mean)
          println(
            s"clip: ${clip.length / (4 * 6 * 24)} latent frames, ${range(clip)}"
          )
          write("clip", clip)
        }
      } finally video.close()
      if (reference) return
      val audio = MiniMaxH3AudioEncoder.open(ops, Paths.get(audioVae))
      try {
        val sine = Array.tabulate(32000)(i =>
          (0.5 * math.sin(2 * math.Pi * 440 * i / 32000)).toFloat
        )
        val latents = audio.encode(sine)
        println(
          s"sound: ${latents.length / audio.channels} latents, ${range(latents)}"
        )
        write("sound", latents)
      } finally audio.close()
    } finally ops.close()
  }
}
