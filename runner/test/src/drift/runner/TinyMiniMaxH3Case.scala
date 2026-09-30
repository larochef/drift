package drift.runner

import drift.runner.models.{MiniMaxH3, MiniMaxH3Layout, MiniMaxH3Vae}
import drift.runner.formats.SafetensorsModel
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny MiniMax H3 (`fixtures/tiny_diffusion.py`) on a backend: the
  * transformer's velocities over the t2va layout the runner builds, and the
  * video VAE's frames, compared with diffusers'.
  */
object TinyMiniMaxH3Case {

  /** The largest difference between the runner's (t, h, w) of each row and
    * diffusers' layout's.
    */
  def layoutError(): Double =
    Fixtures.withSafetensors("tiny/minimax_h3/expected.safetensors") { golden =>
      val expected = golden("positions").decode()
      val actual = layout(golden).positions
      expected.indices.map(i => math.abs(actual(i) - expected(i))).max
    }

  private def layout(golden: SafetensorsModel) = {
    val Seq(text, frames, height, width, audio) =
      golden("shape").decode().map(_.toInt).toSeq
    MiniMaxH3Layout(text, frames, height, width, audio)
  }

  /** The worst error of the video and the audio velocities, each relative to
    * its largest magnitude.
    */
  def velocityError(ops: Ops): Double = {
    val model =
      MiniMaxH3.open(ops, Fixtures.path("tiny/minimax_h3/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/minimax_h3/expected.safetensors") {
        golden =>
          val c = model.config
          val laid = layout(golden)
          val encoded = ops.fromFloats(
            Shape.of(laid.textTokens, c.textWidth),
            golden("text").decode()
          )
          val text =
            ops.allocate(DType.F32, Shape.of(laid.textTokens, c.hidden))
          model.encodeText(encoded, text)
          val video = ops.fromFloats(
            Shape.of(laid.videoRows, c.videoWidth),
            golden("video_rows").decode()
          )
          val audio = ops.fromFloats(
            Shape.of(laid.audioRows, c.audioWidth),
            golden("audio_rows").decode()
          )
          val (videoOut, audioOut) = (
            ops.allocate(DType.F32, video.shape),
            ops.allocate(DType.F32, audio.shape)
          )
          val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
          model.velocity(
            laid,
            text,
            video,
            audio,
            videoTime,
            audioTime,
            videoOut,
            audioOut
          )
          def error(actual: Array[Float], expected: Array[Float]) = {
            val scale = expected.map(math.abs).max.toDouble
            expected.indices
              .map(i => math.abs(actual(i) - expected(i)))
              .max / scale
          }
          math.max(
            error(ops.toFloats(videoOut), golden("video").decode()),
            error(ops.toFloats(audioOut), golden("audio").decode())
          )
      }
    finally model.close()
  }

  /** The worst error of the decoded frames (in [0, 1]): two temporal chunks,
    * four spatial tiles each.
    */
  def framesError(ops: Ops): Double = {
    val decoder = MiniMaxH3Vae.open(
      ops,
      Fixtures.path("tiny/minimax_h3_vae/model.safetensors"),
      tilePixels = 64,
      tileOverlap = 16
    )
    try
      Fixtures.withSafetensors("tiny/minimax_h3_vae/expected.safetensors") {
        golden =>
          val latents = golden("latents").decode()
          val expected = golden("frames").decode()
          val frames = scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
          decoder.decode(latents, 12, 6, 7, frames += _)
          val frameValues = 96 * 112 * 3
          assert(
            frames.size == 39 && decoder.frameCount(12) == 39,
            s"${frames.size} frames, not 39"
          )
          frames.zipWithIndex.map { (frame, f) =>
            frame.indices
              .map(i => math.abs(frame(i) - expected(f * frameValues + i)))
              .max
              .toDouble
          }.max
      }
    finally decoder.close()
  }
}
