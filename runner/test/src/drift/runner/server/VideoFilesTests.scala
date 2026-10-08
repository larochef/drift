package drift.runner.server

import drift.runner.diffusion.{Media, Soundtrack, Video}

import java.awt.image.BufferedImage

import utest.*

/** A video file out of frames and a soundtrack of another length (`bugs/54`):
  * every frame is kept. Needs the `ffmpeg` the runner encodes with.
  */
object VideoFilesTests extends TestSuite {

  private val Fps = 30
  private val Rate = 8000

  /** 640 × 480, so the frames past a short soundtrack's end outgrow the pipe's
    * buffers: a small clip hid the failure.
    */
  private def clip(frames: Int, soundSeconds: Double) =
    Video(
      Seq.fill(frames)(new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB)),
      Fps,
      Some(
        Soundtrack(new Array[Float]((soundSeconds * Rate).toInt * 2), 2, Rate)
      )
    )

  private def decoded(video: Video) =
    VideoFiles.decode(VideoFiles.webm(video)) match {
      case Media.Clip(frames, _, soundtrack) => (frames.size, soundtrack)
      case other => throw new IllegalStateException(s"not a clip: $other")
    }

  val tests = Tests {

    test("a soundtrack shorter than the frames keeps every frame") {
      val (frames, soundtrack) = decoded(clip(90, 1.0))
      assert(frames == 90)
      assert(soundtrack.isDefined)
    }

    test("a soundtrack longer than the frames is cut at the last frame") {
      val (frames, soundtrack) = decoded(clip(30, 4.0))
      assert(frames == 30)
      val track = soundtrack.get
      val seconds = track.samples.length.toDouble / track.channels / track.rate
      assert(seconds < 1.5)
    }
  }
}
