package drift.runner.server

import drift.runner.diffusion.SeedVr2Pipeline

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}

/** One pass of an upscaler over a clip's frames, as they are asked for: what
  * `VideoUpscale` runs, SeedVR2 on the server and a plain resize in the tests.
  */
trait UpscalePass {

  /** How many steps `frames` frames brought to `width × height` take. */
  def steps(frames: Int, width: Int, height: Int): Int

  /** `count` `frames` at `width × height`, `tick` after each step. */
  def upscaled(
      frames: Iterator[BufferedImage],
      count: Int,
      width: Int,
      height: Int,
      tick: () => Unit
  ): Iterator[BufferedImage]
}

/** A video file an upscale made: `frames` frames written of the source's
  * `sourceFrames` (as probed), at the source's `rate`. `stopped` is what ended
  * it before the source's last frame, if anything did: the file is then the
  * shorter video of the frames made until there.
  */
final case class UpscaledVideo(
    frames: Int,
    sourceFrames: Int,
    width: Int,
    height: Int,
    rate: FrameRate,
    stopped: Option[Throwable]
)

/** A video upscaled from a file into a file, whatever it lasts: the source's
  * frames are read one at a time (`ClipFrames`), restored a batch at a time,
  * every pass feeding the next as it goes, and written to the encoder as they
  * are made (`WebmWriter`). Nothing holds the video: a frame being read, the
  * batches being restored and a frame being written are all that is on the
  * heap. The soundtrack is the source's, taken from its file by the encoder,
  * and the frame rate the source's own.
  */
object VideoUpscale {

  /** `source` upscaled into `output` at the size `target` gives for the
    * source's (even sides), in `SeedVr2Pipeline.passes`' passes of `pass`;
    * `starting(clip, size, passes)` once those are known. `progress(done, of)`
    * counts the steps of every pass together, each weighed by its pass' pixels
    * over the first one's, and is where a run is stopped: what it throws ends
    * the run, as a pass' own failure does. The video is then ended on the
    * frames written and kept, with the reason in `stopped`; with no frame
    * written there is no video and the failure is thrown.
    */
  def run(
      source: Path,
      output: Path,
      target: (Int, Int) => (Int, Int),
      pass: UpscalePass,
      progress: (Int, Int) => Unit,
      starting: (ClipFrames, (Int, Int), Int) => Unit
  ): UpscaledVideo = {
    val probed = VideoFiles.probe(source)
    val clip = VideoFiles.frames(
      source,
      probed.frames.getOrElse(
        throw new IllegalArgumentException("a sound has no frames")
      )
    )
    try {
      val (width, height) = target(clip.width, clip.height)
      val sizes =
        SeedVr2Pipeline.passes((clip.width, clip.height), (width, height))
      starting(clip, (width, height), sizes.size)
      // A step of a later pass works on that many times the pixels of the
      // first one's, and counts for as many: the bar then moves with the time
      // spent, and what is left can be told from it.
      val firstArea = sizes.head._1.toLong * sizes.head._2
      val weights = sizes.map((w, h) =>
        math.max(1L, math.round(w.toLong * h / firstArea.toDouble)).toInt
      )
      val steps = sizes
        .zip(weights)
        .map { case ((w, h), weight) => pass.steps(clip.count, w, h) * weight }
        .sum
      var done = 0
      // the probe's count may be short of the frames read: the bar stays full
      def tick(weight: Int)(): Unit = {
        done += weight
        progress(math.min(done, steps), steps)
      }
      val encoder = VideoFiles.writer(
        output,
        width,
        height,
        clip.rate,
        probed.sound.map(VideoFiles.soundOf(source, _))
      )
      val stopped =
        try {
          progress(0, steps)
          sizes
            .zip(weights)
            .foldLeft(clip: Iterator[BufferedImage]) {
              case (frames, ((passWidth, passHeight), weight)) =>
                pass.upscaled(
                  frames,
                  clip.count,
                  passWidth,
                  passHeight,
                  tick(weight)
                )
            }
            .foreach(encoder.write)
          None
        } catch { case reason: Throwable => Some(reason) }
      stopped match {
        case _ if encoder.frames == 0 =>
          encoder.abandon()
          throw stopped.getOrElse(
            new IllegalArgumentException("a video without frames")
          )
        case _ =>
          try encoder.finish(cutSound = stopped.isDefined)
          catch {
            case failure: Throwable =>
              Files.deleteIfExists(output)
              stopped.foreach(failure.addSuppressed)
              throw failure
          }
          UpscaledVideo(
            encoder.frames,
            clip.count,
            width,
            height,
            clip.rate,
            stopped
          )
      }
    } finally clip.close()
  }
}
