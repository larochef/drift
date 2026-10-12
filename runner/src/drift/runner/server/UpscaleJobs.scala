package drift.runner.server

import drift.protocol.*
import drift.protocol.JobEndpoints.Asked
import drift.runner.diffusion.*
import drift.runner.server.JobAnswers.{failed, Answer}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.file.*
import java.util.Base64
import javax.imageio.ImageIO
import scala.util.Random

/** `upscale` requests, a picture's or a video's, read into jobs on `queue`. */
final private[server] class UpscaleJobs(
    options: ImageOptions,
    queue: JobQueue,
    results: JobResults
) {

  /** `POST /sdcpp/v1/upscale`, the runner's own job, restored at
    * `width × height`, or at `scale` times the source's size (4 unless said);
    * `seed`, `batch` and `overlap` as `SeedVr2Options` has them.
    *   - **A picture** is `source`, base64 or a data URL, and comes back as
    *     `img_gen`'s result does.
    *   - **A video** is a file, `source_file`, and is written to another,
    *     `output_file` (a webm, its frame rate and soundtrack the source's):
    *     drift and the runner share the machine, and a video of any length goes
    *     through without being held (`VideoUpscale`). The result names the file
    *     (`file`), its frames (`frame_count`, of `source_frames`) and its rate
    *     (`frame_rate`, exact; `fps`, rounded). A job stopped or failed after
    *     its first frames ends `cancelled` or `failed` with that result all the
    *     same: the file is the video of the frames made.
    */
  def submitUpscale(
      pipeline: SeedVr2Pipeline,
      asked: Asked[UpscaleJobRequest]
  ): Answer = {
    val body = asked match {
      case Right(read)   => read
      case Left(problem) =>
        return failed(400, problem)
    }
    def text(value: Option[String]) = value.filter(_.nonEmpty)
    val scale = body.scale.getOrElse(SeedVr2Pipeline.DefaultScale)
    def even(side: Int) = side / 2 * 2
    def target(width: Int, height: Int) = (
      even(body.width.getOrElse(width * scale)),
      even(body.height.getOrElse(height * scale))
    )
    val seed = body.seed
      .filter(_ >= 0)
      .getOrElse(
        if (options.seed >= 0) options.seed
        else Random.nextInt(Int.MaxValue).toLong
      )
    val upscaleOptions =
      try
        SeedVr2Options.Default.copy(
          batch = body.batch.getOrElse(SeedVr2Options.Default.batch),
          overlap = body.overlap.getOrElse(SeedVr2Options.Default.overlap)
        )
      catch {
        case error: IllegalArgumentException =>
          return failed(400, error.getMessage)
      }
    (text(body.source), text(body.sourceFile), text(body.outputFile)) match {
      case (Some(encoded), None, None) =>
        submitPictureUpscale(
          pipeline,
          encoded,
          target,
          seed,
          upscaleOptions
        )
      case (None, Some(source), Some(output)) =>
        submitVideoUpscale(
          pipeline,
          Paths.get(source),
          Paths.get(output),
          target,
          seed,
          upscaleOptions
        )
      case _ =>
        failed(
          400,
          "a picture as `source`, or a video as `source_file` and `output_file`"
        )
    }
  }

  /** A video's `upscale` job, from a file into a file (`VideoUpscale`). */
  private def submitVideoUpscale(
      pipeline: SeedVr2Pipeline,
      source: Path,
      output: Path,
      target: (Int, Int) => (Int, Int),
      seed: Long,
      upscaleOptions: SeedVr2Options
  ): Answer = {
    if (!Files.isReadable(source))
      return failed(400, s"no readable file at $source")
    if (!Option(output.getParent).exists(Files.isDirectory(_)))
      return failed(400, s"no folder to write $output in")
    val pass = new UpscalePass {
      def steps(frames: Int, width: Int, height: Int): Int =
        pipeline.steps(frames, width, height, upscaleOptions)
      def upscaled(
          frames: Iterator[java.awt.image.BufferedImage],
          count: Int,
          width: Int,
          height: Int,
          tick: () => Unit
      ): Iterator[java.awt.image.BufferedImage] =
        pipeline.upscaled(
          frames,
          count,
          width,
          height,
          seed,
          upscaleOptions,
          tick
        )
    }
    queue.enqueue(
      "upscale",
      () => {
        ProgressReport.pass(None)
        val startedAt = System.nanoTime()
        val made = VideoUpscale.run(
          source,
          output,
          target,
          pass,
          results.stepPrinter(),
          (clip, size, passes) =>
            println(
              s"upscaling ${clip.count} frames of ${clip.width} × ${clip.height} to ${size._1} × ${size._2}${
                  if (passes > 1) s" in $passes passes" else ""
                } (seed $seed)"
            )
        )
        val result = NativeJobResult(
          outputFormat = "webm",
          mimeType = Some("video/webm"),
          fps = Some(made.rate.rounded),
          frameRate = Some(made.rate.toString),
          frameCount = Some(made.frames),
          sourceFrames = Some(made.sourceFrames),
          file = Some(output.toString)
        )
        made.stopped match {
          case Some(reason) =>
            println(
              s"stopped after ${made.frames} frames: the video of those is kept"
            )
            throw new PartialResult(result, reason)
          case None =>
            println(
              f"generate_image completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
            )
            result
        }
      }
    )
  }

  /** A picture's `upscale` job: `encoded` is base64 or a data URL. */
  private def submitPictureUpscale(
      pipeline: SeedVr2Pipeline,
      encoded: String,
      target: (Int, Int) => (Int, Int),
      seed: Long,
      upscaleOptions: SeedVr2Options
  ): Answer = {
    val bytes = Base64.getMimeDecoder.decode(
      if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
      else encoded
    )
    // a picture as it is: `VideoFiles` would bring a large one down
    val frames = Option(ImageIO.read(new ByteArrayInputStream(bytes)))
      .fold(VideoFiles.decode(bytes))(Media.Still(_)) match {
      case Media.Still(image) => Seq(image)
      case _                  =>
        return failed(
          400,
          "`source` is a picture: a video goes as `source_file` and `output_file`"
        )
    }
    val (width, height) = target(frames.head.getWidth, frames.head.getHeight)
    queue.enqueue(
      "upscale",
      () => {
        ProgressReport.pass(None)
        println(
          s"upscaling 1 frame of ${frames.head.getWidth} × ${frames.head.getHeight} to $width × $height (seed $seed)"
        )
        val startedAt = System.nanoTime()
        val sizes = SeedVr2Pipeline.passes(
          (frames.head.getWidth, frames.head.getHeight),
          (width, height)
        )
        val restored = sizes.zipWithIndex.foldLeft(frames) {
          case (current, ((passWidth, passHeight), index)) =>
            if (sizes.size > 1)
              println(
                s"pass ${index + 1} of ${sizes.size}: to $passWidth × $passHeight"
              )
            pipeline.upscale(
              current,
              passWidth,
              passHeight,
              seed,
              upscaleOptions,
              results.stepPrinter()
            )
        }
        println(
          f"generate_image completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
        )
        val png = new ByteArrayOutputStream()
        ImageIO.write(restored.head, "png", png)
        NativeJobResult(
          outputFormat = "png",
          images = List(
            NativeJobImage(0, Base64.getEncoder.encodeToString(png.toByteArray))
          )
        )
      }
    )
  }
}
