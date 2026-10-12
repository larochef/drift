package drift.backend.postprocess

import drift.backend.images.PostProcessImages.*
import drift.backend.nativejobs.NativeJobs
import drift.backend.session.SessionManager
import drift.protocol.UpscaleJobRequest
import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, StandardCopyOption}
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Upscale by SeedVR2 (`specs/51-seedvr2-upscaling.md`) on a drift runner.
  *   - **A picture** is a tiled job like PiD's (`TiledJobs`): the source,
  *     padded to multiples of 16, is cut in tiles of at most
  *     `SeedVr2UpscaleRequest.MaxTile` target px — one tile for most pictures —
  *     each the runner's `upscale` job on its crop, colour-matched to that crop
  *     (`colourMatched`) and blended; it pauses and resumes as the others do.
  *   - **A video** goes whole, as one `upscale` job on a runner the job starts
  *     and stops: the runner reads the source's file and writes the result's, a
  *     batch of frames at a time, so its length is no limit. Its frame rate and
  *     soundtrack are kept. It does not pause; stopped, or failed after its
  *     first frames, it leaves the video of the frames made, which is saved
  *     with a warning saying so.
  */
final private[postprocess] class SeedVr2Upscale(
    jobs: PostProcessJobs,
    tiles: TiledJobs,
    sessionManager: SessionManager
) {

  def start(
      date: String,
      fileName: String,
      request: SeedVr2UpscaleRequest,
      resuming: Option[String] = None
  ): PostProcessJob =
    tiles.tiledJob(
      SeedVr2UpscaleRequest.Kind,
      date,
      fileName,
      request.runConfigurationId,
      request.runtimeId,
      resuming,
      architecture =>
        Option.unless(SeedVr2UpscaleRequest.runs(architecture))(
          s"'${architecture.label}' is not a SeedVR2 architecture"
        ),
      videos = true
    ) { (src, configuration, _, launch, _) =>
      val seeded = request.copy(seed =
        if (request.seed < 0) TiledJobs.drawSeed() else request.seed
      )
      if (launch.runtime.engine != RuntimeEngine.DriftRunner)
        Left(
          s"SeedVR2 runs on the drift runner only, and '${configuration.label}' is set to ${launch.runtime.engine}"
        )
      else if (!SeedVr2UpscaleRequest.Scales.contains(request.scale))
        Left(
          s"the scale is one of ${SeedVr2UpscaleRequest.Scales.mkString(", ")}"
        )
      else if (src.output.mimeType.startsWith("video/"))
        Right(
          jobs.start(SeedVr2UpscaleRequest.Kind, src)(job =>
            run(job, src, seeded, launch.runtime.id)
          )
        )
      else
        for {
          size <- imageSize(src.file).toRight("cannot read the source image")
          target <- SeedVr2UpscaleRequest.target(size, request.scale)
          reference <- referenceFor(src, size)
        } yield {
          val scale = request.scale
          val rows = SeedVr2UpscaleRequest.tilesFor(
            size,
            scale,
            request.tileSize,
            request.gridOffsetX,
            request.gridOffsetY
          )
          tiles.startTiles(
            SeedVr2UpscaleRequest.Kind,
            src,
            rows,
            PausedWork.SeedVr2(seeded),
            resuming,
            // A tile is in target pixels; the picture on screen is the source.
            rows.flatten.map(tile =>
              ImageRegion(
                tile.x / scale,
                tile.y / scale,
                tile.width / scale,
                tile.height / scale
              )
            )
          )(job =>
            tiles.runTiles(
              job,
              src,
              launch,
              reference,
              scale = scale,
              runConfigurationId = configuration.id,
              request = TileRequests.Upscales(input =>
                UpscaleJobRequest.picture(input.image, scale, seeded.seed)
              ),
              target = target,
              rows = rows,
              notes = List(
                s"source ${size._1}x${size._2} padded to ${reference.getWidth}x${reference.getHeight}, ×$scale, cropped back to ${target._1}x${target._2}"
              ),
              derivation =
                derivation(src, seeded, Some(target._1), Some(target._2)),
              // the tile's own crop of the source, at the tile's size, carries
              // the colours to keep
              correctTile = Some((restored, crop) =>
                colourMatched(
                  restored,
                  scaledCopy(crop, restored.getWidth, restored.getHeight)
                )
              ),
              resumed = resuming.isDefined
            )
          )
        }
    }

  /** The source padded to multiples of 16 by repeating its last row and column:
    * what the tiles are cut from.
    */
  private def referenceFor(
      src: PostProcessSource,
      size: (Int, Int)
  ): Either[String, BufferedImage] =
    try
      Option(ImageIO.read(src.file.toFile))
        .toRight("cannot decode the source image")
        .map { image =>
          val (width, height) = SeedVr2UpscaleRequest.referenceSizeOf(size)
          padded(image, width, height)
        }
    catch {
      case NonFatal(e) => Left(s"cannot read the source image: ${e.getMessage}")
    }

  /** A video, whole. A stop is asked of the runner, which ends its video on the
    * frames made (`NativeJobs.cancel`, through `waitingFor`); a runner that has
    * not answered it after `StopGraceMillis` is stopped itself, and the frames
    * are lost with it.
    */
  private def run(
      job: PostProcessJob,
      src: PostProcessSource,
      request: SeedVr2UpscaleRequest,
      runtimeId: String
  ): Unit = {
    var stopServer: () => Unit = () => ()
    val video = src.output.mimeType.startsWith("video/")
    // The runner writes here, beside the result's place: the gallery sees the
    // video once it is whole and has its name.
    val written = jobs.files
      .outputFileOf(job, src, "webm")
      .resolveSibling(s"${job.id}-0.webm.part")
    try {
      Files.createDirectories(jobs.files.logsRoot)
      Files.writeString(
        jobs.files.logFileOf(job),
        s"# drift ${job.kind} ${job.id} on runtime '$runtimeId'\n" +
          s"# ${if (video) "video" else "picture"} ${src.fileName} ×${request.scale}, seed ${request.seed}\n"
      )
      // Not stopped by a cancel at once: the runner is first asked to end
      // the job, which keeps the frames it has made.
      val outcome = ChosenServer
        .own(
          jobs,
          sessionManager,
          job,
          request.runConfigurationId,
          runtimeId,
          stoppedByCancel = false
        )
        .flatMap { server =>
          stopServer = server.stop
          var stopAskedAt = Option.empty[Long]
          if (jobs.isCancelled(job)) Left("stopped before it started")
          else
            NativeJobs.upscaleVideo(
              server.port,
              UpscaleJobRequest.video(
                src.file.toAbsolutePath.toString,
                written.toAbsolutePath.toString,
                request.scale,
                request.seed
              ),
              () => {
                if (jobs.isCancelled(job)) {
                  val now = System.currentTimeMillis()
                  val askedAt = stopAskedAt.getOrElse(now)
                  stopAskedAt = Some(askedAt)
                  if (now - askedAt > SeedVr2Upscale.StopGraceMillis)
                    server.stop()
                }
                server.exitCode()
              },
              nativeJob => jobs.waitingFor(job, server.port, nativeJob),
              jobs.report(job, _)
            )
        }
      jobs.doneWaiting(job)
      outcome match {
        case Left(reason) => jobs.fail(job, reason)
        case Right(made)  =>
          val outputFile = jobs.files.outputFileOf(job, src, made.format)
          Files.move(written, outputFile, StandardCopyOption.REPLACE_EXISTING)
          jobs.complete(
            job,
            src,
            outputFile,
            derivation(src, request, None, None).copy(warning =
              made.stopped.map(reason =>
                SeedVr2Upscale
                  .partial(made.frameCount, made.sourceFrames, reason)
              )
            ),
            OutputMedia(made.mimeType, made.format, made.fps, made.frameCount)
          )
      }
    } catch {
      case NonFatal(err) =>
        jobs.fail(job, s"the SeedVR2 upscale failed: ${err.getMessage}")
    } finally {
      jobs.doneWaiting(job)
      stopServer()
      // what a failed run or a killed runner left half written
      Files.deleteIfExists(written)
    }
  }

  private def derivation(
      src: PostProcessSource,
      request: SeedVr2UpscaleRequest,
      width: Option[Int],
      height: Option[Int]
  ) =
    Derivation(
      parentId = src.parent.id,
      parentDate = src.date,
      parentFileName = src.fileName,
      operation = SeedVr2UpscaleRequest.Kind,
      repeats = Some(request.scale),
      width = width,
      height = height,
      configurationId = Some(request.runConfigurationId),
      seed = Some(request.seed)
    )
}

private[postprocess] object SeedVr2Upscale {

  /** How long a runner asked to stop a video has to end it on the frames made:
    * it stops between two steps, a few seconds apart, then closes its encoder.
    */
  val StopGraceMillis: Long = 120_000

  /** The warning on a video that ends before its source does. */
  def partial(
      frames: Option[Int],
      sourceFrames: Option[Int],
      reason: String
  ): String = {
    val made = (frames, sourceFrames) match {
      case (Some(done), Some(all)) => s"the first $done of $all frames"
      case (Some(done), None)      => s"the first $done frames"
      case _                       => "the first frames"
    }
    s"partial: $made — ${
        if (reason == "stopped") "the upscale was stopped"
        else s"the upscale failed ($reason)"
      }"
  }
}
