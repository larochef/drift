package drift.backend.postprocess

import drift.backend.postprocess.PostProcessImages.*
import drift.backend.sdserver.{NativeJobs, NativeUpscale}
import drift.backend.session.SessionManager
import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.Files
import java.util.Base64
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Upscale by SeedVR2 (`specs/51-seedvr2-upscaling.md`) on a drift runner.
  *   - **A picture** is a tiled job like PiD's (`TiledJobs`): the source,
  *     padded to multiples of 16, is cut in tiles of at most
  *     `SeedVr2UpscaleRequest.MaxTile` target px — one tile for most pictures —
  *     each the runner's `upscale` job on its crop, colour-matched to that crop
  *     (`colourMatched`) and blended; it pauses and resumes as the others do.
  *   - **A video** goes whole, as one `upscale` job on a runner the job starts
  *     and stops: the runner cuts it itself (frames in batches, the VAE in
  *     tiles). It comes back as the runner encoded it, its frame rate and
  *     soundtrack kept, and does not pause.
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
          val rows = SeedVr2UpscaleRequest.tilesFor(size, scale)
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
                NativeUpscale(input.image, scale, seeded.seed)
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

  /** A video, whole. */
  private def run(
      job: PostProcessJob,
      src: PostProcessSource,
      request: SeedVr2UpscaleRequest,
      runtimeId: String
  ): Unit = {
    var stopServer: () => Unit = () => ()
    val video = src.output.mimeType.startsWith("video/")
    try {
      Files.createDirectories(jobs.files.logsRoot)
      Files.writeString(
        jobs.files.logFileOf(job),
        s"# drift ${job.kind} ${job.id} on runtime '$runtimeId'\n" +
          s"# ${if (video) "video" else "picture"} ${src.fileName} ×${request.scale}, seed ${request.seed}\n"
      )
      val outcome = sessionManager
        .startJobServer(
          request.runConfigurationId,
          Some(runtimeId),
          jobs.files.logFileOf(job),
          line => jobs.noteLine(job, line)
        )
        .flatMap { server =>
          stopServer = server.stop
          jobs.runsOn(job, server.stop)
          NativeJobs.upscale(
            server.port,
            NativeUpscale(
              Base64.getEncoder.encodeToString(Files.readAllBytes(src.file)),
              request.scale,
              request.seed
            ),
            () => server.exitCode,
            nativeJob => jobs.waitingFor(job, server.port, nativeJob)
          )
        }
      jobs.doneWaiting(job)
      outcome match {
        case Left(reason)   => jobs.fail(job, reason)
        case Right(Left(_)) =>
          jobs.fail(job, "the runner answered a video with a picture")
        case Right(Right(made)) =>
          val outputFile = jobs.files.outputFileOf(job, src, made.format)
          Files.write(outputFile, made.bytes)
          jobs.complete(
            job,
            src,
            outputFile,
            derivation(src, request, None, None),
            OutputMedia(made.mimeType, made.format, made.fps, made.frameCount)
          )
      }
    } catch {
      case NonFatal(err) =>
        jobs.fail(job, s"the SeedVR2 upscale failed: ${err.getMessage}")
    } finally {
      jobs.doneWaiting(job)
      stopServer()
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
