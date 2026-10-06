package drift.backend.postprocess

import drift.backend.postprocess.PostProcessImages.*
import drift.backend.runtime.LaunchRuntime
import drift.backend.sdserver.{NativeJobs, NativeUpscale}
import drift.backend.session.SessionManager
import drift.shared.*

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.nio.file.Files
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** An edit made once and carried up (`specs/39-seamless-edit.md`): the part to
  * edit is reduced to what the model takes in one pass, so the model sees the
  * whole of what it is asked to change — a shirt, not a sleeve cut out of it —
  * and only what changed is brought back to the picture's size by an upscaler
  * and put into the picture.
  *
  * Measured the night of 2026-10-05: tile by tile, "the shirt is dark red" left
  * the sleeves beige, each tile deciding alone what the shirt was; made once on
  * the picture at a quarter of its size and carried up, the same edit is whole
  * in a third of the time.
  */
final private[postprocess] class EditCarry(
    jobs: PostProcessJobs,
    sessionManager: SessionManager
) {

  /** What the job is asked, resolved: the picture, the part of it to edit
    * (`area`, the whole picture or the window around `region`), the prompt, and
    * the two configurations it runs on.
    */
  case class Work(
      src: PostProcessSource,
      image: BufferedImage,
      area: ImageRegion,
      region: Option[ImageRegion],
      prompt: String,
      request: EditRequest,
      configuration: RunConfiguration,
      launch: LaunchRuntime,
      loras: ConfiguredLoras,
      sizeMultiple: Int,
      upscaler: RunConfiguration,
      upscalerLaunch: LaunchRuntime,
      derivation: Derivation
  )

  /** A server to ask, and how to let go of it. */
  private case class Server(
      port: Int,
      exitCode: () => Option[Int],
      stop: () => Unit
  )

  /** Runs the job to its end: the edited picture written as its result, or the
    * reason it could not be.
    */
  def run(job: PostProcessJob, work: Work): Unit = {
    import work.*
    val outputFile = jobs.files.outputFileOf(job, src)
    var stopServer: () => Unit = () => ()
    def kept(name: String, picture: BufferedImage): Unit =
      if (request.keepTiles) {
        Files.createDirectories(jobs.files.tilesDirOf(job))
        ImageIO.write(
          picture,
          "png",
          jobs.files.tilesDirOf(job).resolve(s"$name.png").toFile
        )
      }
    // The server of a ready session of the configuration, else one of the
    // job's own: a model already loaded is not loaded a second time beside
    // itself.
    def serverOf(
        configurationId: String,
        runtime: LaunchRuntime
    ): Either[String, Server] =
      sessionManager.list
        .find(session =>
          session.runConfigurationId == configurationId &&
            session.runtimeId.contains(runtime.runtime.id) &&
            session.status == SessionStatus.Ready && session.port.isDefined
        )
        .flatMap(_.port)
        .map(port => Right(Server(port, () => None, () => ())))
        .getOrElse(
          sessionManager
            .startJobServer(
              configurationId,
              Some(runtime.runtime.id),
              jobs.files.logFileOf(job),
              line => jobs.noteLine(job, line)
            )
            .map { server =>
              stopServer = server.stop
              jobs.runsOn(job, server.stop)
              Server(server.port, () => server.exitCode, server.stop)
            }
        )
    def decoded(bytes: Array[Byte], width: Int, height: Int) =
      Option(ImageIO.read(ByteArrayInputStream(bytes)))
        .toRight("the image returned cannot be decoded")
        .flatMap(returned =>
          Either.cond(
            returned.getWidth == width && returned.getHeight == height,
            returned,
            s"the image came back ${returned.getWidth}×${returned.getHeight}, not $width×$height"
          )
        )
    def cancelled: Either[String, Unit] =
      Either.cond(!jobs.isCancelled(job), (), "cancelled")
    try {
      val part = copyOf(
        image.getSubimage(area.x, area.y, area.width, area.height)
      )
      val pass = EditRequest.passOf(area.width, area.height, sizeMultiple)
      val reduced =
        if (pass.width == area.width && pass.height == area.height) part
        else scaledCopy(part, pass.width, pass.height)
      val by = area.width.toDouble / pass.width
      jobs.startLog(
        job,
        launch,
        notes = List(
          region.fold(
            s"source ${image.getWidth}x${image.getHeight}"
          )(selection =>
            s"selection ${selection.width}x${selection.height} at ${selection.x},${selection.y} of ${image.getWidth}x${image.getHeight}, edited through a ${area.width}x${area.height} window at ${area.x},${area.y}"
          ) + s"; edited in one pass at ${pass.width}x${pass.height}" +
            (if (pass.scale > 1)
               s", what changed carried up ×${pass.scale} by '${upscaler.label}'"
             else ", the picture's own size: nothing to carry up"),
          s"instruction: ${request.instructions.trim}"
        ) ++ TiledJobs.loraNote(loras.selections)
      )
      jobs.update(job.id)(_.copy(progress = Some(PostProcessProgress(0, 2))))
      kept("pass-input", reduced)
      // 1. The edit, once, where the model sees all of it.
      val edit: Either[String, EditComposite.Result] = for {
        server <- serverOf(configuration.id, launch)
        capabilities <- NativeJobs.imageCapabilities(server.port)
        (defaults, _) = capabilities
        bytes <-
          try
            NativeJobs.image(
              server.port,
              ImageGenerationParameters(
                prompt = prompt,
                clipSkip = defaults.clipSkip,
                width = pass.width,
                height = pass.height,
                seed = request.seed,
                refImages = List(dataUrl(reduced)),
                lora = loras.selections,
                sampleParams = {
                  val sampling = loras.sampling.over(defaults.sampleParams)
                  request.steps.fold(sampling)(steps =>
                    sampling.copy(sampleSteps = steps)
                  )
                },
                vaeTilingParams = defaults.vaeTilingParams
              ),
              server.exitCode,
              nativeJob => jobs.waitingFor(job, server.port, nativeJob)
            )
          finally {
            jobs.doneWaiting(job)
            server.stop()
            stopServer = () => ()
          }
        returned <- decoded(bytes, pass.width, pass.height)
        _ <- cancelled
      } yield {
        kept("pass-edit", returned)
        val result = EditComposite(reduced, returned)
        kept("pass-mask", result.mask)
        kept("pass-composite", result.image)
        jobs.appendLog(
          job,
          f"the edit changed ${result.changedShare * 100}%.1f%% of the pass" +
            (if (result.shift != (0, 0))
               s"; the model had moved the picture, put back by ${result.shift._1},${result.shift._2} px"
             else "") +
            (if (result.changedShare > Edit.RepaintShare)
               " — more than half: the model may have repainted rather than edited"
             else "")
        )
        result
      }
      val outcome: Either[String, BufferedImage] = edit.flatMap { result =>
        EditCarry.changedBox(result.mask) match {
          case None =>
            Left(
              "the model changed nothing the composite can tell from the source" +
                (if (region.isEmpty)
                   " — a change of a few pixels at the pass's size is not seen: draw a box around a small thing, and it is edited larger"
                 else "")
            )
          case Some(_) if pass.scale == 1 && reduced.eq(part) =>
            // Edited at its own size: the composite is the edited part.
            Right(result.image)
          case Some(box) =>
            // 2. What changed, back at the picture's size.
            val small = result.image.getSubimage(
              box.x,
              box.y,
              box.width,
              box.height
            )
            val wide =
              if (pass.scale == 1) Right(small)
              else upscaled(job, small, pass.scale, serverOf, work)
            wide.map { enlarged =>
              // Where the box lies in the part to edit, at its own size.
              val x = math.round(box.x * by).toInt.min(area.width - 1)
              val y = math.round(box.y * by).toInt.min(area.height - 1)
              val width =
                math.round(box.width * by).toInt.min(area.width - x).max(1)
              val height =
                math.round(box.height * by).toInt.min(area.height - y).max(1)
              val carried = EditComposite.carried(
                copyOf(part.getSubimage(x, y, width, height)),
                scaledCopy(enlarged, width, height),
                scaledCopy(
                  result.mask.getSubimage(box.x, box.y, box.width, box.height),
                  width,
                  height
                ),
                by
              )
              kept("carried-mask", carried.mask)
              val edited = copyOf(part)
              edited.setRGB(
                x,
                y,
                width,
                height,
                carried.image.getRGB(0, 0, width, height, null, 0, width),
                0,
                width
              )
              edited
            }
        }
      }
      outcome.flatMap { edited =>
        jobs.update(job.id)(_.copy(progress = None))
        // A selection bounds the change: what the model did beyond it, and
        // beyond the margin the paste feathers over, is not kept.
        val whole = region.fold {
          val result = copyOf(image)
          result.setRGB(
            area.x,
            area.y,
            area.width,
            area.height,
            edited.getRGB(0, 0, area.width, area.height, null, 0, area.width),
            0,
            area.width
          )
          result
        }(selection =>
          TileBlending.paste(
            image,
            edited,
            area,
            selection,
            reach = Some(request.selectionMargin)
          )
        )
        Either.cond(
          ImageIO.write(whole, "png", outputFile.toFile),
          (),
          "no PNG writer available"
        )
      } match {
        case Left(reason) =>
          Files.deleteIfExists(outputFile)
          jobs.fail(job, reason)
        case Right(()) => jobs.complete(job, src, outputFile, derivation)
      }
    } catch {
      case NonFatal(err) =>
        Files.deleteIfExists(outputFile)
        jobs.fail(job, s"the edit failed: ${err.getMessage}")
    } finally {
      jobs.doneWaiting(job)
      stopServer()
    }
  }

  /** `small` ×`scale` by the upscaler, in the tiles a SeedVR2 upscale of a
    * picture runs (`SeedVr2UpscaleRequest.tilesFor`), each given its source's
    * colours back, blended over their overlaps.
    */
  private def upscaled(
      job: PostProcessJob,
      small: BufferedImage,
      scale: Int,
      serverOf: (String, LaunchRuntime) => Either[String, Server],
      work: Work
  ): Either[String, BufferedImage] = {
    val rows =
      SeedVr2UpscaleRequest.tilesFor((small.getWidth, small.getHeight), scale)
    val tiles = rows.flatten
    jobs.update(job.id)(
      _.copy(progress = Some(PostProcessProgress(1, tiles.size + 1)))
    )
    jobs.appendLog(
      job,
      s"what changed is ${small.getWidth}x${small.getHeight} of the pass: ${tiles.size} tile(s) to upscale ×$scale"
    )
    serverOf(work.upscaler.id, work.upscalerLaunch).flatMap { server =>
      try
        tiles.zipWithIndex
          .foldLeft[Either[String, Map[Tiling.Tile, BufferedImage]]](
            Right(Map.empty)
          ) { case (done, (tile, index)) =>
            done.flatMap { images =>
              val crop = copyOf(
                small.getSubimage(
                  tile.x / scale,
                  tile.y / scale,
                  tile.width / scale,
                  tile.height / scale
                )
              )
              (if (jobs.isCancelled(job)) Left("cancelled")
               else
                 try
                   NativeJobs
                     .upscale(
                       server.port,
                       NativeUpscale(dataUrl(crop), scale, work.request.seed),
                       server.exitCode,
                       nativeJob => jobs.waitingFor(job, server.port, nativeJob)
                     )
                     .flatMap(
                       _.left.toOption
                         .toRight("the runner answered a tile with a video")
                     )
                 finally jobs.doneWaiting(job))
                .flatMap(bytes =>
                  Option(ImageIO.read(ByteArrayInputStream(bytes)))
                    .toRight("the tile returned cannot be decoded")
                )
                .flatMap(restored =>
                  Either.cond(
                    restored.getWidth == tile.width &&
                      restored.getHeight == tile.height,
                    restored,
                    s"the tile came back ${restored.getWidth}×${restored.getHeight}, not ${tile.width}×${tile.height}"
                  )
                )
                .left
                .map(reason => s"tile ${index + 1} of ${tiles.size}: $reason")
                .map { restored =>
                  jobs.update(job.id)(
                    _.copy(progress =
                      Some(PostProcessProgress(index + 2, tiles.size + 1))
                    )
                  )
                  // the tile's own crop, at the tile's size, carries the
                  // colours to keep
                  images + (tile -> colourMatched(
                    restored,
                    scaledCopy(crop, restored.getWidth, restored.getHeight)
                  ))
                }
            }
          }
          .map(images =>
            TileBlending.blend(
              rows,
              small.getWidth * scale,
              small.getHeight * scale,
              images
            )
          )
      finally server.stop()
    }
  }
}

private[postprocess] object EditCarry {

  /** How far around what changed the upscaler is given the pass: what it sees
    * beside the object it restores, in the pass's own pixels.
    */
  val BoxMargin: Int = 48

  /** The box of the pass that holds everything `mask` marks, grown by
    * `BoxMargin` and out to multiples of 16, kept inside the pass — none when
    * the mask marks nothing.
    */
  def changedBox(mask: BufferedImage): Option[ImageRegion] = {
    val width = mask.getWidth
    val height = mask.getHeight
    val pixels = mask.getRGB(0, 0, width, height, null, 0, width)
    var left = width
    var top = height
    var right = -1
    var bottom = -1
    for (index <- pixels.indices if (pixels(index) & 0xff) > 0) {
      val x = index % width
      val y = index / width
      if (x < left) left = x
      if (x > right) right = x
      if (y < top) top = y
      if (y > bottom) bottom = y
    }
    Option.when(right >= 0) {
      val x = ((left - BoxMargin).max(0)) / 16 * 16
      val y = ((top - BoxMargin).max(0)) / 16 * 16
      val endX =
        Tiling.roundUp((right + 1 + BoxMargin).min(width), 16).min(width)
      val endY =
        Tiling.roundUp((bottom + 1 + BoxMargin).min(height), 16).min(height)
      ImageRegion(x, y, endX - x, endY - y)
    }
  }
}
