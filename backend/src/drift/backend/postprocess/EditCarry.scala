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

  /** One pass of the edit model: the part of the picture it was shown (`seen`),
    * that part as it is and reduced to the pass, how many times smaller the
    * pass is (`by`), and what the composite kept of the edit.
    */
  private case class Passed(
      seen: ImageRegion,
      part: BufferedImage,
      reduced: BufferedImage,
      pass: EditRequest.Pass,
      by: Double,
      result: EditComposite.Result
  )

  /** The part of the picture the model was shown, edited, at its own size. */
  private case class Carried(seen: ImageRegion, edited: BufferedImage)

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
    // What the gallery entry says of the result: the last pass's repaint, and
    // a change the window cut.
    var repainted = Option.empty[String]
    var warnings = Vector.empty[String]
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
    // What a selection keeps of the change: every changed part that touches
    // it, as far as it goes, or only what lies within it.
    val following = region.filter(_ => request.beyondSelection)
    // 1. The edit, once, where the model sees all of it: `seen` is the part of
    // the picture it is shown.
    def passOn(
        seen: ImageRegion,
        again: Option[String]
    ): Either[String, Passed] = {
      val part = copyOf(
        image.getSubimage(seen.x, seen.y, seen.width, seen.height)
      )
      val pass = EditRequest.passOf(seen.width, seen.height, sizeMultiple)
      val reduced =
        if (pass.width == seen.width && pass.height == seen.height) part
        else scaledCopy(part, pass.width, pass.height)
      val by = seen.width.toDouble / pass.width
      val carrying =
        if (pass.scale > 1)
          s", what changed carried up ×${pass.scale} by '${upscaler.label}'"
        else ", the picture's own size: nothing to carry up"
      again.fold(
        jobs.startLog(
          job,
          launch,
          notes = List(
            region.fold(
              s"source ${image.getWidth}x${image.getHeight}"
            )(selection =>
              s"selection ${selection.width}x${selection.height} at ${selection.x},${selection.y} of ${image.getWidth}x${image.getHeight}, edited through a ${seen.width}x${seen.height} window at ${seen.x},${seen.y}"
            ) + s"; edited in one pass at ${pass.width}x${pass.height}$carrying",
            s"instruction: ${request.instructions.trim}"
          ) ++ TiledJobs.loraNote(loras.selections)
        )
      )(reason =>
        jobs.appendLog(
          job,
          s"$reason: edited again through a ${seen.width}x${seen.height} window at ${seen.x},${seen.y}, in one pass at ${pass.width}x${pass.height}$carrying"
        )
      )
      jobs.update(job.id)(_.copy(progress = Some(PostProcessProgress(0, 2))))
      kept("pass-input", reduced)
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
        val result = EditComposite(
          reduced,
          returned,
          following.map(selection =>
            ImageRegion(
              ((selection.x - seen.x) / by).toInt.max(0),
              ((selection.y - seen.y) / by).toInt.max(0),
              math.ceil(selection.width / by).toInt.max(1),
              math.ceil(selection.height / by).toInt.max(1)
            )
          )
        )
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
      edit.map(Passed(seen, part, reduced, pass, by, _))
    }
    // A selection is first shown to the model at the picture's own size when it
    // fits one pass — nothing to carry up, every pixel kept is the picture's
    // own. Two things send it back for a wider look, twice as much each time,
    // up to the whole picture: the model changed nothing — handed a wrist and
    // a watch it leaves the watch on, with the arm and the table around them
    // it takes it off — or the change runs out of what it was shown — hair cut
    // short that hangs below the window, which the model could not cut and the
    // picture would keep from the window's edge down.
    def wider(passed: Passed): Option[(ImageRegion, String)] =
      region
        .flatMap { selection =>
          val nothing = EditCarry.changedBox(passed.result.mask).isEmpty
          val cut = following.nonEmpty &&
            EditCarry.cutBy(passed.result.mask, passed.seen, image)
          Option
            .when(nothing)("the model changed nothing")
            .orElse(
              Option.when(cut)(
                "the change reaches the edge of what the model was shown"
              )
            )
            .map(reason =>
              (
                Tiling.window(
                  selection,
                  image.getWidth,
                  image.getHeight,
                  2 * math.max(passed.seen.width, passed.seen.height),
                  request.selectionMargin,
                  multiple = sizeMultiple
                ),
                reason
              )
            )
        }
        .filter((seen, _) =>
          seen.width > passed.seen.width || seen.height > passed.seen.height
        )
    def changed(passed: Passed): Boolean =
      EditCarry.changedBox(passed.result.mask).nonEmpty
    def settled(passed: Passed, looks: Int): Either[String, Passed] =
      wider(passed)
        .filter(_ => looks > 0)
        .fold[Either[String, Passed]](Right(passed))((seen, reason) =>
          passOn(seen, Some(reason)).flatMap(next =>
            // A thing the model drew at the picture's size can be too fine to
            // tell from the source once the window is reduced — a necklace's
            // chain: the edit already made is the one to keep.
            if (changed(passed) && !changed(next)) {
              jobs.appendLog(
                job,
                "seen wider, nothing the composite can tell from the source: the edit made before is kept"
              )
              Right(passed)
            } else settled(next, looks - 1)
          )
        )
    try {
      val passed =
        passOn(area, None).flatMap(settled(_, EditCarry.WiderLooks))
      val outcome: Either[String, Carried] = passed.flatMap {
        case Passed(seen, part, reduced, pass, by, result) =>
          repainted = Edit.repaintWarning(result.changedShare)
          if (following.nonEmpty && EditCarry.cutBy(result.mask, seen, image))
            warnings = warnings :+
              "The change runs past what the model was shown: it stops at the edge of that window."
          (EditCarry.changedBox(result.mask) match {
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
                val x = math.round(box.x * by).toInt.min(seen.width - 1)
                val y = math.round(box.y * by).toInt.min(seen.height - 1)
                val width =
                  math.round(box.width * by).toInt.min(seen.width - x).max(1)
                val height =
                  math.round(box.height * by).toInt.min(seen.height - y).max(1)
                val carried = EditComposite.carried(
                  copyOf(part.getSubimage(x, y, width, height)),
                  scaledCopy(enlarged, width, height),
                  scaledCopy(
                    result.mask
                      .getSubimage(box.x, box.y, box.width, box.height),
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
          }).map(Carried(seen, _))
      }
      outcome.flatMap { case Carried(seen, edited) =>
        jobs.update(job.id)(_.copy(progress = None))
        // Outside what changed the edited part is the source's own pixels, so
        // it goes back whole — unless the selection is to bound the change:
        // then what the model did beyond it, and beyond the margin the paste
        // feathers over, is not kept.
        val whole = region
          .filterNot(_ => request.beyondSelection)
          .fold {
            val result = copyOf(image)
            result.setRGB(
              seen.x,
              seen.y,
              seen.width,
              seen.height,
              edited.getRGB(0, 0, seen.width, seen.height, null, 0, seen.width),
              0,
              seen.width
            )
            result
          }(selection =>
            TileBlending.paste(
              image,
              edited,
              seen,
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
        case Right(()) =>
          jobs.complete(
            job,
            src,
            outputFile,
            derivation.copy(warning =
              Option(repainted.toVector ++ warnings)
                .filter(_.nonEmpty)
                .map(_.mkString(" "))
            )
          )
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

  /** How many times a selection's edit is made again through a window twice as
    * large: from one pass at the picture's size (1536) to 6144, which is a
    * whole 4k picture, or most of an 8k one.
    */
  val WiderLooks: Int = 2

  /** How much of a window's side, in the pass's pixels, a change must cover to
    * count as running out of it: a stray strand of hair the model moved, a few
    * pixels wide, does not send the edit back for a wider look.
    */
  val CutLength: Int = 48

  /** Whether the change `mask` marks runs out of `seen`, the part of `image`
    * the model was shown, on a side where the picture goes on: there the edit
    * stops at the window's edge and the picture is as it was beyond it.
    */
  def cutBy(
      mask: BufferedImage,
      seen: ImageRegion,
      image: BufferedImage
  ): Boolean = {
    val width = mask.getWidth
    val height = mask.getHeight
    def marked(x: Int, y: Int): Boolean = (mask.getRGB(x, y) & 0xff) > 127
    def column(x: Int): Boolean =
      (0 until height).count(marked(x, _)) >= CutLength
    def row(y: Int): Boolean =
      (0 until width).count(marked(_, y)) >= CutLength
    (seen.x > 0 && column(0)) ||
    (seen.x + seen.width < image.getWidth && column(width - 1)) ||
    (seen.y > 0 && row(0)) ||
    (seen.y + seen.height < image.getHeight && row(height - 1))
  }

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
