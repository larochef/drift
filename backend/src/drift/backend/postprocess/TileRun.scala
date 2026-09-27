package drift.backend.postprocess

import drift.backend.runtime.LaunchRuntime
import drift.backend.sdserver.NativeJobs
import drift.backend.session.SessionManager
import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.*
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** One run of a tiled job (`specs/26-tiled-pid.md`, `specs/27-redraw.md`,
  * `specs/39-seamless-edit.md`): the server its tiles run on, the tiles
  * themselves — cut, sent, judged, finished and painted in, one after another —
  * and the picture they make, blended and written where the job's result
  * belongs.
  *
  * A run, not a job: `TiledJobs` decides what a job is and records it, and a
  * job paused and carried on later is another `TileRun` over the same tiles,
  * the ones it already has found on disk (`specs/40-pause-and-resume.md`).
  */
final private[postprocess] class TileRun(
    jobs: PostProcessJobs,
    sessionManager: SessionManager,
    job: PostProcessJob,
    src: PostProcessSource,
    launch: LaunchRuntime,
    reference: BufferedImage,
    scale: Int,
    runConfigurationId: String,
    request: (
        GenerationDefaults,
        Map[String, Boolean],
        TileWindow.TileInput
    ) => Either[String, ImageGenerationParameters],
    target: (Int, Int),
    rows: List[List[Tiling.Tile]],
    notes: List[String],
    derivation: Derivation,
    /** What each tile's crop goes through before the model sees it — where a
      * redraw softens away the artifacts it is meant to remove.
      */
    prepareTile: BufferedImage => BufferedImage,
    finish: BufferedImage => BufferedImage,
    keepTiles: Boolean,
    finishTile: Option[TileWindow.FinishTile],
    context: Option[TileWindow.TileContext],
    /** Whether this run carries a paused job on, which changes what the log
      * says and nothing else: the tiles it has are found on disk
      * (`specs/40-pause-and-resume.md`).
      */
    resumed: Boolean
) {

  val outputFile = jobs.files.outputFileOf(job, src)
  val tiles = rows.flatten
  // What a tile costs is measured on this run alone: a resumed job would
  // otherwise average in the hours it spent paused.
  val runStarted = System.currentTimeMillis()
  var doneThisRun = 0
  // What becomes of each tile before it is painted in, when tiles run in
  // order: an edit's composite, or the tile as it came back.
  val finishing = finishTile.orElse(
    Option.when(context.isDefined)(TileWindow.keepReturned)
  )
  // The picture tiles run in order are painted into and cut from.
  val picture = finishing.map(_ => PostProcessImages.copyOf(reference))
  // What the model is handed for a tile: the tile, or the tile with its
  // context around it.
  def windowOf(tile: Tiling.Tile): Tiling.Tile =
    context.fold(tile)(
      TileWindow.windowOf(tile, reference.getWidth, reference.getHeight, _)
    )
  // How far each tile overlaps the ones painted before it: its neighbour in
  // the row and the row above, as `TileBlending.blend` ramps them.
  val overlaps: Map[Tiling.Tile, (Int, Int)] =
    rows.zipWithIndex.flatMap { (row, rowIndex) =>
      val top = rows
        .lift(rowIndex - 1)
        .map(above => above.head.y + above.head.height - row.head.y)
        .getOrElse(0)
        .max(0)
      row.zipWithIndex.map { (tile, column) =>
        val left = row
          .lift(column - 1)
          .map(before => before.x + before.width - tile.x)
          .getOrElse(0)
          .max(0)
        tile -> (left, top)
      }
    }.toMap
  // What the tiles make so far, painted on a thread of its own for the gallery
  // to show while the job runs (`LivePicture`).
  val live = LivePicture(
    job.id,
    reference,
    scale,
    overlaps,
    target,
    finish,
    count => jobs.update(job.id)(_.copy(paintedTiles = Some(count)))
  )
  // Whether a pause handed the picture over to be kept on disk.
  var pictureKept = false
  def tileFile(index: Int, part: String) =
    if (keepTiles)
      jobs.files.tilesDirOf(job).resolve(f"tile-${index + 1}%02d-$part.png")
    else
      jobs.files.logsRoot.resolve(
        s"postprocess-${job.id}-tile-$index-$part.png"
      )
  def tileLine(index: Int, tile: Tiling.Tile) = {
    val window = windowOf(tile)
    s"tile ${index + 1} of ${tiles.size} at ${tile.x},${tile.y} ${tile.width}x${tile.height}" +
      Option
        .when(window != tile)(
          s", seen through a ${window.width}x${window.height} window at ${window.x},${window.y}"
        )
        .getOrElse("")
  }
  // On the job's runtime only: that is the build the job was checked
  // against (a PiD needs one that keeps its reference's size).
  val session = sessionManager.list.find(session =>
    session.runConfigurationId == runConfigurationId &&
      session.runtimeId.contains(launch.runtime.id) &&
      session.status == SessionStatus.Ready && session.port.isDefined
  )
  var stopServer: () => Unit = () => ()

  /** Runs the tiles and ends the job with what they make: the blend written as
    * its result, a pause recorded where it stopped, or the reason it could not
    * go on. The server is stopped and the tiles cleaned up either way.
    */
  def run(): Unit = {
    // Tiles sent to a session drift did not start for this job are run by a
    // server whose output is that session's: the job shows what the session
    // shows rather than nothing at all.
    session.foreach(live =>
      jobs.follows(
        job,
        () =>
          sessionManager.list
            .find(_.id == live.id)
            .map(current => (current.progress, current.activity))
            .getOrElse((Option.empty[SessionProgress], Option.empty[String]))
      )
    )
    try {
      jobs.showPicture(job.id, live)
      if (keepTiles) Files.createDirectories(jobs.files.tilesDirOf(job))
      jobs.startLog(
        job,
        launch,
        resumed = resumed,
        notes =
          // The largest, not the first: a shifted grid (27) cuts its end tiles
          // shorter than the ones in the middle.
          (notes :+ s"${tiles.size} tile(s) of at most " +
            s"${tiles.map(_.width).max}x${tiles.map(_.height).max}, " +
            s"overlapping by at least ${TiledJobs.TileOverlap} px" :+
            session.fold("tiles run on a server of the job's own")(live =>
              s"tiles run on the ready session ${live.id}"
            )) ++ Option
            .when(keepTiles)(s"tiles kept in ${jobs.files.tilesDirOf(job)}")
            .toList ++ Option
            .when(resumed)(
              s"resumed at tile ${jobs.files.tilesDone(job.id, tiles.size) + 1} of ${tiles.size}"
            )
            .toList
      )
      // What makes one tile from its index, the tile, the window the model is
      // handed and its pixels: the tile's output file written, or the reason
      // it was not. A session's server is watched by its session; a job
      // server's exit code is what a tile waiting on it names.
      val makeTile: Either[
        String,
        (Int, Tiling.Tile, Tiling.Tile, BufferedImage) => Either[String, Unit]
      ] =
        session
          .flatMap(_.port)
          .map(port => port -> (() => Option.empty[Int]))
          .toRight(())
          .left
          .flatMap(_ =>
            sessionManager
              .startJobServer(
                runConfigurationId,
                Some(launch.runtime.id),
                jobs.files.logFileOf(job),
                line => jobs.noteLine(job, line)
              )
              .map { server =>
                stopServer = server.stop
                server.port -> (() => server.exitCode)
              }
          )
          .flatMap((port, exitCode) =>
            NativeJobs
              .imageCapabilities(port)
              .map((defaults, features) => (port, exitCode, defaults, features))
          )
          .map { (port, exitCode, defaults, features) =>
            jobs.appendLog(
              job,
              "img_gen features: " +
                features.filter(_._2).keys.toList.sorted.mkString(", ")
            )
            (
                index: Int,
                tile: Tiling.Tile,
                window: Tiling.Tile,
                crop: BufferedImage
            ) => {
              if (keepTiles)
                ImageIO.write(crop, "png", tileFile(index, "input").toFile)
              request(
                defaults,
                features,
                TileWindow.TileInput(
                  tile,
                  window,
                  PostProcessImages.dataUrl(crop),
                  Option.when(window != tile)(
                    PostProcessImages.dataUrl(TileWindow.maskOf(tile, window))
                  )
                )
              )
                .flatMap { parameters =>
                  jobs.appendLog(
                    job,
                    s"${tileLine(index, tile)}: ${parameters.prompt}"
                  )
                  // The job's id as soon as the server has it: a cancel of
                  // this drift job cancels the tile it is waiting on.
                  try
                    NativeJobs.image(
                      port,
                      parameters,
                      exitCode,
                      nativeJobId => jobs.waitingFor(job, port, nativeJobId)
                    )
                  finally jobs.doneWaiting(job)
                }
                .flatMap { bytes =>
                  if (window == tile) {
                    Files.write(tileFile(index, "output"), bytes)
                    Right(())
                  } else
                    // The model painted the whole window; the tile is cut
                    // back out of it, and the rest was only there to be seen.
                    Option(ImageIO.read(java.io.ByteArrayInputStream(bytes)))
                      .toRight("the window returned cannot be decoded")
                      .flatMap { painted =>
                        if (
                          painted.getWidth != window.width ||
                          painted.getHeight != window.height
                        )
                          Left(
                            s"the window came back ${painted.getWidth}×${painted.getHeight}, not ${window.width}×${window.height}"
                          )
                        else {
                          if (keepTiles)
                            ImageIO.write(
                              painted,
                              "png",
                              tileFile(index, "window").toFile
                            )
                          ImageIO.write(
                            painted.getSubimage(
                              tile.x - window.x,
                              tile.y - window.y,
                              tile.width,
                              tile.height
                            ),
                            "png",
                            tileFile(index, "output").toFile
                          )
                          Right(())
                        }
                      }
                }
            }
          }
      makeTile
        .flatMap { make =>
          tiles.zipWithIndex
            .foldLeft[Either[String, Map[Tiling.Tile, Path]]](
              Right(Map.empty)
            ) { case (decoded, (tile, index)) =>
              decoded.flatMap { files =>
                val output = tileFile(index, "output")
                // Between two tiles there is nothing to kill, so a cancel —
                // and a pause — is only a flag: this is where the job reads
                // them.
                if (jobs.isCancelled(job)) Left("cancelled")
                else if (jobs.isPaused(job)) Left(TiledJobs.Paused)
                // A tile this job already has, from the run before the pause.
                else if (Files.isRegularFile(output)) {
                  picture.foreach { painted =>
                    val (left, top) = overlaps(tile)
                    TileBlending.paint(
                      painted,
                      tile,
                      ImageIO.read(output.toFile),
                      left,
                      top
                    )
                  }
                  jobs.update(job.id)(
                    _.copy(progress =
                      Some(PostProcessProgress(index + 1, tiles.size))
                    )
                  )
                  live.paint(tile, output)
                  Right(files + (tile -> output))
                } else {
                  val window = windowOf(tile)
                  val crop = picture.fold(
                    reference.getSubimage(
                      tile.x / scale,
                      tile.y / scale,
                      tile.width / scale,
                      tile.height / scale
                    )
                  )(painted =>
                    // A copy: the picture changes under a view once this
                    // tile is painted in, and the crop is what it is finished
                    // against.
                    PostProcessImages.copyOf(
                      painted.getSubimage(
                        window.x,
                        window.y,
                        window.width,
                        window.height
                      )
                    )
                  )
                  make(index, tile, window, prepareTile(crop))
                    .flatMap { _ =>
                      (PostProcessImages.imageSize(output) match {
                        case None => Some("no readable image was written")
                        case Some((width, height))
                            if width != tile.width || height != tile.height =>
                          Some(
                            s"the image is ${width}×$height, not ${tile.width}×${tile.height}"
                          )
                        case Some(_) =>
                          Option.when(PostProcessImages.isBlack(output))(
                            "the image is entirely black — the model's result was NaN"
                          )
                      }).toLeft(())
                    }
                    .flatMap { _ =>
                      finishing
                        .zip(picture)
                        .fold[Either[String, Unit]](Right(()))(
                          (finished, painted) =>
                            Option(ImageIO.read(output.toFile))
                              .toRight("the image cannot be decoded")
                              .map { returned =>
                                val done = finished(crop, returned)
                                done.note.foreach(note =>
                                  jobs.appendLog(
                                    job,
                                    s"${tileLine(index, tile)}: $note"
                                  )
                                )
                                if (keepTiles)
                                  done.kept.foreach((part, kept) =>
                                    ImageIO.write(
                                      kept,
                                      "png",
                                      tileFile(index, part).toFile
                                    )
                                  )
                                // The finished tile is what the job paints,
                                // so it is what the file holds: a resume
                                // repaints these, and the gallery shows them
                                // while the job runs.
                                ImageIO.write(
                                  done.image,
                                  "png",
                                  output.toFile
                                )
                                val (left, top) = overlaps(tile)
                                TileBlending
                                  .paint(painted, tile, done.image, left, top)
                              }
                        )
                    }
                    .left
                    .map(reason =>
                      s"tile ${index + 1} of ${tiles.size}: $reason"
                    )
                    .map { _ =>
                      doneThisRun += 1
                      val seconds =
                        (System.currentTimeMillis() - runStarted) / 1000.0 /
                          doneThisRun
                      jobs.update(job.id)(
                        _.copy(
                          progress =
                            Some(PostProcessProgress(index + 1, tiles.size)),
                          secondsPerTile = Some(seconds)
                        )
                      )
                      live.paint(tile, output)
                      files + (tile -> output)
                    }
                }
              }
            }
        }
        .flatMap { files =>
          val blended = picture.getOrElse(
            TileBlending.blend(
              rows,
              reference.getWidth * scale,
              reference.getHeight * scale,
              tile => ImageIO.read(files(tile).toFile)
            )
          )
          Either.cond(
            ImageIO.write(
              finish(blended.getSubimage(0, 0, target._1, target._2)),
              "png",
              outputFile.toFile
            ),
            (),
            "no PNG writer available"
          )
        } match {
        // Asked to pause: between two tiles, or with the tile in flight
        // dropped, which ends it with that tile's reason instead. A cancel
        // after the pause wins — it is the later word, and it keeps nothing.
        case Left(_) if jobs.isPaused(job) && !jobs.isCancelled(job) =>
          Files.deleteIfExists(outputFile)
          pictureKept = true
          live.storeAndClose(jobs.files.storePicture(job.id, _))(() =>
            jobs.hidePicture(job.id, live)
          )
          jobs.recordPaused(
            job,
            jobs.files.tilesDone(job.id, tiles.size),
            tiles.size
          )
        case Left(reason) =>
          Files.deleteIfExists(outputFile)
          jobs.fail(job, reason)
        case Right(()) =>
          jobs.forgetPaused(job.id)
          jobs.complete(job, src, outputFile, derivation)
      }
    } catch {
      case NonFatal(err) =>
        Files.deleteIfExists(outputFile)
        jobs.fail(job, s"the tiled job failed: ${err.getMessage}")
    } finally {
      jobs.unfollow(job)
      stopServer()
      if (!pictureKept) {
        live.close()
        jobs.hidePicture(job.id, live)
      }
      // A paused job keeps its tiles: they are what a resume carries on from
      // (`specs/40-pause-and-resume.md`). A cancel after the pause keeps none.
      if (!jobs.isPaused(job) || jobs.isCancelled(job))
        jobs.files.deletePicture(job.id)
      if (!keepTiles && (!jobs.isPaused(job) || jobs.isCancelled(job)))
        tiles.indices.foreach { index =>
          Files.deleteIfExists(tileFile(index, "input"))
          Files.deleteIfExists(tileFile(index, "output"))
          Files.deleteIfExists(tileFile(index, "window"))
        }
    }
  }
}
