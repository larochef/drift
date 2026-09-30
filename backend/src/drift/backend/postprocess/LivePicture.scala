package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.*
import javax.imageio.ImageIO
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger
import ox.*
import ox.channels.{Channel, ChannelClosed}

/** The picture a tiled job is making, as it stands after the tiles it has
  * finished (`specs/15-post-hoc-resize.md`): what the gallery shows over the
  * source while the job runs, scaled for the screen or at full size.
  *
  * Painted on a fork of its own in the job's scope, never on the job's thread:
  * a finished tile is posted to its mailbox by the file it was written to, and
  * read, blended and encoded there, one piece of work after another, so a slow
  * encode of a 16384² picture costs the next tile nothing. The tiles are
  * painted in the order they are handed over — the job's own — with the same
  * ramps as the job's blend, so the picture is what the result would be if the
  * job ended now: the source under the tiles still to come, and the job's
  * `finish` (a partial redraw's paste back into the source) applied on the way
  * out.
  *
  * Two copies are kept. The full-size one is only ever read on the painter, and
  * written to `fullSizeFile` when asked for, once per tile count. The screen's
  * one (`PostProcessPicture.ScreenSide`) is brought up to date around each tile
  * as it is painted in, so what the gallery polls costs a small encode and
  * never a pass over the full picture — at 16384² that pass took longer than a
  * tile.
  */
final private[postprocess] class LivePicture(
    jobId: String,
    /** What the tiles are cut from: the source, padded; ×`scale` it is what
      * lies under the tiles not painted yet.
      */
    reference: BufferedImage,
    scale: Int,
    /** How far each tile overlaps the ones before it, as the job ramps them. */
    overlaps: Map[Tiling.Tile, (Int, Int)],
    target: (Int, Int),
    finish: PictureFinish,
    /** Where the picture is written at full size when it is asked for. */
    fullSizeFile: Path,
    /** Told how many tiles the picture holds after each one is painted in. */
    onPainted: Int => Unit
)(using Ox) {
  import LivePicture.Work

  private val logger = Logger[LivePicture]

  private val (resultWidth, resultHeight) = finish.size(target)
  private val screenScale =
    (PostProcessPicture.ScreenSide.toDouble / resultWidth.max(resultHeight))
      .min(1.0)
  private val screenWidth = math.round(resultWidth * screenScale).toInt.max(1)
  private val screenHeight =
    math.round(resultHeight * screenScale).toInt.max(1)

  // Unbounded: the job thread posting a tile never waits on the painter.
  private val mailbox = Channel.unlimited[Work]

  // The screen's copy is read by the requests polling it while the painter
  // draws into it; `lock` guards it and what goes with it. The full-size
  // picture's pixels change on the painter alone, which reads them outside
  // the lock. Both are built on the painter: a PiD's base is the reference ×4,
  // which the job has no reason to wait for.
  private val lock = Object()
  private var canvas = Option.empty[BufferedImage]
  private var screen = Option.empty[BufferedImage]
  private var painted = 0
  // A base still being built when the job ends must not be put back once the
  // picture is let go.
  private var closed = false

  /** The last encoding of the screen's copy asked for at each size, with the
    * number of tiles it holds — a screen polling the same version is answered
    * without encoding again.
    */
  private var encoded = Map.empty[Int, (Int, Array[Byte])]

  // On the painter alone: the tile count `fullSizeFile` holds.
  private var written = Option.empty[Int]

  // The painter: ends once the mailbox is done, or with the job's scope.
  forkDiscard {
    try {
      attempt(buildBase())
      repeatWhile {
        mailbox.receiveOrClosed() match {
          case Work.Paint(tile, file) =>
            attempt(paintIn(tile, file))
            true
          case Work.FullSize(reply) =>
            reply.send(
              try writeFullSize()
              catch {
                case NonFatal(err) =>
                  logger.warn(
                    s"The picture of job $jobId could not be written at full size",
                    err
                  )
                  None
              }
            )
            true
          case Work.Store(store, reply) =>
            attempt(
              // The screen's copy is drawn into on the painter alone, which
              // this is.
              lock.synchronized(canvas.zip(screen)).foreach {
                (picture, shown) =>
                  store(current(picture), shown)
              }
            )
            close()
            reply.send(())
            true
          case _: ChannelClosed => false
        }
      }
    } finally {
      lock.synchronized(release())
      mailbox.doneOrClosed().discard
      // Whoever still waits is answered: nothing more is coming.
      repeatWhile {
        mailbox.tryReceiveOrClosed() match {
          case Some(Work.FullSize(reply)) => reply.send(None); true
          case Some(Work.Store(_, reply)) => reply.send(()); true
          case Some(Work.Paint(_, _))     => true
          case None | (_: ChannelClosed)  => false
        }
      }
    }
  }

  /** Paints the tile written to `file` in, after every tile handed over before
    * it. Returns at once.
    */
  def paint(tile: Tiling.Tile, file: Path): Unit =
    mailbox.sendOrClosed(Work.Paint(tile, file)).discard

  /** The picture as it stands, scaled for the screen, as PNG bytes: at most
    * `side` px on its longest edge, and never more than
    * `PostProcessPicture.ScreenSide`. None before the base is ready, or once
    * closed.
    */
  def forScreen(side: Int): Option[Array[Byte]] = {
    val shown = lock.synchronized {
      screen.map(picture =>
        encoded.get(side) match {
          case Some((version, bytes)) if version == painted => Left(bytes)
          case _ => Right((painted, PostProcessImages.copyOf(picture)))
        }
      )
    }
    shown.map {
      case Left(bytes)            => bytes
      case Right((version, copy)) =>
        val bytes =
          LivePicture.png(PostProcessImages.fitWithin(copy, side))
        lock.synchronized {
          if (!closed) encoded = encoded.updated(side, (version, bytes))
        }
        bytes
    }
  }

  /** The file holding the picture as it stands at full size, written on the
    * painter — after the tiles already handed over, and before the next — or
    * found there already when no tile has landed since. Blocks until then. None
    * before the base is ready, or once closed.
    */
  def fullSize(): Option[Path] = {
    val reply = Channel.buffered[Option[Path]](1)
    mailbox.sendOrClosed(Work.FullSize(reply)) match {
      case _: ChannelClosed => None
      case _                => reply.receive()
    }
  }

  /** Once every tile handed over is painted in, gives the picture — at full
    * size and scaled for the screen — to `store` and lets it go: a paused job
    * keeps what it did on disk rather than in memory
    * (`specs/40-pause-and-resume.md`). Blocks until it is stored.
    */
  def storeAndClose(store: (BufferedImage, BufferedImage) => Unit): Unit = {
    val reply = Channel.buffered[Unit](1)
    mailbox.sendOrClosed(Work.Store(store, reply)) match {
      case _: ChannelClosed => ()
      case _                => reply.receive()
    }
  }

  /** Drops the picture: what was still to be painted into it is skipped, and
    * whoever waits for it at full size gets nothing.
    */
  def close(): Unit = {
    lock.synchronized(release())
    mailbox.doneOrClosed().discard
  }

  private def release(): Unit = {
    closed = true
    canvas = None
    screen = None
    encoded = Map.empty
  }

  private def buildBase(): Unit = {
    val base =
      if (scale == 1) PostProcessImages.copyOf(reference)
      else
        PostProcessImages.scaledCopy(
          reference,
          reference.getWidth * scale,
          reference.getHeight * scale
        )
    val whole = current(base)
    // A copy even when no smaller: the screen's picture is drawn into.
    val shown =
      if (screenScale < 1)
        PostProcessImages.scaledCopy(whole, screenWidth, screenHeight)
      else PostProcessImages.copyOf(whole)
    lock.synchronized {
      if (!closed) {
        canvas = Some(base)
        screen = Some(shown)
      }
    }
  }

  private def paintIn(tile: Tiling.Tile, file: Path): Unit =
    lock.synchronized(canvas).foreach { picture =>
      Option(ImageIO.read(file.toFile)).foreach { image =>
        val (left, top) = overlaps.getOrElse(tile, (0, 0))
        TileBlending.paint(picture, tile, image, left, top)
        refreshScreen(picture, tile)
        val count = lock.synchronized {
          painted += 1
          painted
        }
        onPainted(count)
      }
    }

  /** Scales the part of the result `tile` changed into the screen's copy: the
    * screen pixels it touches, from the result's pixels under them.
    */
  private def refreshScreen(
      picture: BufferedImage,
      tile: Tiling.Tile
  ): Unit = {
    val width = (tile.x + tile.width).min(target._1) - tile.x
    val height = (tile.y + tile.height).min(target._2) - tile.y
    if (width > 0 && height > 0) {
      val landed = finish.placed(ImageRegion(tile.x, tile.y, width, height))
      val left = math.floor(landed.x * screenScale).toInt
      val top = math.floor(landed.y * screenScale).toInt
      val right = math
        .ceil((landed.x + landed.width) * screenScale)
        .toInt
        .min(screenWidth)
      val bottom = math
        .ceil((landed.y + landed.height) * screenScale)
        .toInt
        .min(screenHeight)
      val partLeft = math.floor(left / screenScale).toInt
      val partTop = math.floor(top / screenScale).toInt
      val part = ImageRegion(
        partLeft,
        partTop,
        math.ceil(right / screenScale).toInt.min(resultWidth) - partLeft,
        math.ceil(bottom / screenScale).toInt.min(resultHeight) - partTop
      )
      val result =
        BufferedImage(part.width, part.height, BufferedImage.TYPE_INT_RGB)
      result.setRGB(
        0,
        0,
        part.width,
        part.height,
        finish.pixels(picture.getSubimage(0, 0, target._1, target._2), part),
        0,
        part.width
      )
      val scaled =
        PostProcessImages.scaledCopy(result, right - left, bottom - top)
      lock.synchronized {
        screen.foreach { shown =>
          val graphics = shown.createGraphics()
          try graphics.drawImage(scaled, left, top, null)
          finally graphics.dispose()
        }
      }
    }
  }

  /** On the painter: `fullSizeFile`, written unless it already holds the tiles
    * painted so far.
    */
  private def writeFullSize(): Option[Path] = {
    val (picture, version) = lock.synchronized((canvas, painted))
    picture.map { painting =>
      if (!written.contains(version) || !Files.isRegularFile(fullSizeFile)) {
        val partial = fullSizeFile.resolveSibling(
          fullSizeFile.getFileName.toString + ".part"
        )
        ImageIO.write(current(painting), "png", partial.toFile)
        Files.move(partial, fullSizeFile, StandardCopyOption.REPLACE_EXISTING)
        written = Some(version)
      }
      fullSizeFile
    }
  }

  /** What the job would write if it ended now. */
  private def current(picture: BufferedImage): BufferedImage =
    finish(picture.getSubimage(0, 0, target._1, target._2))

  /** A piece of the painter's work that failed costs the picture that update,
    * not the painter.
    */
  private def attempt(work: => Unit): Unit =
    try work
    catch {
      case NonFatal(err) =>
        logger.warn(s"The picture of job $jobId missed an update", err)
    }
}

private[postprocess] object LivePicture {

  /** What the painter is asked to do, in the order it is asked. */
  private enum Work {
    case Paint(tile: Tiling.Tile, file: Path)

    /** Answered with the full-size file, or None once there is no picture. */
    case FullSize(reply: Channel[Option[Path]])

    /** Answered once stored, or once there is nothing left to store. */
    case Store(
        store: (BufferedImage, BufferedImage) => Unit,
        reply: Channel[Unit]
    )
  }

  def png(image: BufferedImage): Array[Byte] = {
    val bytes = java.io.ByteArrayOutputStream()
    ImageIO.write(image, "png", bytes)
    bytes.toByteArray
  }
}
