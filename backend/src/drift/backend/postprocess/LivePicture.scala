package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.concurrent.{ExecutorService, Executors}
import javax.imageio.ImageIO
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** The picture a tiled job is making, as it stands after the tiles it has
  * finished (`specs/15-post-hoc-resize.md`): what the gallery shows over the
  * source while the job runs, scaled for the screen or at full size.
  *
  * Painted on a thread of its own, never the job's: a finished tile is handed
  * over by the file it was written to, and read, blended and encoded here, so a
  * slow encode of an 8192² picture costs the next tile nothing. The tiles are
  * painted in the order they are handed over — the job's own — with the same
  * ramps as the job's blend, so the picture is what the result would be if the
  * job ended now: the source under the tiles still to come, and the job's
  * `finish` (a partial redraw's paste back into the source) applied on the way
  * out.
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
    finish: BufferedImage => BufferedImage,
    /** Told how many tiles the picture holds after each one is painted in. */
    onPainted: Int => Unit
) {
  private val logger = Logger[LivePicture]

  private val painter: ExecutorService = Executors.newSingleThreadExecutor {
    runnable =>
      val thread = Thread(runnable, s"drift-picture-$jobId")
      thread.setDaemon(true)
      thread
  }

  private val lock = Object()
  // Guarded by `lock`. The base is built on the painter too: a PiD's is the
  // reference ×4, which the job has no reason to wait for.
  private var canvas = Option.empty[BufferedImage]
  private var painted = 0
  // Guarded by `lock`: a base still being built when the job ends must not be
  // put back once the picture is let go.
  private var closed = false

  /** The last encoding asked for at each size (None: full size), with the
    * number of tiles it holds — a screen polling the same version is answered
    * without encoding again.
    */
  private var encoded = Map.empty[Option[Int], (Int, Array[Byte])]

  submit {
    val base =
      if (scale == 1) PostProcessImages.copyOf(reference)
      else
        PostProcessImages.scaledCopy(
          reference,
          reference.getWidth * scale,
          reference.getHeight * scale
        )
    lock.synchronized { if (!closed) canvas = Some(base) }
  }

  /** Paints the tile written to `file` in, after every tile handed over before
    * it. Returns at once.
    */
  def paint(tile: Tiling.Tile, file: Path): Unit =
    submit {
      Option(ImageIO.read(file.toFile)).foreach { image =>
        val (left, top) = overlaps.getOrElse(tile, (0, 0))
        val count = lock.synchronized {
          canvas.foreach(TileBlending.paint(_, tile, image, left, top))
          painted += 1
          painted
        }
        onPainted(count)
      }
    }

  /** The picture as it stands, as PNG bytes: at most `side` px on its longest
    * edge, or at full size. None before the base is ready, or once closed.
    */
  def snapshot(side: Option[Int]): Option[Array[Byte]] =
    lock.synchronized {
      canvas.map { picture =>
        encoded.get(side) match {
          case Some((version, bytes)) if version == painted => bytes
          case _                                            =>
            val bytes = LivePicture.png(
              side.fold(current(picture))(longest =>
                PostProcessImages.fitWithin(current(picture), longest)
              )
            )
            encoded = encoded.updated(side, (painted, bytes))
            bytes
        }
      }
    }

  /** Once every tile handed over is painted in, gives the picture to `store`
    * and lets it go — a paused job keeps what it did on disk rather than in
    * memory (`specs/40-pause-and-resume.md`) — then calls `stored`.
    */
  def storeAndClose(store: BufferedImage => Unit)(stored: () => Unit): Unit = {
    submit {
      lock.synchronized {
        canvas.foreach(picture => store(current(picture)))
        release()
      }
      stored()
    }
    painter.shutdown()
  }

  /** Drops the picture and whatever was still to be painted into it. */
  def close(): Unit = {
    painter.shutdownNow()
    lock.synchronized(release())
  }

  private def release(): Unit = {
    closed = true
    canvas = None
    encoded = Map.empty
  }

  /** What the job would write if it ended now. */
  private def current(picture: BufferedImage): BufferedImage =
    finish(picture.getSubimage(0, 0, target._1, target._2))

  private def submit(work: => Unit): Unit =
    try
      painter.execute { () =>
        try work
        catch {
          case NonFatal(err) =>
            logger.warn(s"The picture of job $jobId missed an update", err)
          case _: InterruptedException => ()
        }
      }
    catch {
      // Closed: the job has ended, and its picture with it.
      case _: java.util.concurrent.RejectedExecutionException => ()
    }
}

private[postprocess] object LivePicture {

  def png(image: BufferedImage): Array[Byte] = {
    val bytes = java.io.ByteArrayOutputStream()
    ImageIO.write(image, "png", bytes)
    bytes.toByteArray
  }
}
