package drift.backend.images

import drift.backend.process.ChildProcesses

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, IOException}
import java.nio.file.Path
import javax.imageio.ImageIO
import scala.concurrent.duration.DurationInt
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** A video's still, for the places that show a video small (gallery cards,
  * version histories, the detail strip, project covers): only the players load
  * the video itself. Read by the `ffmpeg` on the `PATH`, which the drift runner
  * needs anyway to write its videos.
  */
object VideoFrames {
  private val logger = Logger("drift.backend.images.VideoFrames")

  /** The longest `ffmpeg` gets to decode one frame. */
  private val Timeout = 30.seconds

  /** More than a frame of any video drift shows weighs as a PNG. */
  private val MostBytes = 256 * 1024 * 1024

  /** The first frame of `video`, or `None` when ffmpeg is missing or cannot
    * read it (logged).
    */
  def first(video: Path): Option[BufferedImage] =
    try
      ChildProcesses.answer(
        List(
          "ffmpeg",
          "-v",
          "error",
          "-i",
          video.toString,
          "-frames:v",
          "1",
          "-f",
          "image2pipe",
          "-c:v",
          "png",
          "-"
        ),
        within = Timeout,
        most = MostBytes,
        discardErrors = true
      ) match {
        case Left(_) =>
          logger.warn(s"ffmpeg took too long on the first frame of $video")
          None
        case Right(answer) if answer.exit != 0 || answer.bytes.isEmpty =>
          logger.warn(s"ffmpeg could not read the first frame of $video")
          None
        case Right(answer) =>
          Option(ImageIO.read(ByteArrayInputStream(answer.bytes)))
      }
    catch {
      case error: IOException =>
        logger.warn(
          s"No still of $video: ffmpeg is not installed (${error.getMessage})"
        )
        None
      case NonFatal(error) =>
        logger.warn(s"No still of $video", error)
        None
    }
}
