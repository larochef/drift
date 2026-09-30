package drift.backend.images

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, IOException}
import java.lang.ProcessBuilder.Redirect
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** A video's still, for the places that show a video small (gallery cards,
  * version histories, the detail strip, project covers): only the players load
  * the video itself (bug 37). Read by the `ffmpeg` on the `PATH`, which the
  * drift runner needs anyway to write its videos.
  */
object VideoFrames {
  private val logger = Logger("drift.backend.images.VideoFrames")

  /** The longest `ffmpeg` gets to decode one frame. */
  private val TimeoutSeconds = 30L

  /** The first frame of `video`, or `None` when ffmpeg is missing or cannot
    * read it (logged).
    */
  def first(video: Path): Option[BufferedImage] =
    try {
      val process = new ProcessBuilder(
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
      ).redirectError(Redirect.DISCARD).start()
      process.getOutputStream.close()
      val bytes = process.getInputStream.readAllBytes()
      if (!process.waitFor(TimeoutSeconds, TimeUnit.SECONDS)) {
        process.destroyForcibly()
        logger.warn(s"ffmpeg took too long on the first frame of $video")
        None
      } else if (process.exitValue != 0 || bytes.isEmpty) {
        logger.warn(s"ffmpeg could not read the first frame of $video")
        None
      } else Option(ImageIO.read(ByteArrayInputStream(bytes)))
    } catch {
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
