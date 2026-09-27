package drift.backend.sdserver

import drift.backend.postprocess.PostProcessImages

import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Screen-sized copies of persisted images (`specs/12-gallery.md`).
  *
  * drift makes big files on purpose — a PiD upscale of a 2048 image is 8192²,
  * 150 MB on disk and about 256 MB once decoded — and the pages that show them
  * give them a 700 px box. Handing the browser the original meant decoding a
  * quarter of a gigabyte to paint a thumbnail: measured at a 523 ms frozen
  * frame on an idle machine, and seconds of it on a busy one (François,
  * 2026-09-19).
  *
  * So a scaled copy is made once, kept in the cache beside the logs, and served
  * instead. An image already small enough is served as it is; the original is
  * always one click away, which is what the preview is not for.
  */
final class OutputPreviews(outputsRoot: Path, cacheRoot: Path) {
  private val logger = Logger[OutputPreviews]

  /** The longest side a preview is allowed, when the caller does not say. */
  val defaultSide: Int = 2048

  private def previewsRoot: Path = cacheRoot.resolve("previews")

  /** The file for `date`/`fileName`, provided it really is under the outputs
    * root — the path comes from a URL.
    */
  private def sourceOf(date: String, fileName: String): Option[Path] = {
    val file = outputsRoot.resolve(date).resolve(fileName).normalize()
    Option.when(
      file.startsWith(outputsRoot) && Files.isRegularFile(file)
    )(file)
  }

  /** What the image measures, from its header: no decode, and no preview
    * written.
    */
  def sizeOf(date: String, fileName: String): Option[(Int, Int)] =
    sourceOf(date, fileName).flatMap(PostProcessImages.imageSize)

  /** A copy of the image no larger than `side` on its longest edge, as PNG
    * bytes. The original is answered when it is already that small, or when it
    * cannot be read as an image — a video, say, which has no preview and must
    * be asked for by its own URL.
    */
  def preview(
      date: String,
      fileName: String,
      side: Option[Int]
  ): Option[(Array[Byte], String)] =
    sourceOf(date, fileName).map { file =>
      val longest = side.filter(_ > 0).getOrElse(defaultSide).min(8192)
      val original =
        (
          Files.readAllBytes(file),
          GenerationManager.mimeTypeFor(extensionOf(fileName))
        )
      PostProcessImages.imageSize(file) match {
        case Some((width, height)) if math.max(width, height) > longest =>
          scaled(file, date, fileName, longest).getOrElse(original)
        case _ => original
      }
    }

  private def extensionOf(fileName: String): String =
    fileName.lastIndexOf('.') match {
      case -1 => ""
      case i  => fileName.substring(i + 1).toLowerCase
    }

  /** The cached copy, made if this is the first time it is asked for. */
  private def scaled(
      file: Path,
      date: String,
      fileName: String,
      side: Int
  ): Option[(Array[Byte], String)] =
    try {
      val target = previewsRoot
        .resolve(date)
        .resolve(
          s"${fileName.stripSuffix("." + extensionOf(fileName))}-$side.png"
        )
      if (!Files.isRegularFile(target)) {
        Files.createDirectories(target.getParent)
        val image = ImageIO.read(file.toFile)
        if (image == null) return None
        val scale = side.toDouble / math.max(image.getWidth, image.getHeight)
        val preview = PostProcessImages.scaledCopy(
          image,
          math.round(image.getWidth * scale).toInt.max(1),
          math.round(image.getHeight * scale).toInt.max(1)
        )
        // Written to a temporary name first: a half-written preview served to
        // the next request would look like a corrupt image.
        val partial =
          target.resolveSibling(target.getFileName.toString + ".part")
        ImageIO.write(preview, "png", partial.toFile)
        Files.move(partial, target)
        logger.info(
          s"Preview of $date/$fileName at ${preview.getWidth}x${preview.getHeight}"
        )
      }
      Some((Files.readAllBytes(target), "image/png"))
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Cannot make a preview of $date/$fileName", err)
        None
    }
}
