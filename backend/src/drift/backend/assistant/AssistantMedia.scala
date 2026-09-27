package drift.backend.assistant

import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.*
import java.util.Base64
import java.util.concurrent.atomic.AtomicInteger
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** What an attachment becomes for the model: a data URL and whether it is an
  * image or a video, plus the mime type it carries.
  */
case class EncodedMedia(dataUrl: String, mimeType: String) {
  def isVideo: Boolean = mimeType.startsWith("video/")
}

/** The files behind chat attachments (`specs/21-assistant-page.md`): uploads
  * stored under `~/.cache/drift/assistant-uploads/` (re-uploadable, so under
  * the cache), generated outputs read from the outputs root, and both turned
  * into what llama-server accepts — images scaled to at most `maxSide` pixels
  * and re-encoded as JPEG to bound their cost in context, videos passed through
  * as they are.
  */
final class AssistantMedia(
    val uploadsRoot: Path,
    outputsRoot: Path,
    maxSide: Int = 1024
) {
  private val counter = AtomicInteger(0)

  def store(name: String, mimeType: String, bytes: Array[Byte]): Either[
    String,
    UploadedAttachment
  ] =
    if (bytes.isEmpty) Left("the upload is empty")
    else
      try {
        Files.createDirectories(uploadsRoot)
        val id = s"u${System.currentTimeMillis()}-${counter.incrementAndGet()}"
        val safeName = name.map(c =>
          if (c.isLetterOrDigit || c == '.' || c == '-' || c == '_') c else '_'
        )
        val file = uploadsRoot.resolve(s"$id-$safeName")
        Files.write(file, bytes)
        Right(
          UploadedAttachment(
            id = id,
            name = name,
            mimeType =
              if (mimeType.nonEmpty) mimeType else AssistantMedia.mimeFor(file),
            sizeBytes = bytes.length.toLong
          )
        )
      } catch {
        case NonFatal(err) =>
          Left(s"storing the upload failed: ${err.getMessage}")
      }

  /** The stored file for an upload id — ids are unique prefixes by construction
    * (`u<millis>-<n>-`).
    */
  def uploaded(id: String): Option[Path] =
    if (!Files.isDirectory(uploadsRoot) || id.isEmpty) None
    else {
      val stream = Files.list(uploadsRoot)
      try
        stream
          .iterator()
          .asScala
          .find(_.getFileName.toString.startsWith(s"$id-"))
      finally stream.close()
    }

  /** Resolves an attachment to a file, refusing anything outside the roots. */
  def resolve(attachment: ChatAttachment): Either[String, Path] =
    attachment match {
      case OutputAttachment(date, fileName) =>
        val file = outputsRoot.resolve(date).resolve(fileName).normalize()
        if (file.startsWith(outputsRoot) && Files.isRegularFile(file))
          Right(file)
        else Left(s"output $date/$fileName does not exist")
      case UploadAttachment(id) =>
        uploaded(id).toRight(s"upload '$id' does not exist")
    }

  /** The attachment as the model receives it. */
  def encode(attachment: ChatAttachment): Either[String, EncodedMedia] =
    resolve(attachment).flatMap { file =>
      val mimeType = AssistantMedia.mimeFor(file)
      try
        if (mimeType.startsWith("video/"))
          Right(
            EncodedMedia(dataUrl(mimeType, Files.readAllBytes(file)), mimeType)
          )
        else if (mimeType.startsWith("image/"))
          Option(ImageIO.read(file.toFile)) match {
            case Some(image) =>
              Right(
                EncodedMedia(
                  dataUrl("image/jpeg", scaledJpeg(image)),
                  "image/jpeg"
                )
              )
            case None =>
              // Undecodable here (WebP has no ImageIO reader): the model gets
              // the original bytes and decides for itself.
              Right(
                EncodedMedia(
                  dataUrl(mimeType, Files.readAllBytes(file)),
                  mimeType
                )
              )
          }
        else Left(s"${file.getFileName} is neither an image nor a video")
      catch {
        case NonFatal(err) =>
          Left(s"reading ${file.getFileName} failed: ${err.getMessage}")
      }
    }

  private def scaledJpeg(image: BufferedImage): Array[Byte] =
    drift.backend.images.Thumbnail.jpeg(image, maxSide)

  private def dataUrl(mimeType: String, bytes: Array[Byte]): String =
    s"data:$mimeType;base64,${Base64.getEncoder.encodeToString(bytes)}"
}

object AssistantMedia {
  def mimeFor(file: Path): String = {
    val name = file.getFileName.toString.toLowerCase
    val extension = name.lastIndexOf('.') match {
      case -1 => ""
      case i  => name.substring(i + 1)
    }
    extension match {
      case "png"          => "image/png"
      case "jpg" | "jpeg" => "image/jpeg"
      case "webp"         => "image/webp"
      case "gif"          => "image/gif"
      case "bmp"          => "image/bmp"
      case "mp4"          => "video/mp4"
      case "webm"         => "video/webm"
      case "mkv"          => "video/x-matroska"
      case "mov"          => "video/quicktime"
      case _              => "application/octet-stream"
    }
  }
}
