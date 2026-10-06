package drift.backend.sdserver

import drift.shared.*

import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.concurrent.atomic.AtomicLong
import javax.imageio.ImageIO
import scala.util.Using
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Images and videos from outside drift, brought into the gallery
  * (`specs/30-gallery-ergonomics-and-image-import.md`). Each becomes an entry
  * of its own — kind "import", completed, one output, no session, no
  * configuration, no request, no project — laid out like a generation:
  * `<id>-0.<format>` and the `<id>.json` sidecar under today. Post-processing
  * then takes it like any recorded output, and what it makes chains from it.
  */
final class GenerationImports(outputsRoot: Path) {
  private val logger = Logger[GenerationImports]
  private val files = GenerationFiles(outputsRoot)
  private val counter = AtomicLong(0)

  def importImage(request: ImageImport): Either[String, Generation] =
    try {
      val (bytes, _) = GenerationFiles.decodeMediaData(request.data)
      GenerationImports.formatOf(bytes) match {
        case None =>
          Left(
            s"'${request.fileName}' is not a PNG or JPEG image — the only " +
              "formats every post-processing tool reads"
          )
        case Some(format) =>
          Right(entry(request.fileName, format)(Files.write(_, bytes)))
      }
    } catch {
      case NonFatal(err) =>
        Left(s"Importing '${request.fileName}' failed: ${err.getMessage}")
    }

  /** A video, already on disk where the server received it: `upload` is moved
    * into the gallery, or deleted when it is refused.
    */
  def importVideo(fileName: String, upload: Path): Either[String, Generation] =
    try
      GenerationImports.videoFormatOf(upload) match {
        case None =>
          Left(
            s"'$fileName' is not a WebM, MP4, MOV or Matroska video — the " +
              "containers the gallery plays"
          )
        case Some(format) =>
          Right(
            entry(fileName, format)(
              Files.move(upload, _, StandardCopyOption.REPLACE_EXISTING)
            )
          )
      }
    catch {
      case NonFatal(err) =>
        Left(s"Importing '$fileName' failed: ${err.getMessage}")
    } finally Files.deleteIfExists(upload)

  /** The gallery entry of one imported file, which `write` puts in place. */
  private def entry(importedFileName: String, format: String)(
      write: Path => Unit
  ): Generation = {
    val now = System.currentTimeMillis()
    // The id scheme is the generations' own — the history's lookups and
    // deletes match on it — offset so that an import cannot take the
    // id of a generation or post-processing job of the same instant.
    val id = s"g$now-${2000 + counter.incrementAndGet()}"
    val date = files.dateOf(now)
    val fileName = s"$id-0.$format"
    val directory = outputsRoot.resolve(date)
    Files.createDirectories(directory)
    write(directory.resolve(fileName))
    val generation = Generation(
      id = id,
      sessionId = "",
      runConfigurationId = "",
      kind = "import",
      status = GenerationStatus.Completed,
      submittedAt = now,
      completedAt = Some(now),
      outputs = List(
        GenerationOutput(
          date = date,
          fileName = fileName,
          url = s"/api/outputs/$date/$fileName",
          mimeType = GenerationManager.mimeTypeFor(format),
          format = format
        )
      ),
      importedFileName = Some(importedFileName),
      inputSources = List.empty
    )
    files.writeSidecar(generation)
    logger.info(s"Imported $importedFileName as $id ($date)")
    generation
  }
}

object GenerationImports {

  /** "png" or "jpeg" from the bytes themselves, whatever the file was called;
    * none for anything else. WebP and the rest are refused rather than
    * converted: sd-cli reads PNG and JPEG, and so does every JDK decoder the
    * tiled jobs use.
    */
  def formatOf(bytes: Array[Byte]): Option[String] =
    Using.resource(
      ImageIO.createImageInputStream(ByteArrayInputStream(bytes))
    ) { stream =>
      val readers = ImageIO.getImageReaders(stream)
      Option
        .when(readers.hasNext)(readers.next())
        .map { reader =>
          try reader.getFormatName.toLowerCase
          finally reader.dispose()
        }
        .collect {
          case "png"          => "png"
          case "jpeg" | "jpg" => "jpeg"
        }
    }

  /** "webm", "mkv", "mp4" or "mov" from the file's first bytes, whatever it is
    * called; none for anything else. Matroska and WebM share the EBML header
    * and differ by the document type it names; MP4 and QuickTime share the
    * `ftyp` box and differ by its brand.
    */
  def videoFormatOf(file: Path): Option[String] = {
    val head = Using.resource(Files.newInputStream(file))(_.readNBytes(64))
    val text = String(head, StandardCharsets.ISO_8859_1)
    if (text.startsWith("\u001a\u0045\u00df\u00a3"))
      if (text.contains("webm")) Some("webm")
      else Option.when(text.contains("matroska"))("mkv")
    else if (text.slice(4, 8) == "ftyp")
      Some(if (text.slice(8, 12) == "qt  ") "mov" else "mp4")
    else None
  }
}
