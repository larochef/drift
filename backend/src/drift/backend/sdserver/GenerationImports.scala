package drift.backend.sdserver

import drift.shared.*

import java.io.ByteArrayInputStream
import java.nio.file.{Files, Path}
import java.util.concurrent.atomic.AtomicLong
import javax.imageio.ImageIO
import scala.util.Using
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Images from outside drift, brought into the gallery
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
          val now = System.currentTimeMillis()
          // The id scheme is the generations' own — the history's lookups and
          // deletes match on it — offset so that an import cannot take the
          // id of a generation or post-processing job of the same instant.
          val id = s"g$now-${2000 + counter.incrementAndGet()}"
          val date = files.dateOf(now)
          val fileName = s"$id-0.$format"
          val directory = outputsRoot.resolve(date)
          Files.createDirectories(directory)
          Files.write(directory.resolve(fileName), bytes)
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
            importedFileName = Some(request.fileName),
            inputSources = List.empty
          )
          files.writeSidecar(generation)
          logger.info(s"Imported ${request.fileName} as $id ($date)")
          Right(generation)
      }
    } catch {
      case NonFatal(err) =>
        Left(s"Importing '${request.fileName}' failed: ${err.getMessage}")
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
}
