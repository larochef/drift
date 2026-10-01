package drift.backend.sdserver

import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.time.Instant
import java.util.Base64
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger

/** Where a generation's files live under the outputs root — its day once kept,
  * `scratch/` while it is free play — and how they get there: the inputs
  * written beside the outputs, the outputs decoded from the finished job, the
  * parameter sidecar.
  */
final private[sdserver] class GenerationFiles(outputsRoot: Path) {
  private val logger = Logger[GenerationFiles]

  def dateOf(timestampMillis: Long): String =
    GenerationManager.DateFormat.format(Instant.ofEpochMilli(timestampMillis))

  /** The directory under the outputs root a generation's files live in: its day
    * once kept, `scratch/` while it is free play.
    */
  def directoryOf(generation: Generation): String =
    if (generation.scratch) GenerationManager.ScratchDirectory
    else dateOf(generation.submittedAt)

  /** Writes one input (raw base64 or data URL) beside the outputs and answers
    * with the URL it will be served from. A write failure downgrades to a
    * placeholder rather than failing the generation — the input is convenience
    * history, the generation is the point.
    */
  def externalizeInput(
      generationId: String,
      submittedAt: Long,
      tag: String,
      data: String,
      scratch: Boolean
  ): String =
    try {
      val (bytes, extension) = GenerationFiles.decodeMediaData(data)
      val date =
        if (scratch) GenerationManager.ScratchDirectory else dateOf(submittedAt)
      val directory = outputsRoot.resolve(date)
      Files.createDirectories(directory)
      val fileName = s"$generationId-$tag.$extension"
      Files.write(directory.resolve(fileName), bytes)
      s"/api/outputs/$date/$fileName"
    } catch {
      case NonFatal(err) =>
        logger.warn(
          s"Generation $generationId: persisting input '$tag' failed: ${err.getMessage}"
        )
        "<input not saved>"
    }

  /** Decodes the base64 payload(s) into files under the generation's directory.
    * Images come one per batch index; a video is a single encoded container.
    */
  def persist(
      generation: Generation,
      result: NativeJobResult
  ): List[GenerationOutput] = {
    val date = directoryOf(generation)
    val directory = outputsRoot.resolve(date)
    Files.createDirectories(directory)

    def write(fileName: String, base64: String): Unit =
      Files.write(
        directory.resolve(fileName),
        Base64.getMimeDecoder.decode(base64)
      )

    if (generation.kind == "vid_gen") {
      val format =
        Option(result.outputFormat).filter(_.nonEmpty).getOrElse("webm")
      result.b64Json.toList.map { payload =>
        val fileName = s"${generation.id}.$format"
        write(fileName, payload)
        GenerationOutput(
          date = date,
          fileName = fileName,
          url = s"/api/outputs/$date/$fileName",
          mimeType =
            result.mimeType.getOrElse(GenerationManager.mimeTypeFor(format)),
          format = format,
          fps = result.fps,
          frameCount = result.frameCount
        )
      }
    } else {
      val format =
        Option(result.outputFormat).filter(_.nonEmpty).getOrElse("png")
      result.images.map { image =>
        val fileName = s"${generation.id}-${image.index}.$format"
        write(fileName, image.b64Json)
        GenerationOutput(
          date = date,
          fileName = fileName,
          url = s"/api/outputs/$date/$fileName",
          mimeType = GenerationManager.mimeTypeFor(format),
          format = format,
          index = image.index
        )
      }
    }
  }

  /** The parameter sidecar, `<id>.json` beside the outputs — what "reuse these
    * parameters" reloads after a restart. Free play writes none.
    */
  def writeSidecar(generation: Generation): Unit =
    if (generation.scratch) ()
    else
      try {
        val date = dateOf(generation.submittedAt)
        Files.write(
          outputsRoot.resolve(date).resolve(s"${generation.id}.json"),
          writeToString(generation, WriterConfig.withIndentionStep(2))
            .getBytes(StandardCharsets.UTF_8)
        )
      } catch {
        case NonFatal(err) =>
          logger.warn(
            s"Generation ${generation.id}: writing the sidecar failed: ${err.getMessage}"
          )
      }

  /** The files one generation owns in a directory. The naming scheme is this
    * class's own and an id is never a prefix of another id's files, the rule
    * `GenerationHistory.delete` relies on too.
    */
  def filesOwnedBy(directory: Path, generationId: String): List[Path] =
    if (!Files.isDirectory(directory)) List.empty
    else
      Using.resource(Files.list(directory)) { stream =>
        stream.iterator.asScala.filter { file =>
          val name = file.getFileName.toString
          name.startsWith(s"$generationId-") ||
          name.startsWith(s"$generationId.")
        }.toList
      }

  /** A recorded input URL, moved from the scratch directory into a day. */
  def rehome(url: String, date: String): String =
    url.replace(
      s"/api/outputs/${GenerationManager.ScratchDirectory}/",
      s"/api/outputs/$date/"
    )
}

private[sdserver] object GenerationFiles {

  /** Base64 payload plus file extension, from a raw base64 string or a data URL
    * of any medium drift serves (`GenerationManager.extensionFor`): an image,
    * or a video model's clip or sound. A bare payload or an unknown type is
    * taken for a PNG, what the image fields have always carried.
    */
  def decodeMediaData(data: String): (Array[Byte], String) =
    if (data.startsWith("data:")) {
      val comma = data.indexOf(',')
      val mime = data.substring(5, comma).takeWhile(_ != ';')
      (
        Base64.getMimeDecoder.decode(data.substring(comma + 1)),
        GenerationManager.extensionFor(mime).getOrElse("png")
      )
    } else (Base64.getMimeDecoder.decode(data), "png")
}
