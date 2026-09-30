package drift.backend.routes

import drift.backend.sdserver.{GenerationManager, OutputPreviews}
import drift.shared.*

import java.nio.file.Files

import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.{FileRange, RangeValue}
import sttp.tapir.server.ServerEndpoint

def generationEndpoints(
    manager: GenerationManager,
    previews: OutputPreviews
): List[ServerEndpoint[Any, Identity]] = List(
  getSessionCapabilities.serverLogicSuccess[Identity](manager.capabilities),
  submitImageGeneration.serverLogicSuccess[Identity] {
    (sessionId, context, scratch, parameters) =>
      manager.submitImage(
        sessionId,
        parameters,
        context,
        scratch.getOrElse(false)
      )
  },
  submitVideoGeneration.serverLogicSuccess[Identity] {
    (sessionId, context, scratch, parameters) =>
      manager.submitVideo(
        sessionId,
        parameters,
        context,
        scratch.getOrElse(false)
      )
  },
  listGenerations.serverLogicSuccess[Identity](manager.list),
  cancelGeneration.serverLogicSuccess[Identity]((id, force) =>
    manager.cancel(id, force)
  ),
  // The date and file name are single path segments by construction, but
  // normalize-and-contain anyway: this endpoint must never serve a byte from
  // outside the outputs root.
  // What the pages show: the original scaled to the screen, made once and
  // cached (`specs/12-gallery.md`). The full file stays one click away.
  getOutputPreview.serverLogic[Identity] { (date, fileName, side) =>
    previews.preview(date, fileName, side).toRight(())
  },
  getOutputSize.serverLogic[Identity] { (date, fileName) =>
    previews
      .sizeOf(date, fileName)
      .map((width, height) => OutputSize(width, height))
      .toRight(())
  },
  getOutputFile.serverLogic[Identity] { (date, fileName, range) =>
    val file = manager.outputsRoot.resolve(date).resolve(fileName).normalize()
    if (file.startsWith(manager.outputsRoot) && Files.isRegularFile(file)) {
      val extension = fileName.lastIndexOf('.') match {
        case -1 => ""
        case i  => fileName.substring(i + 1).toLowerCase
      }
      val contentType = GenerationManager.mimeTypeFor(extension)
      val size = Files.size(file)
      ByteRanges.of(range, size) match {
        case ByteRanges.Request.Whole =>
          Right(
            (StatusCode.Ok, contentType, None, size, FileRange(file.toFile))
          )
        case ByteRanges.Request.Part(first, last) =>
          Right(
            (
              StatusCode.PartialContent,
              contentType,
              Some(s"bytes $first-$last/$size"),
              last - first + 1,
              FileRange(
                file.toFile,
                Some(RangeValue(Some(first), Some(last), size))
              )
            )
          )
        case ByteRanges.Request.Unsatisfiable =>
          Left((StatusCode.RangeNotSatisfiable, Some(s"bytes */$size")))
      }
    } else Left((StatusCode.NotFound, None))
  }
)
