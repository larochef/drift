package drift.backend.sdserver

import drift.shared.*

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromArray
import com.typesafe.scalalogging.Logger

/** The gallery's view of the outputs root (`specs/12-gallery.md`): the
  * `<id>.json` sidecars `GenerationManager` writes on completion, read back by
  * day. No database and no scan of image bytes — the date directories are the
  * index, and a day's listing is one directory walk.
  *
  * Decode failures are skipped loudly, the `StorageService.list` convention:
  * pre-release, a sidecar an older build wrote in a shape the current
  * `Generation` no longer decodes is an acceptable loss, not a reason for
  * tolerant decoders.
  */
final class GenerationHistory(
    outputsRoot: Path,
    generationManager: GenerationManager
) {
  private val logger = Logger[GenerationHistory]

  /** Every date directory holding at least one sidecar, newest first. */
  def days: List[HistoryDay] =
    if (!Files.isDirectory(outputsRoot)) List.empty
    else
      Using.resource(Files.list(outputsRoot)) { stream =>
        stream.iterator.asScala
          .filter(directory =>
            Files.isDirectory(directory) &&
              GenerationHistory.isDate(directory.getFileName.toString)
          )
          .map(directory =>
            HistoryDay(
              directory.getFileName.toString,
              sidecarsIn(directory).size
            )
          )
          .filter(_.count > 0)
          .toList
          .sortBy(_.date)(using Ordering[String].reverse)
      }

  /** One day's generations, newest first; empty for an unknown or malformed
    * date.
    */
  def day(date: String): List[Generation] =
    if (!GenerationHistory.isDate(date)) List.empty
    else {
      val directory = outputsRoot.resolve(date)
      if (!Files.isDirectory(directory)) List.empty
      else
        sidecarsIn(directory)
          .flatMap { sidecar =>
            try Some(readFromArray[Generation](Files.readAllBytes(sidecar)))
            catch {
              case NonFatal(err) =>
                logger.warn(
                  s"Skipping undecodable sidecar ${sidecar.getFileName}: ${err.getMessage}"
                )
                None
            }
          }
          .sortBy(g => (-g.submittedAt, g.id))
    }

  /** The day one generation's sidecar sits in, or nothing when no generation by
    * that id is recorded. One existence check per date directory: the sidecar
    * is named after the generation, so no day has to be read to find it.
    */
  def dayOf(generationId: String): Option[HistoryDay] =
    if (
      !GenerationHistory.isGenerationId(generationId) ||
      !Files.isDirectory(outputsRoot)
    ) None
    else
      Using.resource(Files.list(outputsRoot)) { stream =>
        stream.iterator.asScala
          .filter(directory =>
            Files.isDirectory(directory) &&
              GenerationHistory.isDate(directory.getFileName.toString) &&
              Files.isRegularFile(directory.resolve(s"$generationId.json"))
          )
          .nextOption()
          .map(directory =>
            HistoryDay(
              directory.getFileName.toString,
              sidecarsIn(directory).size
            )
          )
      }

  /** Removes every file the generation left under its day — outputs
    * (`<id>-<index>.<format>` / `<id>.<format>`), externalized inputs
    * (`<id>-init.png` and the like) and the sidecar. The naming scheme is
    * `GenerationManager`'s, and an id is never a prefix of another id's files:
    * `g1-5` matches `g1-5-…` and `g1-5.…`, never `g1-50-…`. The in-memory
    * record of a live session is dropped too, so the panel stops showing a
    * result whose file is gone. Answers false when nothing matched.
    */
  def delete(date: String, generationId: String): Boolean =
    if (
      !GenerationHistory.isDate(date) ||
      !GenerationHistory.isGenerationId(generationId)
    ) false
    else {
      val directory = outputsRoot.resolve(date)
      if (!Files.isDirectory(directory)) false
      else {
        val owned = Using.resource(Files.list(directory)) { stream =>
          stream.iterator.asScala.filter { file =>
            val name = file.getFileName.toString
            name.startsWith(s"$generationId-") ||
            name.startsWith(s"$generationId.")
          }.toList
        }
        if (owned.isEmpty) false
        else {
          owned.foreach(Files.deleteIfExists)
          generationManager.forget(generationId)
          logger.info(
            s"Deleted generation $generationId ($date): ${owned.size} file(s)"
          )
          // A day with nothing left in it disappears from the listing on its
          // own, but not from the disk without this.
          val empty = Using.resource(Files.list(directory))(_.findAny.isEmpty)
          if (empty) Files.deleteIfExists(directory)
          true
        }
      }
    }

  private def sidecarsIn(directory: Path): List[Path] =
    Using.resource(Files.list(directory)) { stream =>
      stream.iterator.asScala
        .filter(file =>
          Files.isRegularFile(file) &&
            file.getFileName.toString.endsWith(".json")
        )
        .toList
    }
}

object GenerationHistory {
  private val DatePattern = """\d{4}-\d{2}-\d{2}""".r
  private val GenerationIdPattern = """g\d+-\d+""".r

  def isDate(value: String): Boolean = DatePattern.matches(value)
  def isGenerationId(value: String): Boolean =
    GenerationIdPattern.matches(value)
}
