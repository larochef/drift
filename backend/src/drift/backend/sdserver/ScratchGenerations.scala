package drift.backend.sdserver

import drift.backend.projects.ProjectManager
import drift.shared.*

import java.nio.file.*
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Free play's results (`specs/22-free-play-and-scratch-generations.md`): kept
  * one by one into the gallery or a project, and otherwise cleared when a
  * session stops and at startup.
  */
final private[sdserver] class ScratchGenerations(
    outputsRoot: Path,
    registry: GenerationRegistry,
    files: GenerationFiles,
    projectManager: ProjectManager
) {
  private val logger = Logger[ScratchGenerations]

  /** Promotes one free-play generation to a kept one: its files move from
    * `outputs/scratch/` into the day it was submitted, the recorded input URLs
    * follow them, a named project turns the recipe into one of its versions the
    * way a submission would, and only then is the sidecar written — which is
    * what puts it in the gallery.
    *
    * Answers `None` for an id that is not a finished scratch generation: it was
    * kept already, it is still running, or its session is gone and took the
    * files with it.
    */
  def keep(
      generationId: String,
      projectId: Option[String]
  ): Option[Generation] =
    registry
      .get(generationId)
      .map(_.generation)
      .filter(generation => generation.scratch && !generation.status.isActive)
      .flatMap { generation =>
        val date = files.dateOf(generation.submittedAt)
        val target = outputsRoot.resolve(date)
        val scratchRoot =
          outputsRoot.resolve(GenerationManager.ScratchDirectory)
        try {
          Files.createDirectories(target)
          files
            .filesOwnedBy(scratchRoot, generationId)
            .foreach(file =>
              Files.move(
                file,
                target.resolve(file.getFileName),
                StandardCopyOption.REPLACE_EXISTING
              )
            )
          val moved = generation.copy(
            scratch = false,
            outputs = generation.outputs.map(output =>
              output.copy(
                date = date,
                url = s"/api/outputs/$date/${output.fileName}"
              )
            ),
            imageParameters = generation.imageParameters.map(p =>
              p.copy(
                initImage = p.initImage.map(files.rehome(_, date)),
                maskImage = p.maskImage.map(files.rehome(_, date)),
                refImages = p.refImages.map(files.rehome(_, date))
              )
            ),
            videoParameters = generation.videoParameters.map(p =>
              p.copy(
                initImage = p.initImage.map(files.rehome(_, date)),
                endImage = p.endImage.map(files.rehome(_, date)),
                controlFrames = p.controlFrames.map(files.rehome(_, date))
              )
            )
          )
          val tagged = projectId
            .flatMap { project =>
              projectManager
                .versionFor(
                  SubmitContext(
                    projectId = Some(project),
                    origin = Some("manual")
                  ),
                  moved
                )
                .toOption
                .flatten
            }
            .map((project, version) =>
              moved.copy(
                projectId = Some(project),
                promptVersionId = Some(version)
              )
            )
            .getOrElse(moved)
          files.writeSidecar(tagged)
          registry.get(generationId).foreach(_.generation = tagged)
          logger.info(
            s"Kept generation $generationId into $date" +
              tagged.projectId.map(id => s" (project $id)").getOrElse("")
          )
          Some(tagged)
        } catch {
          case NonFatal(err) =>
            logger.warn(
              s"Keeping generation $generationId failed: ${err.getMessage}"
            )
            None
        }
      }

  /** Empties `outputs/scratch/` and forgets the records that pointed into it,
    * answering with the ids that went. A generation still running is left
    * alone, files and record both: the button is reachable while a job is in
    * flight.
    */
  def clear(): List[String] = {
    val running = registry.all
      .filter(entry =>
        entry.generation.scratch && entry.generation.status.isActive
      )
      .map(_.generation.id)
      .toSet
    val dropped = registry.all
      .filter(entry =>
        entry.generation.scratch && !entry.generation.status.isActive
      )
      .map(_.generation.id)
    dropped.foreach(registry.remove)
    val directory = outputsRoot.resolve(GenerationManager.ScratchDirectory)
    if (Files.isDirectory(directory))
      try {
        val kept = Using.resource(Files.list(directory)) { stream =>
          stream.iterator.asScala.toList.count { file =>
            val name = file.getFileName.toString
            val owned = running.exists(id =>
              name.startsWith(s"$id-") || name.startsWith(s"$id.")
            )
            if (!owned) Files.deleteIfExists(file)
            owned
          }
        }
        if (kept == 0) Files.deleteIfExists(directory)
      } catch {
        case NonFatal(err) =>
          logger.warn(
            s"Clearing the scratch directory failed: ${err.getMessage}"
          )
      }
    if (dropped.nonEmpty)
      logger.info(s"Cleared ${dropped.size} scratch generation(s)")
    dropped
  }
}
