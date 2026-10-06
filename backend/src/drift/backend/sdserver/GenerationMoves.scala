package drift.backend.sdserver

import drift.backend.projects.{ProjectCovers, ProjectManager}
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Path

import com.typesafe.scalalogging.Logger

/** Moving gallery entries to a project, or out of any
  * (`specs/19-projects-and-prompt-versions.md`). Only the sidecars change: an
  * entry stays under its day, and its project and version are rewritten.
  *
  *   - What was derived from a moved entry — its upscales, redraws, edits, and
  *     theirs — follows it, with its version: a derived entry belongs with its
  *     source.
  *   - A generation with a recipe becomes a version of the project it arrives
  *     in, the one already holding that recipe when there is one. An import, or
  *     a derived entry moved without its source, has none and shows as
  *     untagged.
  *   - A project whose chosen cover has left goes back to its newest result.
  */
final class GenerationMoves(
    outputsRoot: Path,
    history: GenerationHistory,
    generationManager: GenerationManager,
    projectManager: ProjectManager,
    covers: ProjectCovers,
    storage: StorageService
) {
  private val logger = Logger[GenerationMoves]
  private val files = GenerationFiles(outputsRoot)

  def move(request: GenerationMove): Either[String, List[Generation]] =
    request.projectId.filter(
      storage.get[Project]("projects", _).isEmpty
    ) match {
      case Some(missing) => Left(s"project '$missing' does not exist")
      case None          =>
        val roots = request.generations
          .flatMap(reference =>
            history
              .day(reference.date)
              .find(_.id == reference.generationId)
              .map(Filed(reference.date, _))
          )
          .filterNot(_.generation.status.isActive)
        val moved = withDerived(roots)
        // Sources before what was made from them, so a derived entry finds
        // the version its source was given
        val versions = scala.collection.mutable.Map.empty[String, String]
        val rewritten = moved.map { case Filed(date, generation) =>
          val version = request.projectId.flatMap { projectId =>
            generation.derivation.map(_.parentId) match {
              case Some(parentId)                => versions.get(parentId)
              case None if hasRecipe(generation) =>
                projectManager
                  .adopt(projectId, generation)
                  .toOption
                  .flatten
                  .map(_._2)
              case None => None
            }
          }
          version.foreach(versions.put(generation.id, _))
          val updated = generation.copy(
            projectId = request.projectId,
            promptVersionId = version
          )
          // Back where it was read: a derived entry sits under its source's
          // day, which is not the day its own timestamp names
          files.writeSidecar(updated, date)
          generationManager.replace(updated)
          updated
        }
        clearCovers(moved.map(_.generation))
        covers.invalidate()
        logger.info(
          s"Moved ${roots.size} generation(s), ${rewritten.size} entries " +
            s"with what was derived from them, to " +
            request.projectId.fold("no project")(id => s"project $id")
        )
        Right(rewritten)
    }

  /** A generation and the day its sidecar is filed under. */
  final private case class Filed(date: String, generation: Generation)

  private def hasRecipe(generation: Generation): Boolean =
    generation.imageParameters.isDefined ||
      generation.videoParameters.isDefined

  /** The entries and everything derived from them, each source before what was
    * made from it. A derived entry names its source, not the reverse, so
    * finding them reads every day once.
    */
  private def withDerived(roots: List[Filed]): List[Filed] =
    if (roots.isEmpty) Nil
    else {
      val children = history.days
        .flatMap(day => history.day(day.date).map(Filed(day.date, _)))
        .filter(_.generation.derivation.isDefined)
        .groupBy(_.generation.derivation.get.parentId)
      val seen = scala.collection.mutable.LinkedHashMap.empty[String, Filed]
      def visit(filed: Filed): Unit =
        if (!seen.contains(filed.generation.id)) {
          seen.put(filed.generation.id, filed)
          children.getOrElse(filed.generation.id, Nil).foreach(visit)
        }
      // A derived entry picked together with its source is reached from it
      val ids = roots.map(_.generation.id).toSet
      roots
        .sortBy(_.generation.derivation.exists(d => ids(d.parentId)))
        .foreach(visit)
      seen.values.toList
    }

  /** A chosen cover that is one of the entries leaving its project. */
  private def clearCovers(moved: List[Generation]): Unit =
    moved
      .groupBy(_.projectId)
      .foreach {
        case (Some(projectId), generations) =>
          storage
            .get[Project]("projects", projectId)
            .filter(
              _.cover.exists(cover =>
                generations.exists(_.outputs.exists(cover.isOf))
              )
            )
            .foreach(project =>
              storage.save("projects", projectId, project.copy(cover = None))
            )
        case _ => ()
      }
}
