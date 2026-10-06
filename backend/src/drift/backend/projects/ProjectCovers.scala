package drift.backend.projects

import drift.backend.images.{Thumbnail, VideoFrames}
import drift.backend.sdserver.GenerationHistory
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.{Files, Path}
import java.util.concurrent.ConcurrentHashMap
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** The cover of each project, served as bytes
  * (`specs/19-projects-and-prompt-versions.md`): the output the user chose, or
  * else the newest result - its newest image for an image project, its newest
  * video for a video project (`specs/31-project-kinds.md`).
  *
  * The sidecars are the only record of what belongs to a project, so finding
  * the newest means walking the outputs root. Doing that per request, per
  * project, would be one walk per tile on the list - and the worst case is a
  * project that never generated anything, which reads *every* sidecar before
  * answering "none". So the walk happens once for all projects and is
  * remembered.
  *
  * The memo is checked against the newest day directory on each request, which
  * is one listing: a completed generation changes it, and the covers are
  * rebuilt. A cover whose file has since been deleted rebuilds too, which is
  * what makes deleting from an older day self-correcting without watching for
  * it - and so does a project whose kind changed, since that changes which
  * outputs can be its cover.
  */
final class ProjectCovers(
    storage: StorageService,
    history: GenerationHistory,
    outputsRoot: Path,
    maxSide: Int = ProjectCovers.DefaultMaxSide
) {

  /** An image is scaled once and held as JPEG bytes; a video too, from its
    * first frame (read by ffmpeg): the list shows stills, only the players load
    * videos (bug 37). `thumbnail` is empty only for a video ffmpeg could not
    * read, served as it is.
    */
  final private case class Entry(
      source: Path,
      mimeType: String,
      thumbnail: Option[Array[Byte]]
  )

  /** The newest result of each project. */
  private val entries = ConcurrentHashMap[String, Entry]()

  /** The chosen cover of each project, as last served. */
  private val chosenEntries = ConcurrentHashMap[String, Entry]()

  @volatile private var signature: Option[String] = None

  /** The kind each project had when the newest results were last found. */
  @volatile private var kinds: Map[String, ProjectKind] = Map.empty

  /** One directory listing: the newest day and how many sidecars it holds.
    * Enough to notice a generation completing, which is the only thing that
    * changes a cover in the ordinary course of events.
    */
  private def currentSignature: String =
    history.days.headOption
      .map(day => s"${day.date}:${day.count}")
      .getOrElse("empty")

  /** Forgets the newest results: a generation changed project, which no
    * directory listing shows.
    */
  def invalidate(): Unit = synchronized { signature = None }

  /** The cover for a project as bytes and their content type, or `None` when it
    * has made nothing of its kind yet - which is not a failure, just a project
    * the user has not run.
    */
  def cover(projectId: String): Option[(Array[Byte], String)] = synchronized {
    val project = storage.get[Project]("projects", projectId)
    project
      .flatMap(chosen)
      .orElse(newest(projectId, project.map(_.kind)))
  }

  /** The cover the user chose, while it can still be one: its file is there and
    * it is of the project's kind. Otherwise the newest result stands in, so
    * deleting the chosen generation or changing the kind needs nothing cleared.
    * The name arrives with a project update, from the client, so it must
    * resolve inside the outputs root.
    */
  private def chosen(project: Project): Option[(Array[Byte], String)] = {
    val root = outputsRoot.toAbsolutePath.normalize
    project.cover
      .filter(_.mimeType.startsWith(s"${project.kind.noun}/"))
      .map(cover =>
        (
          root.resolve(cover.date).resolve(cover.fileName).normalize,
          cover.mimeType
        )
      )
      .filter((source, _) =>
        source.startsWith(root) && Files.isRegularFile(source)
      )
      .flatMap { (source, mimeType) =>
        Option(chosenEntries.get(project.id))
          .filter(_.source == source)
          .orElse(entryOf(source, mimeType).map { entry =>
            chosenEntries.put(project.id, entry)
            entry
          })
      }
      .flatMap(served)
  }

  private def newest(
      projectId: String,
      kind: Option[ProjectKind]
  ): Option[(Array[Byte], String)] = {
    val now = currentSignature
    val stale = !signature.contains(now) ||
      kind != kinds.get(projectId) ||
      Option(entries.get(projectId)).exists(entry =>
        !Files.isRegularFile(entry.source)
      )
    if (stale) rebuild(now)
    Option(entries.get(projectId)).flatMap(served)
  }

  private def served(entry: Entry): Option[(Array[Byte], String)] =
    entry.thumbnail match {
      case Some(bytes) => Some((bytes, entry.mimeType))
      case None        =>
        try Some((Files.readAllBytes(entry.source), entry.mimeType))
        catch { case NonFatal(_) => None }
    }

  private def rebuild(currentSignature: String): Unit = {
    val wanted =
      storage
        .list[Project]("projects")
        .map(project => project.id -> project.kind)
        .toMap
    val found = scala.collection.mutable.Map.empty[String, GenerationOutput]
    history.days.iterator
      .takeWhile(_ => found.size < wanted.size)
      .foreach { day =>
        history.day(day.date).foreach { generation =>
          generation.projectId
            .filterNot(found.contains)
            .foreach { id =>
              wanted.get(id).foreach { kind =>
                generation.outputs
                  .find(_.mimeType.startsWith(s"${kind.noun}/"))
                  .foreach(output => found.put(id, output))
              }
            }
        }
      }
    entries.clear()
    found.foreach { (id, output) =>
      entryOf(
        outputsRoot.resolve(output.date).resolve(output.fileName),
        output.mimeType
      ).foreach(entry => entries.put(id, entry))
    }
    kinds = wanted
    signature = Some(currentSignature)
  }

  private def entryOf(source: Path, mimeType: String): Option[Entry] =
    if (mimeType.startsWith("video/"))
      Option.when(Files.isRegularFile(source))(
        VideoFrames
          .first(source)
          .map(frame =>
            Entry(source, "image/jpeg", Some(Thumbnail.jpeg(frame, maxSide)))
          )
          .getOrElse(Entry(source, mimeType, None))
      )
    else
      // Scaled once and held: a cover is small, and the alternative is
      // decoding a multi-megabyte PNG on every visit to the list.
      thumbnailOf(source).map(bytes => Entry(source, "image/jpeg", Some(bytes)))

  private def thumbnailOf(source: Path): Option[Array[Byte]] =
    try
      if (!Files.isRegularFile(source)) None
      else Option(ImageIO.read(source.toFile)).map(Thumbnail.jpeg(_, maxSide))
    catch {
      // A format ImageIO cannot read (WebP has no reader) is a project without
      // a cover, not an error worth surfacing.
      case NonFatal(_) => None
    }
}

object ProjectCovers {

  /** Wide enough for a tile on a high-density screen, small enough that a dozen
    * of them cost nothing to send or decode.
    */
  val DefaultMaxSide: Int = 640
}
