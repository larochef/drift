package drift.backend.sdserver

import drift.backend.Background
import drift.backend.session.SessionManager
import drift.shared.*

import java.net.http.*
import java.nio.file.*
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger

/** Proxies generation to a session's `sd-server` and records every result
  * (`specs/08-inference-ui.md`).
  *
  * drift keeps no queue of its own — sd-server already queues and reports
  * `queue_position`, and a second scheduler would fight the first. Each
  * submitted job gets one daemon thread polling the native job until it is
  * terminal, so a completed image is persisted even if no browser is watching.
  * Completed outputs and their parameter sidecar go under
  * `~/.local/share/drift/outputs/<date>/` — the user's work, deliberately not
  * in the deletable `~/.cache`.
  *
  * Generations are runtime state like sessions: on drift restart the in-memory
  * list is gone, and the sidecars on disk are what history rebuilds from.
  *
  * The work lives beside this file (`specs/29-split-oversized-files.md`):
  * `GenerationSubmissions`, `GenerationMonitor`, `GenerationFiles` and
  * `ScratchGenerations`, over the in-memory `GenerationRegistry`.
  */
final class GenerationManager(
    sessionManager: SessionManager,
    projectManager: drift.backend.projects.ProjectManager,
    val outputsRoot: Path,
    /** Where each job is followed to its end. */
    background: Background
) {
  private val logger = Logger[GenerationManager]
  private val registry = GenerationRegistry(sessionManager)
  private val files = GenerationFiles(outputsRoot)
  private val submissions = GenerationSubmissions(
    registry,
    files,
    GenerationMonitor(registry, files, background),
    projectManager,
    sessionManager.architectureAndRuntimeOf
  )
  private val scratchGenerations =
    ScratchGenerations(outputsRoot, registry, files, projectManager)

  /** What the session's loaded model supports, straight off its port. Empty for
    * an unknown or not-yet-ready session — the form has nothing to build from
    * until the server answers, and this is the same call readiness already
    * proved.
    */
  def capabilities(sessionId: String): Option[SessionCapabilities] =
    registry.readySession(sessionId).toOption.flatMap { (_, port) =>
      try {
        val response = registry.send(
          HttpRequest
            .newBuilder(registry.uri(port, "/sdcpp/v1/capabilities"))
            .timeout(java.time.Duration.ofSeconds(10))
            .GET()
            .build()
        )
        if (response.statusCode == 200)
          Some(readFromString[SessionCapabilities](response.body))
        else None
      } catch {
        case NonFatal(err) =>
          logger.warn(
            s"Capabilities of session $sessionId failed: ${err.getMessage}"
          )
          None
      }
    }

  def submitImage(
      sessionId: String,
      requested: ImageGenerationParameters,
      context: SubmitContext = SubmitContext(),
      /** Free play (`specs/22-…`): outputs go to `outputs/scratch/` and no
        * sidecar is written, until the user keeps it.
        */
      scratch: Boolean = false
  ): Generation = submissions.image(sessionId, requested, context, scratch)

  def submitVideo(
      sessionId: String,
      requested: VideoGenerationParameters,
      context: SubmitContext = SubmitContext(),
      scratch: Boolean = false
  ): Generation = submissions.video(sessionId, requested, context, scratch)

  def list(sessionId: String): List[Generation] =
    registry.all
      .map(_.generation)
      .filter(_.sessionId == sessionId)
      .sortBy(_.submittedAt)

  /** Every session's generations at once, keyed by session id, each list oldest
    * first — what the status socket samples, so one diff covers all sessions.
    */
  def listBySession: Map[String, List[Generation]] =
    registry.all
      .map(_.generation)
      .groupBy(_.sessionId)
      .view
      .mapValues(_.sortBy(_.submittedAt))
      .toMap

  /** Replaces the in-memory record of a generation whose sidecar was rewritten
    * — moved to another project — so a live session's list does not push the
    * old one back.
    */
  def replace(generation: Generation): Unit =
    registry.get(generation.id).foreach(_.generation = generation)

  /** Drops the in-memory record — what the gallery's delete calls once the
    * files are gone, so a live session's panel does not keep showing an output
    * that no longer exists. An active job is never forgotten: its monitor
    * thread would write a sidecar for a generation nobody lists.
    */
  def forget(generationId: String): Unit =
    registry
      .get(generationId)
      .filterNot(_.generation.status.isActive)
      .foreach(_ => registry.remove(generationId))

  /** Forwards the cancel and answers with the generation's current state; the
    * poll thread picks up the resulting `cancelled` status. sd-server may
    * refuse (409) a job past the point its build can cancel — the job then
    * simply completes.
    *
    * `force` is for the builds and models that cannot interrupt a generation at
    * all: the generation is marked cancelled here and its server is killed and
    * launched again on the same configuration (`SessionManager.restart`), which
    * costs a model reload and is the only thing short of killing drift. The
    * monitor stops at its next turn round the loop and leaves the state alone,
    * the polls that fail meanwhile being the consequence of the kill.
    */
  def cancel(generationId: String, force: Boolean = false): Option[Generation] =
    registry.get(generationId).map { entry =>
      if (entry.generation.status.isActive && entry.nativeJobId.nonEmpty)
        NativeJobs.cancel(entry.port, entry.nativeJobId)
      if (force && entry.generation.status.isActive) {
        val sessionId = entry.generation.sessionId
        val now = System.currentTimeMillis()
        // The queue lives in the server that is about to die, so every job
        // waiting on it ends here — said plainly, rather than surfacing a
        // second later as five polls that could not reach anything.
        registry.all
          .filter(other =>
            other.generation.sessionId == sessionId &&
              other.generation.status.isActive
          )
          .foreach { other =>
            other.generation = other.generation.copy(
              status = GenerationStatus.Cancelled,
              completedAt = Some(now),
              error = Some(
                if (other.generation.id == generationId)
                  "cancelled by restarting sd-cpp"
                else "dropped when sd-cpp was restarted"
              )
            )
          }
        logger.info(
          s"Generation $generationId: cancelled by restarting session $sessionId"
        )
        sessionManager.restart(sessionId)
      }
      entry.generation
    }

  /** Keeps a free-play generation (`ScratchGenerations.keep`). */
  def keep(
      generationId: String,
      projectId: Option[String]
  ): Option[Generation] = scratchGenerations.keep(generationId, projectId)

  /** Clears free play (`ScratchGenerations.clear`) — when a session stops and
    * at startup, which free play is not meant to survive.
    */
  def clearScratch(): List[String] = scratchGenerations.clear()
}

object GenerationManager {
  def randomSeed(): Long =
    java.util.concurrent.ThreadLocalRandom
      .current()
      .nextInt(Int.MaxValue)
      .toLong

  /** Where free play's outputs live under the outputs root. Not a date, which
    * is exactly why `GenerationHistory` — which only counts date directories —
    * never sees them (`specs/22-free-play-and-scratch-generations.md`).
    */
  val ScratchDirectory: String = "scratch"

  /** How long a job's polls may time out before it is abandoned. Generous for
    * the same reason as `readinessTimeoutMinutes`: a hires/ESRGAN pass can hold
    * the machine for a long while, and abandoning a running job loses its
    * output.
    */
  val SilenceGraceMillis: Long = 15L * 60 * 1000

  private[sdserver] val DateFormat =
    DateTimeFormatter.ofPattern("yyyy-MM-dd").withZone(ZoneId.systemDefault())

  /** The media drift writes and serves under the outputs root, extension ↔ MIME
    * type: its own outputs, and the inputs a generation carried — video models
    * take clips and sounds as well as images (`specs/42`, step 14). One table
    * both ways, so an input persisted from a data URL is served back with the
    * type it came with, and "reuse these parameters" rebuilds the same data URL
    * from it. The first extension of a type is the one written.
    */
  private val MediaTypes: List[(String, String)] = List(
    "png" -> "image/png",
    "jpeg" -> "image/jpeg",
    "jpg" -> "image/jpeg",
    "webp" -> "image/webp",
    "gif" -> "image/gif",
    "webm" -> "video/webm",
    "mp4" -> "video/mp4",
    "mov" -> "video/quicktime",
    "mkv" -> "video/x-matroska",
    "avi" -> "video/x-msvideo",
    "wav" -> "audio/wav",
    "mp3" -> "audio/mpeg",
    "ogg" -> "audio/ogg",
    "flac" -> "audio/flac",
    "m4a" -> "audio/mp4"
  )

  def mimeTypeFor(format: String): String =
    MediaTypes
      .collectFirst { case (`format`, mimeType) => mimeType }
      .getOrElse("application/octet-stream")

  /** The extension a file of this MIME type is written with. Browsers name some
    * types two ways (`audio/x-wav`, `audio/wave`, `audio/mp3`), so the aliases
    * are folded in first.
    */
  def extensionFor(mimeType: String): Option[String] = {
    val canonical = mimeType match {
      case "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => "audio/wav"
      case "audio/mp3"                                     => "audio/mpeg"
      case "audio/x-flac"                                  => "audio/flac"
      case "audio/x-m4a"                                   => "audio/mp4"
      case other                                           => other
    }
    MediaTypes.collectFirst { case (extension, `canonical`) => extension }
  }
}
