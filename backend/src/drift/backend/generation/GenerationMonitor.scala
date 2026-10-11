package drift.backend.generation

import drift.backend.Background
import drift.backend.nativejobs.NativeJobClient
import drift.backend.outputs.GenerationFiles
import drift.protocol.*
import drift.shared.*

import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Follows a submitted job to its end: one fork per job polls the native job
  * every second, mirrors its state into the generation, and persists the result
  * the moment it completes — whether or not a browser is still watching. A
  * job's poll finds the step's preview (`specs/54-step-preview.md`) when its
  * request asked for one.
  */
final private[backend] class GenerationMonitor(
    registry: GenerationRegistry,
    files: GenerationFiles,
    background: Background
) {
  private val logger = Logger[GenerationMonitor]

  /** Follows the generation's job to its end (`NativeJobClient.follow`),
    * mirroring each document it reads into the generation.
    */
  def watch(entry: GenerationEntry): Unit =
    background.start(s"drift-generation-${entry.generation.id}") {
      NativeJobClient.follow(
        entry.port,
        entry.nativeJobId,
        label = s"Generation ${entry.generation.id}",
        stillWanted = () => entry.generation.status.isActive
      )(mirror(entry, _)) match {
        case Left(reason) => fail(entry, reason)
        case Right(_)     => ()
      }
      entry.preview = None
    }

  /** Mirrors one native job document into the drift generation. */
  private def mirror(entry: GenerationEntry, job: NativeJob): Unit = {
    val base = entry.generation.copy(
      queuePosition = job.queuePosition,
      startedAt = job.started.map(_ * 1000),
      completedAt = job.completed.map(_ * 1000),
      warnings = job.warnings
    )
    job.status match {
      case "queued" =>
        entry.generation = base.copy(status = GenerationStatus.Queued)
      case "generating" =>
        entry.generation = base.copy(status = GenerationStatus.Generating)
        entry.preview = job.preview
          .filter(_.b64Json.nonEmpty)
          .map(preview =>
            StepPreview(
              math.max(preview.pass - 1, 0),
              preview.step,
              preview.totalSteps,
              preview.mimeType,
              preview.b64Json
            )
          )
      case "completed" =>
        val outputs =
          try files.persist(base, job.result.getOrElse(NativeJobResult()))
          catch {
            case NonFatal(err) =>
              logger.warn(
                s"Generation ${base.id}: persisting outputs failed: ${err.getMessage}"
              )
              fail(entry, s"persisting the outputs failed: ${err.getMessage}")
              return
          }
        entry.generation = base.copy(
          status = GenerationStatus.Completed,
          outputs = outputs
        )
        files.writeSidecar(entry.generation)
        logger.info(
          s"Generation ${base.id}: completed with ${outputs.size} output(s)"
        )
      case "cancelled" =>
        entry.generation = base.copy(
          status = GenerationStatus.Cancelled,
          error = job.error.map(_.message)
        )
      case "failed" =>
        entry.generation = base.copy(
          status = GenerationStatus.Failed,
          error = Some(
            job.error
              .map(e => s"${e.code}: ${e.message}")
              .getOrElse("generation failed")
          )
        )
      case other =>
        logger.warn(s"Generation ${base.id}: unknown job status '$other'")
    }
  }

  /** A generation that has already ended is left alone: a forced cancel takes
    * its server down with it, so the polls that follow fail as a consequence of
    * what the user asked for, not as a failure of their own.
    */
  private def fail(entry: GenerationEntry, reason: String): Unit =
    if (entry.generation.status.isActive)
      entry.generation = entry.generation.copy(
        status = GenerationStatus.Failed,
        error = Some(reason)
      )
}
