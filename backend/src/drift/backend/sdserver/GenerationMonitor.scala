package drift.backend.sdserver

import drift.shared.*

import java.net.http.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger

/** Follows a submitted job to its end: one daemon thread per job polls the
  * native job every second, mirrors its state into the generation, and persists
  * the result the moment it completes — whether or not a browser is still
  * watching.
  */
final private[sdserver] class GenerationMonitor(
    registry: GenerationRegistry,
    files: GenerationFiles
) {
  private val logger = Logger[GenerationMonitor]

  /** A completed video job is one giant base64 string — a 3-second webm already
    * blows jsoniter's default 4M-char buffer cap (a real job failed at 5.9 MB).
    * 256M chars covers any clip drift plausibly handles.
    */
  private val nativeJobReaderConfig =
    ReaderConfig.withMaxCharBufSize(256 * 1024 * 1024)

  def watch(entry: GenerationEntry): Unit = {
    val thread = Thread(
      new Runnable {
        def run(): Unit = {
          var consecutiveFailures = 0
          // When sd-server last answered a poll. A hires/ESRGAN pass (or any
          // compute saturating the GPU and every core) can starve the HTTP
          // loop for minutes while the job is still running, so timeouts are
          // judged by how long the silence has lasted, not how many polls
          // sampled it — five timed-out polls used to abandon a healthy job
          // after less than a minute.
          var lastAnswerAt = System.currentTimeMillis()
          def silentMillis: Long = System.currentTimeMillis() - lastAnswerAt
          while (entry.generation.status.isActive) {
            try {
              val response = registry.send(
                HttpRequest
                  .newBuilder(
                    registry.uri(
                      entry.port,
                      s"/sdcpp/v1/jobs/${entry.nativeJobId}"
                    )
                  )
                  .timeout(java.time.Duration.ofSeconds(10))
                  .GET()
                  .build()
              )
              response.statusCode match {
                case 200 =>
                  mirror(
                    entry,
                    readFromString[NativeJob](
                      response.body,
                      nativeJobReaderConfig
                    )
                  )
                  // Only after the document was decoded AND applied: resetting
                  // on the bare 200 let a repeatable decode failure poll
                  // forever — the completed job was re-fetched every second
                  // until sd-server's TTL evicted it, which then surfaced as
                  // the misleading "no longer knows the job".
                  consecutiveFailures = 0
                  lastAnswerAt = System.currentTimeMillis()
                case 404 | 410 =>
                  fail(entry, "sd-server no longer knows the job")
                case other =>
                  consecutiveFailures += 1
                  if (consecutiveFailures >= 5)
                    fail(entry, s"polling the job kept failing (HTTP $other)")
              }
            } catch {
              // A timeout means busy, not dead: the process dying turns polls
              // into instant connection refusals, which the branch below
              // fails fast. Tolerate silence for as long as a heavy pass can
              // plausibly last, and let Stop/Cancel remain the way out of a
              // genuinely wedged server.
              case err: HttpTimeoutException =>
                logger.warn(
                  s"Generation ${entry.generation.id}: poll timed out " +
                    s"(server silent for ${silentMillis / 1000}s)"
                )
                if (silentMillis > GenerationManager.SilenceGraceMillis)
                  fail(
                    entry,
                    s"sd-server answered nothing for ${silentMillis / 60000} " +
                      "minutes — giving up on the job"
                  )
              case NonFatal(err) =>
                consecutiveFailures += 1
                logger.warn(
                  s"Generation ${entry.generation.id}: poll failed " +
                    s"($consecutiveFailures/5): ${firstLine(err)}"
                )
                if (consecutiveFailures >= 5)
                  fail(
                    entry,
                    s"handling the job kept failing: ${firstLine(err)}"
                  )
            }
            if (entry.generation.status.isActive) Thread.sleep(1000)
          }
        }
      },
      s"drift-generation-${entry.generation.id}"
    )
    thread.setDaemon(true)
    thread.start()
  }

  /** Mirrors one native job document into the drift generation. */
  private def mirror(entry: GenerationEntry, job: NativeJob): Unit = {
    val base = entry.generation.copy(
      queuePosition = job.queuePosition,
      startedAt = job.started.map(_ * 1000),
      completedAt = job.completed.map(_ * 1000)
    )
    job.status match {
      case "queued" =>
        entry.generation = base.copy(status = GenerationStatus.Queued)
      case "generating" =>
        entry.generation = base.copy(status = GenerationStatus.Generating)
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

  /** The first line of an exception message, bounded — jsoniter's parse errors
    * append a multi-line hex dump that has no business in the UI.
    */
  private def firstLine(err: Throwable): String = {
    val line = Option(err.getMessage)
      .getOrElse(err.toString)
      .linesIterator
      .nextOption()
      .getOrElse("")
    if (line.length > 200) line.take(200) + "…" else line
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
