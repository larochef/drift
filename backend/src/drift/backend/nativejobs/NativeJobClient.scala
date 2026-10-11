package drift.backend.nativejobs

import drift.protocol.{JobRoutes, NativeJob}
import drift.shared.SessionCapabilities

import java.net.URI
import java.net.http.*
import scala.concurrent.duration.{DurationInt, FiniteDuration}
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger
import ox.sleep

/** The one client of the job protocol (`drift.protocol`): a job submitted to a
  * server on this machine, followed until it ends, cancelled — whichever server
  * it is, sd-server or drift's runner, and whoever asks, a generation or a tile
  * of a post-processing job.
  */
object NativeJobClient {
  private val logger = Logger("drift.backend.nativejobs.NativeJobClient")

  /** How long a job's polls may time out before it is abandoned. Generous for
    * the same reason as `readinessTimeoutMinutes`: a hires/ESRGAN pass can hold
    * the machine for a long while, and abandoning a running job loses its
    * output.
    */
  val SilenceGraceMillis: Long = 15L * 60 * 1000

  /** How many polls in a row may fail, other than by timing out. */
  private val FailuresTolerated = 5

  private val client = HttpClient
    .newBuilder()
    .connectTimeout(java.time.Duration.ofSeconds(3))
    .build()

  /** A completed job is one base64 picture or video: megabytes of text, far
    * above the 4M characters a JSON string is read up to unless told otherwise.
    */
  private val readerConfig = ReaderConfig.withMaxCharBufSize(256 * 1024 * 1024)

  private def uri(port: Int, path: String): URI =
    URI.create(s"http://127.0.0.1:$port$path")

  private def get(port: Int, path: String, seconds: Long) =
    client.send(
      HttpRequest
        .newBuilder(uri(port, path))
        .timeout(java.time.Duration.ofSeconds(seconds))
        .GET()
        .build(),
      HttpResponse.BodyHandlers.ofString()
    )

  /** What the server is and takes, or why it could not be read. */
  def capabilities(port: Int): Either[String, SessionCapabilities] =
    try {
      val response = get(port, JobRoutes.Capabilities, 30)
      if (response.statusCode != 200)
        Left(
          s"the server answered its capabilities with HTTP ${response.statusCode}"
        )
      else Right(readFromString[SessionCapabilities](response.body))
    } catch {
      case NonFatal(err) =>
        Left(s"reading the server's capabilities failed: ${firstLine(err)}")
    }

  /** Posts a job's request, already written, to `path`: the job as the server
    * took it, or why it refused.
    */
  def submit(port: Int, path: String, body: String): Either[String, NativeJob] =
    try {
      val response = client.send(
        HttpRequest
          .newBuilder(uri(port, path))
          // A request carries its pictures and videos: seconds to read.
          .timeout(java.time.Duration.ofSeconds(120))
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(body))
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      if (response.statusCode == 202)
        Right(readFromString[NativeJob](response.body))
      else
        Left(
          s"the server refused the job (${response.statusCode}): ${response.body.trim.take(500)}"
        )
    } catch {
      case NonFatal(err) =>
        Left(s"submitting the job failed: ${firstLine(err)}")
    }

  /** Asks the server to cancel a job it accepted. Whoever follows the job sees
    * it end `cancelled`; a server past the point its build can cancel answers
    * 409 and the job simply finishes, so nothing here is an error worth failing
    * over — whoever cancelled already knows what they asked for.
    */
  def cancel(port: Int, jobId: String): Unit =
    try {
      val response = client.send(
        HttpRequest
          .newBuilder(uri(port, JobRoutes.cancel(jobId)))
          .timeout(java.time.Duration.ofSeconds(10))
          .POST(HttpRequest.BodyPublishers.noBody())
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      if (response.statusCode != 200)
        logger.info(
          s"The server answered ${response.statusCode} to cancelling job $jobId"
        )
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Cancelling job $jobId failed", err)
    }

  /** Follows a job, a poll a second, until it ends — completed, failed or
    * cancelled — and gives it as it ended; or the reason the following was
    * given up. Each document read is handed to `onDocument` first.
    *
    * One policy, for every job:
    *   - A poll that times out means a busy server, not a dead one: a heavy
    *     pass can starve its HTTP loop for minutes. Silence is judged by how
    *     long it has lasted (`SilenceGraceMillis`), never by how many polls
    *     sampled it.
    *   - Any other failure — an error status, a document that does not read, an
    *     `onDocument` that throws — counts, and five in a row end the
    *     following. The count is reset only once a document has been read and
    *     handed over: reset on the bare answer, a document that never reads
    *     would be fetched for ever.
    *   - A server that no longer knows the job (404, 410) ends it at once, and
    *     so does one that has died (`serverExit`, its exit code once it has).
    *   - `stillWanted` turning false ends it too, without a reason: whoever
    *     follows has lost interest (a generation ended some other way).
    */
  def follow(
      port: Int,
      jobId: String,
      label: String,
      serverExit: () => Option[Int] = () => None,
      stillWanted: () => Boolean = () => true,
      every: FiniteDuration = 1.second
  )(onDocument: NativeJob => Unit): Either[String, Option[NativeJob]] = {
    var lastAnswerAt = System.currentTimeMillis()
    def silentMillis: Long = System.currentTimeMillis() - lastAnswerAt
    var failures = 0
    var outcome: Option[Either[String, Option[NativeJob]]] = None
    def failedAgain(reason: String): Unit = {
      failures += 1
      logger.warn(
        s"$label: poll failed ($failures/$FailuresTolerated): $reason"
      )
      if (failures >= FailuresTolerated) outcome = Some(Left(reason))
    }
    while (outcome.isEmpty) {
      serverExit().foreach(code => outcome = Some(Left(exited(code))))
      if (outcome.isEmpty && !stillWanted()) outcome = Some(Right(None))
      if (outcome.isEmpty)
        try {
          val response = get(port, JobRoutes.job(jobId), 10)
          response.statusCode match {
            case 200 =>
              val job = readFromString[NativeJob](response.body, readerConfig)
              onDocument(job)
              failures = 0
              lastAnswerAt = System.currentTimeMillis()
              job.status match {
                case "completed" | "failed" | "cancelled" =>
                  outcome = Some(Right(Some(job)))
                case _ => ()
              }
            case 404 | 410 =>
              outcome = Some(Left("the server no longer knows the job"))
            case other =>
              failedAgain(s"polling the job kept failing (HTTP $other)")
          }
        } catch {
          case _: HttpTimeoutException =>
            logger.warn(
              s"$label: poll timed out (server silent for ${silentMillis / 1000}s)"
            )
            if (silentMillis > SilenceGraceMillis)
              outcome = Some(
                Left(
                  s"the server answered nothing for ${silentMillis / 60000} " +
                    "minutes — giving up on the job"
                )
              )
          case NonFatal(err) =>
            failedAgain(s"handling the job kept failing: ${firstLine(err)}")
        }
      if (outcome.isEmpty) sleep(every)
    }
    outcome.get
  }

  /** A dead server's reason: its exit code, and the signal when one killed it
    * (the out-of-memory killer's 9, for instance).
    */
  def exited(code: Int): String =
    if (code > 128)
      s"the server was killed (exit code $code, signal ${code - 128})"
    else s"the server exited with code $code"

  /** The first line of an exception's message, bounded: jsoniter's parse errors
    * append a multi-line dump of the bytes that has no business in a reason
    * shown to the user.
    */
  private def firstLine(err: Throwable): String = {
    val line = Option(err.getMessage)
      .getOrElse(err.toString)
      .linesIterator
      .nextOption()
      .getOrElse("")
    if (line.length > 200) line.take(200) + "…" else line
  }
}
