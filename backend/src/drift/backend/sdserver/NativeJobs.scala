package drift.backend.sdserver

import drift.backend.sdserver.ServerRequests.given
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.util.Base64
import scala.concurrent.duration.DurationInt
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.*
import com.typesafe.scalalogging.Logger
import ox.sleep

/** The native sdcpp job document, as `GET /sdcpp/v1/jobs/{id}` answers it.
  * Timestamps are unix seconds; drift converts to millis at the boundary.
  * `b64_json` is pinned by annotation: `enforce_snake_case` would split the
  * digit boundary into `b_64_json` and silently miss the field.
  */
private[backend] case class NativeJobImage(
    index: Int = 0,
    @named("b64_json") b64Json: String = ""
)
private[backend] case class NativeJobError(
    code: String = "",
    message: String = ""
)
private[backend] case class NativeJobResult(
    outputFormat: String = "",
    mimeType: Option[String] = None,
    fps: Option[Int] = None,
    frameCount: Option[Int] = None,
    images: List[NativeJobImage] = List.empty,
    @named("b64_json") b64Json: Option[String] = None
)
private[backend] case class NativeJob(
    id: String = "",
    kind: String = "",
    status: String = "",
    created: Long = 0,
    started: Option[Long] = None,
    completed: Option[Long] = None,
    queuePosition: Int = 0,
    result: Option[NativeJobResult] = None,
    error: Option[NativeJobError] = None
)

/** A video the runner's `upscale` job made. */
private[backend] case class UpscaledVideo(
    bytes: Array[Byte],
    format: String,
    mimeType: String,
    fps: Option[Int],
    frameCount: Option[Int]
)
private[backend] object NativeJob {
  given JsonValueCodec[NativeJob] = JsonCodecMaker.make(
    CodecMakerConfig.withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
  )
}

/** The runner's `upscale` job (`specs/51`): `source` is a picture or a video,
  * base64; a negative seed is the server's to draw.
  */
private[backend] case class NativeUpscale(
    source: String,
    scale: Int,
    seed: Long
)
private[backend] object NativeUpscale {
  given JsonValueCodec[NativeUpscale] = JsonCodecMaker.make
}

/** Images straight from an sd-server, for jobs that drive a server themselves
  * rather than through a session's generation list — the tiles of PiD and
  * redraw (`specs/26-tiled-pid.md`, `specs/27-redraw.md`).
  */
object NativeJobs {
  private val logger = Logger("drift.backend.sdserver.NativeJobs")

  private val client = HttpClient
    .newBuilder()
    .connectTimeout(java.time.Duration.ofSeconds(3))
    .build()

  /** A completed job is one base64 image; a 1536² PNG runs to megabytes. */
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

  /** The server's img_gen defaults — its launch flags — and its img_gen
    * features (`ref_images`, …), as its capabilities report them.
    */
  def imageCapabilities(
      port: Int
  ): Either[String, (GenerationDefaults, Map[String, Boolean])] =
    try {
      val response = get(port, "/sdcpp/v1/capabilities", 30)
      if (response.statusCode != 200)
        Left(
          s"sd-server answered its capabilities with HTTP ${response.statusCode}"
        )
      else {
        val capabilities = readFromString[SessionCapabilities](response.body)
        capabilities.defaultsByMode
          .get("img_gen")
          .toRight("sd-server reports no img_gen defaults")
          .map(_ -> capabilities.featuresByMode.getOrElse("img_gen", Map.empty))
      }
    } catch {
      case NonFatal(err) =>
        Left(s"reading sd-server's capabilities failed: ${err.getMessage}")
    }

  /** Submits `parameters` as an img_gen job and waits for its first image.
    * `serverExit` is the server's exit code once it has died, if drift runs it:
    * the wait then ends at once, naming it. `onSubmitted` is handed the native
    * job's id the moment the server accepts it, which is what a cancel needs to
    * reach it (`cancel`).
    */
  def image(
      port: Int,
      parameters: ImageGenerationParameters,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit
  ): Either[String, Array[Byte]] =
    try {
      val submitted = client.send(
        HttpRequest
          .newBuilder(uri(port, "/sdcpp/v1/img_gen"))
          .timeout(java.time.Duration.ofSeconds(30))
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(writeToString(parameters)))
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      if (submitted.statusCode != 202)
        Left(
          s"sd-server refused the job (${submitted.statusCode}): ${submitted.body.trim.take(300)}"
        )
      else {
        val jobId = readFromString[NativeJob](submitted.body).id
        onSubmitted(jobId)
        await(port, jobId, serverExit).flatMap(firstImage)
      }
    } catch {
      case NonFatal(err) =>
        Left(
          serverExit().fold(s"submitting the job failed: ${err.getMessage}")(
            exited
          )
        )
    }

  private def firstImage(result: NativeJobResult): Either[String, Array[Byte]] =
    result.images.headOption
      .filter(_.b64Json.nonEmpty)
      .map(image => Base64.getMimeDecoder.decode(image.b64Json))
      .toRight("sd-server completed the job without an image")

  /** Submits the runner's `upscale` job and waits for its result: a picture
    * (`Left`, a PNG) or a video (`Right`: its bytes, format, frame rate and
    * frame count). `serverExit` and `onSubmitted` as `image` has them.
    */
  def upscale(
      port: Int,
      request: NativeUpscale,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit
  ): Either[String, Either[Array[Byte], UpscaledVideo]] =
    try {
      val submitted = client.send(
        HttpRequest
          .newBuilder(uri(port, "/sdcpp/v1/upscale"))
          .timeout(java.time.Duration.ofSeconds(120))
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(writeToString(request)))
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      if (submitted.statusCode != 202)
        Left(
          s"the runner refused the job (${submitted.statusCode}): ${submitted.body.trim.take(300)}"
        )
      else {
        val jobId = readFromString[NativeJob](submitted.body).id
        onSubmitted(jobId)
        await(port, jobId, serverExit).flatMap(result =>
          result.b64Json.filter(_.nonEmpty) match {
            case Some(video) =>
              Right(
                Right(
                  UpscaledVideo(
                    Base64.getMimeDecoder.decode(video),
                    result.outputFormat,
                    result.mimeType.getOrElse(s"video/${result.outputFormat}"),
                    result.fps,
                    result.frameCount
                  )
                )
              )
            case None => firstImage(result).map(Left(_))
          }
        )
      }
    } catch {
      case NonFatal(err) =>
        Left(
          serverExit().fold(s"submitting the job failed: ${err.getMessage}")(
            exited
          )
        )
    }

  /** A dead server's reason: its exit code, and the signal when one killed it
    * (the out-of-memory killer's 9, for instance).
    */
  private def exited(code: Int): String =
    if (code > 128)
      s"the server was killed (exit code $code, signal ${code - 128})"
    else s"the server exited with code $code"

  /** Asks the server to cancel a job it accepted. The wait ends on its own once
    * the poll sees the `cancelled` status; a server past the point its build
    * can cancel answers 409 and the job simply finishes, so nothing here is an
    * error worth failing over — whoever cancelled already knows what they asked
    * for.
    */
  def cancel(port: Int, jobId: String): Unit =
    try {
      val response = client.send(
        HttpRequest
          .newBuilder(uri(port, s"/sdcpp/v1/jobs/$jobId/cancel"))
          .timeout(java.time.Duration.ofSeconds(10))
          .POST(HttpRequest.BodyPublishers.noBody())
          .build(),
        HttpResponse.BodyHandlers.ofString()
      )
      if (response.statusCode != 200)
        logger.info(
          s"sd-server answered ${response.statusCode} to cancelling job $jobId"
        )
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Cancelling job $jobId on sd-server failed", err)
    }

  /** Polls the job once a second until it ends. A timed-out poll means a busy
    * server and is tolerated as long as a session's would be
    * (`GenerationManager.SilenceGraceMillis`); a dead server, a vanished or
    * failed job, or five other failures in a row, end the wait with the reason.
    */
  private def await(
      port: Int,
      jobId: String,
      serverExit: () => Option[Int]
  ): Either[String, NativeJobResult] = {
    var lastAnswerAt = System.currentTimeMillis()
    var failures = 0
    var outcome: Option[Either[String, NativeJobResult]] = None
    while (outcome.isEmpty) {
      sleep(1.second)
      serverExit().foreach(code => outcome = Some(Left(exited(code))))
      if (outcome.isEmpty) try {
        val response = get(port, s"/sdcpp/v1/jobs/$jobId", 10)
        response.statusCode match {
          case 200 =>
            val job = readFromString[NativeJob](response.body, readerConfig)
            lastAnswerAt = System.currentTimeMillis()
            failures = 0
            job.status match {
              case "completed" =>
                outcome = Some(
                  job.result.toRight(
                    "sd-server completed the job without a result"
                  )
                )
              case "failed" | "cancelled" =>
                outcome = Some(
                  Left(
                    s"sd-server ${job.status} the job: " +
                      job.error
                        .map(_.message)
                        .filter(_.nonEmpty)
                        .getOrElse("no reason given")
                  )
                )
              case _ => ()
            }
          case 404 | 410 =>
            outcome = Some(Left("sd-server no longer knows the job"))
          case other =>
            failures += 1
            if (failures >= 5)
              outcome = Some(
                Left(s"polling the job kept failing (HTTP $other)")
              )
        }
      } catch {
        case _: HttpTimeoutException =>
          if (
            System.currentTimeMillis() - lastAnswerAt >
              GenerationManager.SilenceGraceMillis
          )
            outcome = Some(
              Left("sd-server answered nothing for too long — giving up")
            )
        case NonFatal(err) =>
          failures += 1
          if (failures >= 5)
            outcome = Some(
              Left(s"polling the job kept failing: ${err.getMessage}")
            )
      }
    }
    outcome.get
  }
}
