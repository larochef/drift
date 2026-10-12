package drift.backend.nativejobs

import drift.backend.nativejobs.ServerRequests.given
import drift.protocol.*
import drift.protocol.Requests.given
import drift.shared.*

import java.util.Base64

import com.github.plokhotnyuk.jsoniter_scala.core.*

/** A video the runner's `upscale` job wrote, in the file its request named.
  * `stopped` is why it holds fewer frames than its source, when the job was
  * cancelled or failed after its first frames: the file is the video of those.
  */
private[backend] case class UpscaledVideo(
    format: String,
    mimeType: String,
    fps: Option[Int],
    frameCount: Option[Int],
    sourceFrames: Option[Int],
    stopped: Option[String]
)

/** Images straight from an sd-server, for jobs that drive a server themselves
  * rather than through a session's generation list — the tiles of PiD and
  * redraw (`specs/26-tiled-pid.md`, `specs/27-redraw.md`).
  */
object NativeJobs {

  /** The server's img_gen defaults — its launch flags — and its img_gen
    * features (`ref_images`, …), as its capabilities report them.
    */
  def imageCapabilities(
      port: Int
  ): Either[String, (GenerationDefaults, Map[String, Boolean])] =
    NativeJobClient
      .capabilities(port)
      .flatMap(capabilities =>
        capabilities.defaultsByMode
          .get("img_gen")
          .toRight("the server reports no img_gen defaults")
          .map(_ -> capabilities.featuresByMode.getOrElse("img_gen", Map.empty))
      )

  /** Submits `parameters` as an img_gen job and waits for its first image.
    * `serverExit` is the server's exit code once it has died, if drift runs it:
    * the wait then ends at once, naming it. `onSubmitted` is handed the native
    * job's id the moment the server accepts it, which is what a cancel needs to
    * reach it (`cancel`); `onProgress` what the job says it is doing, each time
    * it is polled, from the servers whose jobs say (`JobProgress`).
    */
  def image(
      port: Int,
      parameters: ImageGenerationParameters,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit,
      onProgress: JobProgress => Unit
  ): Either[String, Array[Byte]] =
    submitted(
      port,
      JobRoutes.ImageJobs,
      writeToString(parameters),
      serverExit,
      onSubmitted
    )
      .flatMap(jobId => await(port, jobId, serverExit, onProgress))
      .flatMap(firstImage)

  private def firstImage(result: NativeJobResult): Either[String, Array[Byte]] =
    result.images.headOption
      .filter(_.b64Json.nonEmpty)
      .map(image => Base64.getMimeDecoder.decode(image.b64Json))
      .toRight("the server completed the job without an image")

  /** Submits the runner's `upscale` job on a picture and waits for it: a PNG.
    * `serverExit`, `onSubmitted` and `onProgress` as `image` has them.
    */
  def upscale(
      port: Int,
      request: UpscaleJobRequest,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit,
      onProgress: JobProgress => Unit
  ): Either[String, Array[Byte]] =
    submitted(
      port,
      JobRoutes.UpscaleJobs,
      writeToString(request),
      serverExit,
      onSubmitted
    )
      .flatMap(jobId => await(port, jobId, serverExit, onProgress))
      .flatMap(firstImage)

  /** Submits the runner's `upscale` job on a video and waits for it: the video
    * is in the request's `outputFile`. A job cancelled or failed after its
    * first frames is a result too, the shorter video those make, and says so
    * (`UpscaledVideo.stopped`).
    */
  def upscaleVideo(
      port: Int,
      request: UpscaleJobRequest,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit,
      onProgress: JobProgress => Unit
  ): Either[String, UpscaledVideo] =
    submitted(
      port,
      JobRoutes.UpscaleJobs,
      writeToString(request),
      serverExit,
      onSubmitted
    )
      .flatMap(jobId => awaitEnd(port, jobId, serverExit, onProgress))
      .flatMap { job =>
        val reason = job.status match {
          case "completed" => None
          case "cancelled" => Some("stopped")
          case _           =>
            Some(
              job.error.map(_.message).filter(_.nonEmpty).getOrElse("failed")
            )
        }
        job.result.filter(_.file.exists(_.nonEmpty)) match {
          case Some(result) =>
            Right(
              UpscaledVideo(
                result.outputFormat,
                result.mimeType.getOrElse(s"video/${result.outputFormat}"),
                result.fps,
                result.frameCount,
                result.sourceFrames,
                reason
              )
            )
          case None =>
            Left(
              reason.fold("the runner completed the job without a video")(why =>
                s"the runner ${job.status} the job: $why"
              )
            )
        }
      }

  /** Posts a job; its id once the server has taken it. A server that died
    * meanwhile is the reason, rather than the refused connection.
    */
  private def submitted(
      port: Int,
      path: String,
      body: String,
      serverExit: () => Option[Int],
      onSubmitted: String => Unit
  ): Either[String, String] =
    NativeJobClient.submit(port, path, body) match {
      case Right(job) =>
        onSubmitted(job.id)
        Right(job.id)
      case Left(reason) =>
        Left(serverExit().fold(reason)(NativeJobClient.exited))
    }

  /** Asks the server to cancel a job it accepted (`NativeJobClient.cancel`). */
  def cancel(port: Int, jobId: String): Unit =
    NativeJobClient.cancel(port, jobId)

  /** A completed job's result; a failed or cancelled one's reason. */
  private def await(
      port: Int,
      jobId: String,
      serverExit: () => Option[Int],
      onProgress: JobProgress => Unit
  ): Either[String, NativeJobResult] =
    awaitEnd(port, jobId, serverExit, onProgress).flatMap(job =>
      job.status match {
        case "completed" =>
          job.result.toRight("the server completed the job without a result")
        case _ =>
          Left(
            s"the server ${job.status} the job: " +
              job.error
                .map(_.message)
                .filter(_.nonEmpty)
                .getOrElse("no reason given")
          )
      }
    )

  /** The job as it ended — completed, failed or cancelled — once it has
    * (`NativeJobClient.follow`).
    */
  private def awaitEnd(
      port: Int,
      jobId: String,
      serverExit: () => Option[Int],
      onProgress: JobProgress => Unit
  ): Either[String, NativeJob] =
    NativeJobClient
      .follow(port, jobId, label = s"Job $jobId", serverExit = serverExit)(
        _.progress.foreach(onProgress)
      )
      .flatMap(_.toRight("the job was let go before it ended"))
}
