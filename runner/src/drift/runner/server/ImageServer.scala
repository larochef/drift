package drift.runner.server

import drift.protocol.*
import drift.runner.diffusion.*
import drift.runner.server.JobAnswers.{failed, Answer}

import scala.util.control.NonFatal

import ox.*
import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.NettySyncServer

/** `sd-server`'s native API (`/sdcpp/v1`) on the runner's image or video
  * pipeline, as drift drives it (`specs/42`, steps 12 and 14): the capabilities
  * document (the form's defaults, from the launch flags), `img_gen` or
  * `vid_gen` jobs queued and run one at a time, polled, and cancelled — while
  * queued, or between two steps once generating. A request starts from the
  * launch flags' values and overrides what it sends. Progress goes to the log
  * as sd-cpp prints it (`| i/n - Xs/it`) and onto the job, which says what it
  * is doing each time it is polled (`ProgressReport`); an image or video job
  * whose request asks for it (`preview`, sd-server's field) also carries the
  * last step's estimate of its picture, or of its video as a picture that moves
  * (`specs/54-step-preview.md`), and any job what it went on without
  * (`warnings`: a LoRA left unapplied). Images come as base64 or data URLs;
  * references are stretched to the output's size first unless
  * `auto_resize_ref_image` is false, as sd-server does. A generated video comes
  * back as one webm (`VideoFiles`); an upscaled one is written to the file its
  * request names (`VideoUpscale`).
  */
final class ImageServer(
    options: ImageOptions,
    pipeline: Pipeline.Opened,
    ops: drift.runner.ops.Ops
) {

  private val mode = pipeline.mode

  private val family = pipeline.family

  private val queue = new JobQueue

  private val launch = new LaunchDefaults(options, pipeline)

  private val capabilities = new ServerCapabilities(options, pipeline, launch)

  private val results = new JobResults(queue, ops)

  private val imageJobs = new ImageJobs(options, launch, queue, results)

  private val videoJobs = new VideoJobs(options, launch, queue, results)

  private val upscaleJobs = new UpscaleJobs(options, queue, results)

  /** Whatever a request throws is the answer's reason, not a silent 500. */
  private def answered(answer: => Answer): Answer =
    try answer
    catch {
      case NonFatal(error) => failed(500, String.valueOf(error.getMessage))
    }

  private val endpoints: List[ServerEndpoint[Any, Identity]] = List(
    JobEndpoints.capabilities
      .serverLogicSuccess[Identity](_ => ujson.write(capabilities.document)),
    JobEndpoints.submitImage.serverLogic[Identity](asked =>
      answered(pipeline match {
        case image: ImagePipeline => imageJobs.submit(image, asked)
        case _                    => wrongMode
      })
    ),
    JobEndpoints.submitVideo.serverLogic[Identity](asked =>
      answered(pipeline match {
        case video: VideoPipeline => videoJobs.submitVideo(video, asked)
        case _                    => wrongMode
      })
    ),
    JobEndpoints.submitUpscale.serverLogic[Identity](asked =>
      answered(pipeline match {
        case upscaler: SeedVr2Pipeline =>
          upscaleJobs.submitUpscale(upscaler, asked)
        case _ => wrongMode
      })
    ),
    JobEndpoints.job.serverLogic[Identity](id => answered(queue.polled(id))),
    JobEndpoints.cancel.serverLogic[Identity](id =>
      answered(queue.cancelled(id))
    )
  )

  /** Serves until the process ends: the jobs' worker and the server are forks
    * of one scope, so neither outlives the other.
    */
  def serve(): Unit = supervised {
    forkUser(queue.work()).discard
    NettySyncServer()
      .addEndpoints(endpoints)
      .host(options.host)
      .port(options.port)
      .start()
      .discard
    println(s"listening on http://${options.host}:${options.port}")
    never
  }

  private def wrongMode: Answer =
    failed(400, s"$family generates with $mode only")
}
