package drift.backend.routes

import drift.backend.conversion.ConversionManager
import drift.backend.download.DownloadManager
import drift.backend.lora.LoraManager
import drift.backend.postprocess.PostProcessManager
import drift.backend.runtime.RuntimeManager
import drift.backend.sdserver.GenerationManager
import drift.backend.session.SessionManager
import drift.backend.upscale.UpscalerManager
import drift.shared.*

import scala.concurrent.duration.*

import com.github.plokhotnyuk.jsoniter_scala.core.writeToString
import ox.flow.Flow
import sttp.capabilities.WebSockets
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.OxStreams

/** Everything the status socket mirrors, sampled together so one tick compares
  * like with like. The manager list methods all sort, so equality means "really
  * unchanged" rather than "same elements, shuffled".
  */
private case class StatusSnapshot(
    sessions: List[Session],
    downloads: List[DownloadJob],
    loraDownloads: List[LoraDownloadJob],
    upscalerDownloads: List[UpscalerDownloadJob],
    runtimeInstalls: List[RuntimeInstallJob],
    generations: Map[String, List[Generation]],
    postProcessJobs: List[PostProcessJob],
    conversions: List[ConversionJob]
)

/** The status WebSocket at `GET /api/status` (`statusSocketPath` in the shared
  * module) — see [[drift.shared.StatusUpdate]] for the protocol.
  *
  * The managers keep their own state fresh on background threads (download
  * workers, the per-generation monitors, the session supervisor), so each
  * connection simply samples them twice a second and pushes the topics that
  * differ from what it last sent. Between changes nothing crosses the wire —
  * the sampling loop is a handful of in-memory list copies.
  */
def statusEndpoint(
    sessionManager: SessionManager,
    downloadManager: DownloadManager,
    loraManager: LoraManager,
    upscalerManager: UpscalerManager,
    runtimeManager: RuntimeManager,
    generationManager: GenerationManager,
    postProcessManager: PostProcessManager,
    conversionManager: ConversionManager
): ServerEndpoint[OxStreams & WebSockets, Identity] = {

  def sample(): StatusSnapshot = StatusSnapshot(
    sessions = sessionManager.list,
    downloads = downloadManager.list,
    loraDownloads = loraManager.listJobs,
    upscalerDownloads = upscalerManager.listJobs,
    runtimeInstalls = runtimeManager.listInstalls,
    generations = generationManager.listBySession,
    postProcessJobs = postProcessManager.listJobs,
    conversions = conversionManager.listJobs
  )

  /** Every topic, unconditionally — the connection's first message. A client
    * that reconnected after a backend restart must also learn what is *gone*,
    * so empty topics are sent like any other.
    */
  def everything(current: StatusSnapshot): StatusUpdate = StatusUpdate(
    sessions = Some(current.sessions),
    downloads = Some(current.downloads),
    loraDownloads = Some(current.loraDownloads),
    upscalerDownloads = Some(current.upscalerDownloads),
    runtimeInstalls = Some(current.runtimeInstalls),
    generations = Some(current.generations),
    postProcessJobs = Some(current.postProcessJobs),
    conversions = Some(current.conversions)
  )

  /** Only what changed, or nothing. Generations diff per session, so one
    * session generating does not resend every other session's history.
    */
  def changesOnly(
      previous: StatusSnapshot,
      current: StatusSnapshot
  ): Option[StatusUpdate] = {
    def changed[A](topic: StatusSnapshot => A): Option[A] =
      Some(topic(current)).filter(_ != topic(previous))
    val update = StatusUpdate(
      sessions = changed(_.sessions),
      downloads = changed(_.downloads),
      loraDownloads = changed(_.loraDownloads),
      upscalerDownloads = changed(_.upscalerDownloads),
      runtimeInstalls = changed(_.runtimeInstalls),
      generations = changed(_.generations).map(
        _.filter((sessionId, list) =>
          !previous.generations.get(sessionId).contains(list)
        )
      ),
      postProcessJobs = changed(_.postProcessJobs),
      conversions = changed(_.conversions)
    )
    Some(update).filter(_ != StatusUpdate())
  }

  val pipe: OxStreams.Pipe[String, String] = incoming => {
    val updates = Flow
      .tick(500.millis)
      .mapStateful(Option.empty[StatusSnapshot]) { (previous, _) =>
        val current = sample()
        val update = previous match {
          case None       => Some(everything(current))
          case Some(prev) => changesOnly(prev, current)
        }
        (Some(current), update)
      }
      .collect { case Some(update) => writeToString(update) }
    // The outgoing flow must complete when the client closes (the server logs
    // an error otherwise), and the tick flow alone never would — merging with
    // the drained input ties the two lifetimes together.
    incoming.drain().merge(updates, propagateDoneLeft = true)
  }

  endpoint.get
    .in("api" / "status")
    .out(
      webSocketBody[
        String,
        CodecFormat.TextPlain,
        String,
        CodecFormat.TextPlain
      ](
        OxStreams
      )
    )
    .serverLogicSuccess[Identity](_ => pipe)
}
