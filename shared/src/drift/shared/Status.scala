package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}

/** One message on the status WebSocket — the push replacement for the frontend
  * status polls. Commands and CRUD stay REST; this socket only mirrors state
  * the backend changes on its own.
  *
  * Every present field is a full snapshot of its topic; an absent field means
  * that topic did not change since the connection's previous message. The first
  * message after (re)connect carries every topic, empty or not, so the socket
  * alone re-syncs a client — including one that missed a backend restart.
  *
  * Deliberately not a tapir endpoint: the server's WebSocket capability
  * (`OxStreams`) is JVM-only, and the browser opens a plain `dom.WebSocket`.
  * Both sides use [[statusSocketPath]] and this codec, which is the whole
  * contract.
  */
case class StatusUpdate(
    sessions: Option[List[Session]] = None,
    downloads: Option[List[DownloadJob]] = None,
    loraDownloads: Option[List[LoraDownloadJob]] = None,
    upscalerDownloads: Option[List[UpscalerDownloadJob]] = None,
    runtimeInstalls: Option[List[RuntimeInstallJob]] = None,
    /** Keyed by session id; only sessions whose generation list changed (all of
      * them on the connection's first message).
      */
    generations: Option[Map[String, List[Generation]]] = None,
    postProcessJobs: Option[List[PostProcessJob]] = None,
    conversions: Option[List[ConversionJob]] = None,
    /** Absent too where the machine's memory cannot be read. */
    machine: Option[MachineStatus] = None
)
object StatusUpdate {
  // No discriminator, so the singleton enums inside (job states, session and
  // generation statuses) encode as plain strings. The config must be inline --
  // the macro needs a constant expression.
  given JsonValueCodec[StatusUpdate] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** The backend serves the socket at `GET /api/status` (with the WebSocket
  * upgrade); the tapir definition spells the segments out, so keep the two in
  * sync.
  */
val statusSocketPath = "/api/status"
