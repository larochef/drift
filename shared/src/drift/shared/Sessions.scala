package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** The lifecycle of one `sd-server` process
  * (`specs/07-launch-and-supervision.md`). `Starting` becomes `Ready` when the
  * server answers HTTP on its port — not when the process spawns, because the
  * expensive, failure-prone step is loading the weights, which takes minutes.
  */
enum SessionStatus derives CanEqual {
  case Starting, Ready, Failed, Stopped

  def isActive: Boolean = this match {
    case Starting | Ready => true
    case _                => false
  }
}
object SessionStatus {
  // String-encoded like the other singleton enums; the explicit schema keeps
  // tapir's derivation in agreement (see RuntimeBackend).
  given Schema[SessionStatus] =
    Schema.derivedEnumeration[SessionStatus].defaultStringBased
}

/** One live (or recently dead) `sd-server` process started from a run
  * configuration. Runtime state, not persisted config: on drift restart,
  * sessions are gone.
  *
  * A launch that is *refused* — blockers, no usable runtime, no free port —
  * also answers with a `Session`, already `Failed` with the reason in `error`
  * and no port or pid, mirroring how `startDownload` reports a model that
  * cannot be fetched.
  */
case class Session(
    id: String,
    runConfigurationId: String,
    /** The configuration's architecture's tool — what process this is, so the
      * UI can tell a generation session from an assistant one without a lookup.
      */
    tool: RuntimeTool,
    /** The runtime the launch resolved; empty when the launch was refused
      * before one was chosen.
      */
    runtimeId: Option[String],
    port: Option[Int],
    pid: Option[Long],
    status: SessionStatus,
    startedAt: Long,
    error: Option[String] = None,
    /** Where the session's current phase has got to, read out of its log
      * (`specs/13-log-streaming.md`): loading tensors while it starts, sampling
      * steps while it generates. `None` between phases and whenever the shape
      * of the log stops matching - an honest absence beats a wrong number.
      */
    progress: Option[SessionProgress] = None,
    /** The last thing the session said that was not a progress redraw - what it
      * is working on when there is no bar to show (François, 2026-09-09).
      * Trimmed to something a single line can hold.
      */
    activity: Option[String] = None,
    /** What resolving the parameters had to say — a flag this build cannot take
      * and was dropped, or one it added of its own accord
      * (`specs/16-parameter-resolution.md`). Rendered messages: the card only
      * displays them.
      */
    parameterNotes: List[String] = List.empty
)
object Session {
  // No discriminator: `SessionStatus` has only singleton cases, so it encodes
  // as a plain string ("Starting"). The config must be inline -- the macro
  // needs a constant expression.
  given JsonValueCodec[Session] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[Session]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[Session]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** `runtimeId` is the launch-time runtime choice; empty means the default
  * runtime. The chosen runtime lives on the session, not the configuration —
  * picking one at launch pins nothing for next time.
  */
case class LaunchSessionRequest(
    runConfigurationId: String,
    runtimeId: Option[String] = None
)
object LaunchSessionRequest {
  given JsonValueCodec[LaunchSessionRequest] = JsonCodecMaker.make
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val sessionsBase = endpoint.in("api")

val listSessions: PublicEndpoint[Unit, Unit, List[Session], Any] =
  sessionsBase.get.in("sessions").out(jsonBody[List[Session]])

/** Launches `sd-server` for a run configuration. A configuration that already
  * has a live session answers with that session instead of starting a second; a
  * launch that cannot proceed answers with a `Failed` session naming why.
  */
val launchSession: PublicEndpoint[LaunchSessionRequest, Unit, Session, Any] =
  sessionsBase.post
    .in("sessions")
    .in(jsonBody[LaunchSessionRequest])
    .out(jsonBody[Session])

/** Stops the session's process — SIGTERM, then SIGKILL after a grace period.
  * Answers with the stopped session, or nothing for an unknown id.
  */
/** Kills the session's server and launches its configuration again on the same
  * runtime, once the old process is gone: a LoRA installed since the launch is
  * listed, a server gone wrong starts afresh. Answers with the session as it
  * stands, stopped; the relaunch arrives on the status socket.
  */
val restartSession: PublicEndpoint[String, Unit, Option[Session], Any] =
  sessionsBase.post
    .in("sessions" / path[String] / "restart")
    .out(jsonBody[Option[Session]])

val stopSession: PublicEndpoint[String, Unit, Option[Session], Any] =
  sessionsBase.delete
    .in("sessions" / path[String])
    .out(jsonBody[Option[Session]])
