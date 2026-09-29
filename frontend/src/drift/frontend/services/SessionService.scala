package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object SessionService {
  enum Command {
    case Load

    /** `runtimeId` empty launches on the default runtime; `projectId` is the
      * workspace launching it, none from the Sandbox (`specs/47`).
      */
    case Launch(
        runConfigurationId: String,
        runtimeId: Option[String],
        projectId: Option[String]
    )
    case Stop(sessionId: String)

    /** Stop and launch again on the same runtime (`restartSession`). */
    case Restart(sessionId: String)
  }
}

/** The live `sd-server` sessions (`specs/07-launch-and-supervision.md`).
  *
  * The backend holds one session slot per run configuration, so the state is a
  * map keyed the way the cards look it up. Launch and stop answer with the
  * session's immediate state; the status socket pushes the transitions the
  * backend makes on its own — `starting` becoming `ready`, or a crash becoming
  * `failed`.
  */
class SessionService(statusSocket: StatusSocketService) extends ServiceErrors {
  import SessionService.Command

  private val listFn = ApiClient.stream(drift.shared.listSessions)
  private val launchFn = ApiClient.stream(drift.shared.launchSession)
  private val stopFn = ApiClient.stream(drift.shared.stopSession)
  private val restartFn = ApiClient.stream(drift.shared.restartSession)

  private val _sessions = Var(Map.empty[String, Session])

  /** Keyed by run configuration id. */
  val sessions: Signal[Map[String, Session]] = _sessions.signal

  private val cmdBus = new EventBus[Command]

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def applyOne(session: Session): Unit =
    _sessions.update(_ + (session.runConfigurationId -> session))

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[Session]]] {
        case Success(listing) =>
          clearError()
          _sessions.set(
            listing.map(session => session.runConfigurationId -> session).toMap
          )
        case Failure(err) => reportFailure("Listing sessions", err)
      },
    cmdBus.events
      .collect {
        case Command.Launch(runConfigurationId, runtimeId, projectId) =>
          (runConfigurationId, runtimeId, projectId)
      }
      .flatMapMerge((id, runtimeId, projectId) =>
        launchFn(LaunchSessionRequest(id, runtimeId, projectId)).recoverToTry
      ) --> Observer[Try[Session]] {
      case Success(session) =>
        clearError()
        applyOne(session)
      case Failure(err) => reportFailure("Launching the session", err)
    },
    cmdBus.events
      .collect { case Command.Stop(sessionId) => sessionId }
      .flatMapMerge(id =>
        stopFn(id).map(stopped => (id, stopped)).recoverToTry
      ) --> Observer[Try[(String, Option[Session])]] {
      case Success((_, Some(session))) =>
        clearError()
        applyOne(session)
      case Success((id, None)) =>
        reportFailure("Stopping the session", s"'$id' no longer exists.")
      case Failure(err) => reportFailure("Stopping the session", err)
    },
    cmdBus.events
      .collect { case Command.Restart(sessionId) => sessionId }
      .flatMapMerge(id =>
        restartFn(id).map(stopped => (id, stopped)).recoverToTry
      ) --> Observer[Try[(String, Option[Session])]] {
      case Success((_, Some(session))) =>
        clearError()
        applyOne(session)
      case Success((id, None)) =>
        reportFailure("Restarting the session", s"'$id' no longer exists.")
      case Failure(err) => reportFailure("Restarting the session", err)
    },
    // The socket pushes the whole session list whenever it changes, so
    // `starting` flips to `ready` (or a crash to `failed`) without the user
    // touching anything.
    statusSocket.sessions --> Observer[List[Session]](listing =>
      _sessions.set(
        listing.map(session => session.runConfigurationId -> session).toMap
      )
    )
  )
}
