package drift.backend.postprocess

import drift.backend.runtime.LaunchRuntime
import drift.backend.session.SessionManager
import drift.shared.*

/** The server a job's requests go to, and how to let go of it: its port, its
  * exit code once it has died — which a request waiting on it names — and
  * `stop`, which does nothing to a session's server, not the job's to stop.
  */
final private[postprocess] case class ChosenServer(
    port: Int,
    exitCode: () => Option[Int],
    stop: () => Unit
)

/** Where every kind of job gets its server. */
private[postprocess] object ChosenServer {

  /** The ready session of the configuration on that runtime, if there is one: a
    * model already loaded is not loaded a second time beside itself. On the
    * job's runtime only: that is the build the job was checked against.
    */
  def readySession(
      sessionManager: SessionManager,
      configurationId: String,
      runtime: LaunchRuntime
  ): Option[Session] =
    sessionManager.list.find(session =>
      session.runConfigurationId == configurationId &&
        session.runtimeId.contains(runtime.runtime.id) &&
        session.status == SessionStatus.Ready && session.port.isDefined
    )

  /** A server of the job's own, its output read into the job's log. With
    * `stoppedByCancel` a cancel stops it at once: what waits on it then ends on
    * a dead server, which the job records as the cancel it is.
    */
  def own(
      jobs: PostProcessJobs,
      sessionManager: SessionManager,
      job: PostProcessJob,
      configurationId: String,
      runtimeId: String,
      stoppedByCancel: Boolean
  ): Either[String, ChosenServer] =
    sessionManager
      .startJobServer(
        configurationId,
        Some(runtimeId),
        jobs.files.logFileOf(job),
        line => jobs.noteLine(job, line)
      )
      .map { server =>
        if (stoppedByCancel) jobs.runsOn(job, server.stop)
        ChosenServer(server.port, () => server.exitCode, server.stop)
      }

  /** `session`'s server when there is one — watched by its session, so it has
    * no exit code to name — else one of the job's own.
    */
  def of(
      jobs: PostProcessJobs,
      sessionManager: SessionManager,
      job: PostProcessJob,
      configurationId: String,
      runtime: LaunchRuntime,
      session: Option[Session]
  ): Either[String, ChosenServer] =
    session
      .flatMap(_.port)
      .map(port => Right(ChosenServer(port, () => None, () => ())))
      .getOrElse(
        own(
          jobs,
          sessionManager,
          job,
          configurationId,
          runtime.runtime.id,
          stoppedByCancel = true
        )
      )
}
