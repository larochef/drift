package drift.backend.session

import drift.backend.Background
import drift.backend.cache.ModelCache
import drift.backend.runtime.{LaunchRuntime, RuntimeManager}
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import java.util.concurrent.{ConcurrentHashMap, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean
import scala.concurrent.duration.DurationInt
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger
import ox.{discard, sleep}

/** Turns a ready run configuration into a live `sd-server` process on a known
  * port, and manages its lifecycle (`specs/07-launch-and-supervision.md`).
  *
  * Sessions are runtime state, held in memory and keyed by run configuration —
  * one slot each, so relaunching after a failure replaces the failed session
  * rather than piling up history. On drift restart, sessions are gone.
  *
  * The process's stdout and stderr go to `~/.cache/drift/logs/<sessionId>.log`
  * from the moment it spawns: nothing can deadlock on a full pipe, a crash has
  * a tail to attach, and `specs/13-log-streaming.md` builds its live view on
  * the same file.
  *
  * Beside this file (`specs/29-split-oversized-files.md`): the argv
  * (`LaunchArguments`), the output (`SessionOutput`), ports, probes and
  * termination (`ServerProcesses`), and the servers post-process jobs own
  * (`JobServers`).
  */
final class SessionManager(
    storage: StorageService,
    modelCache: ModelCache,
    runtimeManager: RuntimeManager,
    val logsRoot: Path,
    /** The LoRA root; each launch is given its architecture's folder in it as
      * `--lora-model-dir` (`specs/09-lora-management.md`).
      */
    lorasRoot: Path,
    /** The flat upscaler store passed as `--hires-upscalers-dir` on every
      * launch (`specs/10-generation-time-upscaling.md`).
      */
    upscaleRoot: Path,
    /** Where each session is watched, its output drained and its server reaped.
      */
    background: Background
) {
  private val logger = Logger[SessionManager]

  final private class Entry(
      @volatile var session: Session,
      val process: Option[Process],
      val stopRequested: AtomicBoolean = AtomicBoolean(false)
  )
  private val entries = ConcurrentHashMap[String, Entry]()

  /** Captured output, keyed by session id rather than by configuration: a log
    * outlives the entry it came from, since the view may still be open on a
    * session that has since stopped.
    */
  private val logs = ConcurrentHashMap[String, SessionLog]()

  private val arguments =
    LaunchArguments(storage, modelCache, lorasRoot, upscaleRoot)

  private val jobServers = JobServers(
    storage,
    runtimeManager,
    arguments,
    lorasRoot,
    upscaleRoot,
    allocatePort,
    this,
    background
  )

  /** Progress is attached here rather than stored on the session: it changes
    * several times a second while sampling, and the status socket already
    * pushes the sessions topic whenever it differs
    * (`specs/13-log-streaming.md`) - so it costs no new topic and no new
    * service on the other side.
    */
  def list: List[Session] =
    entries.values.asScala
      .map(_.session)
      .map { session =>
        val log = Option(logs.get(session.id))
        session.copy(
          progress = log.flatMap(_.progress),
          activity = log.flatMap(_.activity),
          batch = log.flatMap(_.batch)
        )
      }
      .toList
      .sortBy(_.startedAt)

  def settings: SessionSettings =
    storage
      .get[SessionSettings]("settings", "sessions")
      .getOrElse(SessionSettings())

  // ------------------------------------------------------------------- launch

  /** One session per run configuration: launching one that is already live
    * answers with the existing session instead of starting a second, whatever
    * runtime the request names — restarting would throw away a warm load.
    * `runtimeId` empty means the default runtime.
    */
  def launch(
      runConfigurationId: String,
      runtimeId: Option[String],
      projectId: Option[String]
  ): Session = synchronized {
    Option(entries.get(runConfigurationId))
      .map(_.session)
      .filter(_.status.isActive) match {
      case Some(alive) => alive
      case None => attemptLaunch(runConfigurationId, runtimeId, projectId)
    }
  }

  /** A launch that cannot proceed still answers with a session — `Failed`, with
    * the reason — so the refusal is visible wherever sessions are shown.
    */
  private def refusal(
      runConfigurationId: String,
      tool: RuntimeTool,
      reason: String
  ): Session = {
    val startedAt = System.currentTimeMillis()
    val session = Session(
      id = s"$runConfigurationId-$startedAt",
      runConfigurationId = runConfigurationId,
      tool = tool,
      runtimeId = None,
      port = None,
      pid = None,
      status = SessionStatus.Failed,
      startedAt = startedAt,
      projectId = None,
      error = Some(reason)
    )
    entries.put(runConfigurationId, Entry(session, None))
    session
  }

  /** The engine a configuration runs on (`specs/43`). */
  def runnerOf(runConfigurationId: String): Either[String, RuntimeEngine] =
    storage
      .get[RunConfiguration]("run-configurations", runConfigurationId)
      .map(_.runner)
      .toRight(s"run configuration '$runConfigurationId' does not exist")

  /** The configuration, its architecture and a one-shot `sd-cli` argv of it
    * (`LaunchArguments.resolve`).
    */
  def resolveArguments(
      runConfigurationId: String,
      launchRuntime: LaunchRuntime
  ): Either[String, (RunConfiguration, Architecture, List[String])] =
    arguments.resolve(runConfigurationId, launchRuntime)

  private def attemptLaunch(
      runConfigurationId: String,
      runtimeId: Option[String],
      projectId: Option[String]
  ): Session = {
    val current = settings
    storage.get[RunConfiguration](
      "run-configurations",
      runConfigurationId
    ) match {
      case None =>
        refusal(
          runConfigurationId,
          RuntimeTool.SdCpp,
          s"run configuration '$runConfigurationId' does not exist"
        )
      case Some(configuration) =>
        val architectures = storage.list[Architecture]("architectures")
        val models = storage.list[Model]("models")
        // The tool decides the cap, the runtime and the readiness probe. A
        // missing architecture is named by the blockers below; it reads as
        // sd-cpp until then.
        val tool = architectures
          .find(_.id == configuration.architectureId)
          .map(_.tool)
          .getOrElse(RuntimeTool.SdCpp)
        val (cap, capName) = tool match {
          case RuntimeTool.SdCpp =>
            (current.maximumConcurrentSessions, "maximumConcurrentSessions")
          case RuntimeTool.LlamaCpp =>
            (
              current.maximumConcurrentAssistantSessions,
              "maximumConcurrentAssistantSessions"
            )
        }
        val running = entries.asScala.values
          .map(_.session)
          .filter(session => session.status.isActive && session.tool == tool)
          .toList
        if (running.size >= cap)
          return refusal(
            runConfigurationId,
            tool,
            s"'${running.head.runConfigurationId}' is already running and these models fill memory — " +
              s"stop it first, or raise $capName in settings/sessions.json"
          )

        val problems =
          CommandLine.blockers(
            configuration,
            architectures,
            models,
            arguments.cacheStatuses(configuration, models)
          )
        if (problems.nonEmpty)
          return refusal(
            runConfigurationId,
            tool,
            "cannot launch: " + problems.map(_.message).mkString("; ")
          )

        val runs = architectures
          .find(_.id == configuration.architectureId)
          .fold(List(configuration.runner))(_.runners)
        runtimeManager
          .resolveForLaunch(tool, configuration.runner, runtimeId)
          .flatMap(launch =>
            Either.cond(
              runs.contains(launch.runtime.engine),
              launch,
              s"'${launch.runtime.label}' is ${launch.runtime.engine.displayName}, which this architecture does not run on; its runners are ${runs.map(_.displayName).mkString(", ")}"
            )
          ) match {
          case Left(reason)         => refusal(runConfigurationId, tool, reason)
          case Right(launchRuntime) =>
            allocatePort(current) match {
              case None =>
                refusal(
                  runConfigurationId,
                  tool,
                  s"no free port between ${current.portRangeStart} and ${current.portRangeEnd}"
                )
              case Some(port) =>
                arguments.argumentsFor(
                  configuration,
                  architectures,
                  models,
                  launchRuntime,
                  Some(port)
                ) match {
                  case Left(problems) =>
                    refusal(
                      runConfigurationId,
                      tool,
                      "cannot launch: " + problems.map(_.message).mkString("; ")
                    )
                  case Right((argv, notes)) =>
                    spawn(
                      configuration,
                      tool,
                      launchRuntime,
                      port,
                      argv,
                      current,
                      notes.map(_.message),
                      projectId
                    )
                }
            }
        }
    }
  }

  /** One session's captured output, for the log view. */
  def logOf(sessionId: String): Option[SessionLog] = Option(logs.get(sessionId))

  /** The architecture and runtime a session runs, while both still resolve —
    * what a generation sent to it needs to know to hand sd-server its reference
    * images (`sdserver/ReferenceImages`).
    */
  def architectureAndRuntimeOf(
      sessionId: String
  ): Option[(Architecture, Runtime)] =
    for {
      session <- list.find(_.id == sessionId)
      configuration <- storage.get[RunConfiguration](
        "run-configurations",
        session.runConfigurationId
      )
      architecture <- storage.get[Architecture](
        "architectures",
        configuration.architectureId
      )
      runtimeId <- session.runtimeId
      runtime <- storage.get[Runtime]("runtimes", runtimeId)
    } yield (architecture, runtime)

  /** The port of a ready assistant session, for the chat proxy
    * (`specs/18-assistant-models-and-sessions.md`); a named refusal for
    * anything else.
    */
  def assistantPort(sessionId: String): Either[String, Int] =
    list.find(_.id == sessionId) match {
      case None => Left(s"session '$sessionId' does not exist")
      case Some(session) if session.tool != RuntimeTool.LlamaCpp =>
        Left(
          s"session '$sessionId' is a ${session.tool.displayName} session, not an assistant"
        )
      case Some(session) if session.status != SessionStatus.Ready =>
        Left(
          s"session '$sessionId' is ${session.status.toString.toLowerCase}" +
            session.error.map(e => s": $e").getOrElse("")
        )
      case Some(session) =>
        session.port.toRight(s"session '$sessionId' has no port")
    }

  /** A free port from the configured range. Ports promised to sessions that are
    * still starting are excluded — `sd-server` has not bound them yet — and so
    * are the job servers'.
    */
  private def allocatePort(current: SessionSettings): Option[Int] =
    ServerProcesses.freePort(
      current,
      entries.asScala.values
        .filter(_.session.status.isActive)
        .flatMap(_.session.port)
        .toSet ++ jobServers.ports
    )

  private def spawn(
      configuration: RunConfiguration,
      tool: RuntimeTool,
      launchRuntime: LaunchRuntime,
      port: Int,
      argv: List[String],
      current: SessionSettings,
      parameterNotes: List[String],
      projectId: Option[String]
  ): Session = {
    val startedAt = System.currentTimeMillis()
    val sessionId = s"${configuration.id}-$startedAt"
    val logFile = logsRoot.resolve(s"$sessionId.log")
    val commandLine =
      CommandLine.render(argv, launchRuntime.executable.toString)
    try {
      Files.createDirectories(logsRoot)
      // sd-server scans --lora-model-dir and --hires-upscalers-dir at
      // startup; they must at least exist. The LoRA directory is the
      // architecture's own folder (`CommandLine.resolve`).
      if (launchRuntime.runtime.tool == RuntimeTool.SdCpp)
        Files.createDirectories(lorasRoot.resolve(configuration.architectureId))
      Files.createDirectories(upscaleRoot)
      // The spawn header goes into the log before the process does: a crash
      // that prints nothing (bugs/17) still leaves the exact environment and
      // command behind, and the log tail attached to a failure carries them.
      val workingDirectory = launchRuntime.executable.getParent
      val environmentLines = launchRuntime.environment.toList.sorted
        .map((name, value) => s"# $name=$value")
      Files.writeString(
        logFile,
        (s"# drift session $sessionId — ${configuration.label} on runtime '${launchRuntime.runtime.id}'"
          :: s"# cwd=$workingDirectory"
          :: environmentLines)
          .mkString("", "\n", s"\n# $commandLine\n")
      )
      val builder =
        ProcessBuilder((launchRuntime.executable.toString :: argv)*)
          // Beside its own libraries: sd-server may resolve resources
          // relative to itself, and the replaced shell scripts ran there.
          .directory(workingDirectory.toFile)
          // Merged, then drained by drift itself rather than redirected to the
          // file: the buffer, the progress and the file all come off the one
          // reader fork (`specs/13-log-streaming.md`).
          .redirectErrorStream(true)
      launchRuntime.environment.foreach { (name, value) =>
        builder.environment().put(name, value)
      }
      val process = builder.start()
      val log = SessionLog()
      logs.put(sessionId, log)
      SessionOutput.capture(sessionId, process, logFile, log, background)
      logger.info(
        s"Session $sessionId: spawned pid ${process.pid()} on port $port: $commandLine"
      )
      val session = Session(
        id = sessionId,
        runConfigurationId = configuration.id,
        tool = tool,
        runtimeId = Some(launchRuntime.runtime.id),
        port = Some(port),
        pid = Some(process.pid()),
        status = SessionStatus.Starting,
        startedAt = startedAt,
        projectId = projectId,
        parameterNotes = parameterNotes
      )
      val entry = Entry(session, Some(process))
      entries.put(configuration.id, entry)
      // The inference page orders configurations by recency of use, and
      // sessions do not survive a drift restart, so the launch is persisted.
      storage.save(
        "run-configurations",
        configuration.id,
        configuration.copy(lastUsedAt = startedAt)
      )
      monitor(entry, process, port, logFile, current, tool)
      session
    } catch {
      case NonFatal(err) =>
        refusal(
          configuration.id,
          tool,
          s"failed to spawn ${tool.executableName}: ${Option(err.getMessage).getOrElse(err.toString)}"
        )
    }
  }

  // -------------------------------------------------------------- supervision

  /** One fork per session: flips `starting` to `ready` when the server answers
    * on its port, and reaps a died process into `failed` with the log tail
    * attached — whichever state it was in — rather than leaving it apparently
    * live. Readiness is an HTTP probe — `/sdcpp/v1/capabilities` for sd-server,
    * `/health` for llama-server — never a parse of the log
    * (`specs/13-log-streaming.md`).
    */
  private def monitor(
      entry: Entry,
      process: Process,
      port: Int,
      logFile: Path,
      current: SessionSettings,
      tool: RuntimeTool
  ): Unit = {
    background.start(s"drift-session-${entry.session.id}")(
      follow(entry, process, port, logFile, current, tool)
    )
  }

  /** The monitor's loop, on its own fork until the process ends or fails to
    * come up.
    */
  private def follow(
      entry: Entry,
      process: Process,
      port: Int,
      logFile: Path,
      current: SessionSettings,
      tool: RuntimeTool
  ): Unit = {
    val deadline = System.currentTimeMillis() +
      current.readinessTimeoutMinutes.toLong * 60 * 1000
    while (true) {
      if (!process.isAlive) {
        if (entry.stopRequested.get())
          entry.session = entry.session.copy(status = SessionStatus.Stopped)
        else {
          val tail = SessionOutput.tail(logFile)
          // The line that reads like the reason goes first: on a failed
          // load it is the one thing worth seeing without scrolling, and
          // the tail underneath still carries the context.
          val reason = Option(logs.get(entry.session.id))
            .flatMap(_.errorLine)
            .map(line => s"\n$line")
            .getOrElse("")
          entry.session = entry.session.copy(
            status = SessionStatus.Failed,
            error = Some(
              s"${tool.executableName} exited with code ${process.exitValue()}" +
                reason + (if (tail.isEmpty) "" else s"\n$tail")
            )
          )
          logger.warn(
            s"Session ${entry.session.id}: exited with code ${process.exitValue()}"
          )
        }
        return
      }
      if (entry.session.status == SessionStatus.Starting) {
        if (ServerProcesses.answersProbe(port, tool)) {
          entry.session = entry.session.copy(status = SessionStatus.Ready)
          logger.info(s"Session ${entry.session.id}: ready on :$port")
        } else if (System.currentTimeMillis() > deadline) {
          val tail = SessionOutput.tail(logFile)
          entry.session = entry.session.copy(
            status = SessionStatus.Failed,
            error = Some(
              s"not ready after ${current.readinessTimeoutMinutes} minutes; giving up" +
                (if (tail.isEmpty) "" else s"\n$tail")
            )
          )
          ServerProcesses.terminate(process)
          return
        }
      }
      sleep(1.second)
    }
  }

  // -------------------------------------------------------------- job servers

  /** An sd-server a post-process job owns (`JobServers.start`). */
  def startJobServer(
      runConfigurationId: String,
      runtimeId: Option[String],
      logFile: Path,
      onLine: String => Unit = _ => ()
  ): Either[String, JobServer] =
    jobServers.start(runConfigurationId, runtimeId, logFile, settings, onLine)

  // --------------------------------------------------------------------- stop

  /** SIGTERM now, SIGKILL after a grace period. The session is marked `stopped`
    * immediately — the process is doomed either way — and the monitor knows not
    * to reinterpret the exit as a crash.
    */
  def stop(sessionId: String): Option[Session] =
    entries.asScala.values.find(_.session.id == sessionId).map { entry =>
      entry.process.filter(_.isAlive) match {
        case None          => entry.session
        case Some(process) =>
          entry.stopRequested.set(true)
          entry.session = entry.session.copy(status = SessionStatus.Stopped)
          logger.info(s"Session $sessionId: stopping pid ${process.pid()}")
          background.start(s"drift-session-reaper-$sessionId")(
            ServerProcesses.terminate(process)
          )
          entry.session
      }
    }

  /** Kills the session's server and launches the same configuration again on
    * the same runtime: what stopping a generation costs on a build that cannot
    * interrupt one (`specs/08-inference-ui.md`), and the generation form's
    * **Restart** — a LoRA installed since the launch is only listed by a new
    * server. The relaunch waits for the old process to be gone, so the port it
    * holds is free by the time the new one binds; it runs on a fork of its own,
    * since a load takes minutes and the answer is due now. Answers with the
    * session as it stands, stopped.
    */
  def restart(sessionId: String): Option[Session] =
    entries.asScala.values.find(_.session.id == sessionId).map { entry =>
      val configurationId = entry.session.runConfigurationId
      val runtimeId = entry.session.runtimeId
      val projectId = entry.session.projectId
      val dying = entry.process
      val stopped = stop(sessionId).getOrElse(entry.session)
      background.start(s"drift-session-restart-$sessionId") {
        dying.foreach(
          _.waitFor(SessionManager.GraceSeconds + 5, TimeUnit.SECONDS)
        )
        logger.info(
          s"Session $sessionId: restarting '$configurationId'"
        )
        launch(configurationId, runtimeId, projectId).discard
      }
      stopped
    }

  /** Kills every child, synchronously — killing drift must not orphan an
    * `sd-server` holding 30GB of VRAM. Wired into the shutdown hook in
    * `Main.scala`.
    */
  def stopAll(): Unit = {
    val alive = entries.asScala.values.toList.flatMap { entry =>
      entry.stopRequested.set(true)
      if (entry.session.status.isActive)
        entry.session = entry.session.copy(status = SessionStatus.Stopped)
      entry.process.filter(_.isAlive)
    } ++ jobServers.drain()
    alive.foreach(_.destroy())
    val deadline =
      System.nanoTime() + TimeUnit.SECONDS.toNanos(SessionManager.GraceSeconds)
    alive.foreach { process =>
      val remaining = deadline - System.nanoTime()
      if (remaining <= 0 || !process.waitFor(remaining, TimeUnit.NANOSECONDS)) {
        process.destroyForcibly()
        process.waitFor(5, TimeUnit.SECONDS)
      }
    }
  }
}

object SessionManager {

  /** How long SIGTERM gets before SIGKILL. */
  val GraceSeconds: Long = 10
}
