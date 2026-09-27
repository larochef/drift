package drift.backend.session

import drift.backend.process.ProcessOutput
import drift.backend.runtime.RuntimeManager
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import java.util.concurrent.ConcurrentHashMap
import scala.jdk.CollectionConverters.*

import com.typesafe.scalalogging.Logger

/** A job server that serves: its port, the way to stop it, and its exit code
  * once it has died — what a job waiting on it names when it fails.
  */
final class JobServer(val port: Int, process: Process, val stop: () => Unit) {
  def exitCode: Option[Int] = Option.when(!process.isAlive)(process.exitValue())
}

/** The sd-servers post-process jobs own (`specs/27-redraw.md`): never sessions
  * — not listed, not counted against `maximumConcurrentSessions` — but their
  * ports are taken and `SessionManager.stopAll` kills them.
  */
final private[session] class JobServers(
    storage: StorageService,
    runtimeManager: RuntimeManager,
    arguments: LaunchArguments,
    lorasRoot: Path,
    upscaleRoot: Path,
    /** A free port, excluding every promised one — sessions' and job servers'.
      */
    freePort: SessionSettings => Option[Int],
    /** The session manager's lock, so a launch and a job server never take the
      * same port.
      */
    lock: AnyRef
) {
  private val logger = Logger[JobServers]
  private val servers = ConcurrentHashMap[Int, Process]()

  def ports: Set[Int] = servers.keySet.asScala.toSet

  /** Forgets every job server, answering with the processes still alive for the
    * caller to kill.
    */
  def drain(): List[Process] = {
    val alive = servers.values.asScala.filter(_.isAlive).toList
    servers.clear()
    alive
  }

  /** The argv a session of the configuration would launch with, on a free port,
    * its output appended to `logFile` and handed line by line to `onLine` — the
    * job's way to show what loading and sampling are doing, the way a session's
    * panel does (`specs/13-log-streaming.md`). Blocks until the server serves —
    * or with why it could not start.
    */
  def start(
      runConfigurationId: String,
      runtimeId: Option[String],
      logFile: Path,
      current: SessionSettings,
      onLine: String => Unit = _ => ()
  ): Either[String, JobServer] = {
    val started = lock.synchronized {
      for {
        configuration <- storage
          .get[RunConfiguration]("run-configurations", runConfigurationId)
          .toRight(s"run configuration '$runConfigurationId' does not exist")
        launchRuntime <- runtimeManager
          .resolveForLaunch(RuntimeTool.SdCpp, configuration.runner, runtimeId)
        port <- freePort(current).toRight(
          s"no free port between ${current.portRangeStart} and ${current.portRangeEnd}"
        )
        (argv, _) <- arguments
          .argumentsFor(
            configuration,
            storage.list[Architecture]("architectures"),
            storage.list[Model]("models"),
            launchRuntime,
            Some(port)
          )
          .left
          .map(problems =>
            "cannot run: " + problems.map(_.message).mkString("; ")
          )
      } yield {
        Files.createDirectories(lorasRoot.resolve(configuration.architectureId))
        Files.createDirectories(upscaleRoot)
        Files.writeString(
          logFile,
          s"# job server on :$port: ${CommandLine.render(argv, launchRuntime.executable.toString)}\n",
          StandardOpenOption.CREATE,
          StandardOpenOption.APPEND
        )
        val builder =
          ProcessBuilder((launchRuntime.executable.toString :: argv)*)
            .directory(launchRuntime.executable.getParent.toFile)
            .redirectErrorStream(true)
        launchRuntime.environment.foreach((name, value) =>
          builder.environment().put(name, value)
        )
        val process = builder.start()
        // Drained here rather than redirected by the OS: the file is written
        // just the same, and the lines pass by on their way to it. A pipe
        // nobody reads would block the server mid-load.
        ProcessOutput.capture(s"job-server-$port", process, logFile)(onLine)(
          () => ()
        )
        servers.put(port, process)
        logger.info(
          s"Job server for '$runConfigurationId': pid ${process.pid()} on port $port"
        )
        (port, process)
      }
    }
    started.flatMap { (port, process) =>
      val stop = () => {
        servers.remove(port)
        ServerProcesses.terminate(process)
      }
      val deadline = System.currentTimeMillis() +
        current.readinessTimeoutMinutes.toLong * 60 * 1000
      while (
        process.isAlive &&
        !ServerProcesses.answersProbe(port, RuntimeTool.SdCpp) &&
        System.currentTimeMillis() < deadline
      ) Thread.sleep(1000)
      if (
        process.isAlive && ServerProcesses.answersProbe(port, RuntimeTool.SdCpp)
      )
        Right(JobServer(port, process, stop))
      else {
        val reason =
          if (process.isAlive)
            s"sd-server was not ready after ${current.readinessTimeoutMinutes} minutes"
          else
            s"sd-server exited with code ${process.exitValue()} while loading"
        stop()
        Left(reason)
      }
    }
  }
}
