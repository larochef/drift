package drift.backend.process

import drift.backend.Background

import java.lang.ProcessBuilder.Redirect
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.concurrent.{ConcurrentHashMap, TimeUnit}
import scala.concurrent.duration.{DurationInt, FiniteDuration}
import scala.jdk.CollectionConverters.*

import com.typesafe.scalalogging.Logger
import ox.*
import ox.channels.Channel

/** Every process drift starts — a server, an `sd-cli`, a tool run for its
  * answer — is started here, so that each is known while it lives: stopping
  * drift stops them all, and there is one way to stop one.
  */
object ChildProcesses {
  private val logger = Logger("drift.backend.process.ChildProcesses")

  /** How long a child gets to end by itself once asked to. */
  val GraceSeconds: Long = 10

  private val running = ConcurrentHashMap.newKeySet[Process]()

  /** Starts `command` with `environment` added to drift's own, in `directory`
    * when one is given. Its error output is merged into its output unless it is
    * discarded. The output is a pipe the caller must read, or the child blocks
    * once it is full — unless it is sent to a file (`outputTo`).
    */
  def start(
      command: List[String],
      environment: Map[String, String] = Map.empty,
      directory: Option[Path] = None,
      discardErrors: Boolean = false,
      outputTo: Option[Path] = None
  ): Process = {
    val builder = ProcessBuilder(command*)
    directory.foreach(path => builder.directory(path.toFile))
    outputTo.foreach(file => builder.redirectOutput(file.toFile))
    if (discardErrors) builder.redirectError(Redirect.DISCARD)
    else builder.redirectErrorStream(true)
    environment.foreach((name, value) => builder.environment().put(name, value))
    val process = builder.start()
    running.add(process)
    process.onExit().thenRun(() => { running.remove(process); () })
    process
  }

  /** SIGTERM, then SIGKILL after the grace period. */
  def terminate(process: Process): Unit = stop(List(process))

  /** Every child still alive, stopped: what drift does as it stops. */
  def stopAll(): Unit = stop(running.asScala.toList)

  /** Asks all of `processes` to end at once and gives them the grace period
    * together, so stopping many takes no longer than stopping one.
    */
  def stop(processes: List[Process]): Unit = {
    val alive = processes.filter(_.isAlive)
    alive.foreach(_.destroy())
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(GraceSeconds)
    alive.foreach { process =>
      val remaining = deadline - System.nanoTime()
      if (remaining <= 0 || !process.waitFor(remaining, TimeUnit.NANOSECONDS)) {
        logger.warn(
          s"Pid ${process.pid()} ignored SIGTERM for ${GraceSeconds}s; killing"
        )
        process.destroyForcibly()
        process.waitFor(5, TimeUnit.SECONDS)
      }
    }
  }

  /** A long run followed to its end: its output drained to `logFile` and handed
    * over line by line (`ProcessOutput`), the process stopped when it outlasts
    * `within`. Answers the exit code, or `None` for a run that was stopped for
    * lasting too long.
    *
    * The output is read in `background`, not in a fork this call waits for: a
    * child may leave a helper holding its output open, a read of that pipe
    * cannot be interrupted, and the run must still end with the child. The
    * reader is given a moment to hand over the last lines, then left to end
    * when the pipe does.
    */
  def runToEnd(
      name: String,
      process: Process,
      logFile: Path,
      within: Option[FiniteDuration],
      background: Background
  )(onLine: String => Unit): Option[Int] = {
    val drained = Channel.unlimited[Unit]
    ProcessOutput.capture(name, process, logFile, background)(onLine)(() =>
      drained.send(())
    )
    val finished = within match {
      case Some(limit) => process.waitFor(limit.toMillis, TimeUnit.MILLISECONDS)
      case None        => process.waitFor(); true
    }
    if (!finished) terminate(process)
    timeoutOption(LastLines)(drained.receive()).discard
    Option.when(finished)(process.exitValue())
  }

  /** How long the reader has, once the process has ended, to hand over what was
    * still in the pipe.
    */
  private val LastLines = 5.seconds

  /** What a short run answered. */
  case class Answer(exit: Int, bytes: Array[Byte]) {
    def text: String = String(bytes, StandardCharsets.UTF_8)
  }

  /** A tool run for its answer: its output goes to a file of its own rather
    * than a pipe, so nothing has to read it while it runs and a helper it
    * leaves behind holds nothing drift waits on. At most `most` bytes of it are
    * kept. The tool is stopped when it outlasts `within`, which is then the
    * reason on the left. A tool that cannot be started throws, as `start` does.
    */
  def answer(
      command: List[String],
      within: FiniteDuration,
      most: Int,
      environment: Map[String, String] = Map.empty,
      discardErrors: Boolean = false
  ): Either[String, Answer] = {
    val output = Files.createTempFile("drift-answer", ".out")
    try {
      val process = start(
        command,
        environment,
        discardErrors = discardErrors,
        outputTo = Some(output)
      )
      // Nothing is typed to it: a tool waiting on its input ends at once.
      process.getOutputStream.close()
      if (process.waitFor(within.toMillis, TimeUnit.MILLISECONDS)) {
        val stream = Files.newInputStream(output)
        val bytes =
          try stream.readNBytes(most)
          finally stream.close()
        Right(Answer(process.exitValue(), bytes))
      } else {
        terminate(process)
        Left(s"${command.head} did not finish within $within")
      }
    } finally Files.deleteIfExists(output).discard
  }
}
