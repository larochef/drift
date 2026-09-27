package drift.backend.conversion

import drift.backend.process.ProcessOutput
import drift.backend.runtime.LaunchRuntime
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.concurrent.*
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}
import scala.collection.mutable.ArrayDeque
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** The pre-step a quantized source needs before sd-cli sees it: drift
  * dequantizes `source` into `intermediate`, which the command then reads, and
  * removes it afterwards unless `keep`.
  */
private[conversion] case class Dequantization(
    source: Path,
    intermediate: Path,
    keep: Boolean
)

/** A conversion waiting for the worker: the command to run and what to do with
  * the output once sd-cli is done with it.
  */
private[conversion] case class QueuedConversion(
    id: String,
    /** Where sd-cli writes; renamed to the job's `outputPath` on success. */
    partFile: Path,
    command: List[String],
    launch: LaunchRuntime,
    /** Registers the output; the completed job, or why it could not be. */
    complete: ConversionJob => Either[String, ConversionJob],
    dequantize: Option[Dequantization] = None
)

/** The conversions of this drift run, run one at a time — quantizing is CPU and
  * disk bound, so two at once only make both slower. Each dequantizes first
  * when its source needs it, spawns one `sd-cli -M convert`, reads its progress
  * bar off the pipe like a session's log, and keeps the last lines for the
  * failure report.
  */
final private[conversion] class ConversionJobs(logsRoot: Path) {
  private val logger = Logger[ConversionJobs]
  private val jobs = ConcurrentHashMap[String, ConversionJob]()
  private val queue = LinkedBlockingQueue[QueuedConversion]()
  private val counter = AtomicLong(0)

  /** The job running now and its process, for Cancel. */
  private val running = AtomicReference(Option.empty[(String, Process)])

  private val worker = Thread(
    () => {
      while (true) {
        try run(queue.take())
        catch {
          case NonFatal(err) => logger.warn("conversion worker", err)
        }
      }
    },
    "drift-conversions"
  )
  worker.setDaemon(true)
  worker.start()

  def list: List[ConversionJob] =
    jobs.values.asScala.toList.sortBy(-_.startedAt)

  def nextId(): String =
    s"c${System.currentTimeMillis()}-${1000 + counter.incrementAndGet()}"

  def enqueue(job: ConversionJob, queued: QueuedConversion): ConversionJob = {
    jobs.put(job.id, job)
    queue.put(queued)
    job
  }

  private def update(id: String)(f: ConversionJob => ConversionJob): Unit =
    Option(jobs.get(id)).foreach(job => jobs.put(id, f(job)))

  private def stateOf(id: String): ConversionState = jobs.get(id).state

  /** Cancels a queued or running job. A queued one is simply skipped when its
    * turn comes; a dequantizing one stops at the next tensor; a converting one
    * has its process killed. The worker removes the partial outputs.
    */
  def cancel(id: String): Boolean = Option(jobs.get(id)) match {
    case Some(job) if job.state.isActive =>
      update(id)(
        _.copy(
          state = ConversionState.Cancelled,
          detail = "",
          completedAt = Some(System.currentTimeMillis())
        )
      )
      running.get().filter(_._1 == id).foreach(_._2.destroyForcibly())
      true
    case _ => false
  }

  private def logFileOf(id: String): Path =
    logsRoot.resolve(s"conversion-$id.log")

  private def appendLog(logFile: Path, line: String): Unit =
    Files.writeString(
      logFile,
      s"# $line\n",
      StandardCharsets.UTF_8,
      StandardOpenOption.CREATE,
      StandardOpenOption.APPEND
    )

  private def run(item: QueuedConversion): Unit = {
    val id = item.id
    val job = jobs.get(id)
    // Cancelled while waiting: nothing was written, nothing to undo.
    if (job == null || job.state != ConversionState.Queued) return
    Files.createDirectories(logsRoot)
    val logFile = logFileOf(id)
    Files.writeString(
      logFile,
      s"# drift conversion $id on runtime '${item.launch.runtime.id}'\n",
      StandardCharsets.UTF_8
    )
    try {
      if (item.dequantize.forall(step => dequantize(item, step, logFile)))
        convert(item, logFile)
    } finally
      item.dequantize
        .filterNot(_.keep)
        .foreach(step => Files.deleteIfExists(step.intermediate))
  }

  /** The pre-step: false when the job ended here, cancelled or failed. */
  private def dequantize(
      item: QueuedConversion,
      step: Dequantization,
      logFile: Path
  ): Boolean = {
    val id = item.id
    update(id)(
      _.copy(state = ConversionState.Dequantizing, detail = "dequantizing…")
    )
    val part =
      step.intermediate.resolveSibling(s"${step.intermediate.getFileName}.part")
    appendLog(logFile, s"dequantizing ${step.source} → ${step.intermediate}")
    Dequantizer.run(
      step.source,
      part,
      keepGoing = () => stateOf(id) == ConversionState.Dequantizing,
      onProgress = (done, total) =>
        update(id)(
          _.copy(
            progress = Some(PostProcessProgress(done, total)),
            detail = s"$done of $total tensors"
          )
        )
    ) match {
      case Right(plan) =>
        Files.move(part, step.intermediate, StandardCopyOption.ATOMIC_MOVE)
        appendLog(
          logFile,
          s"dequantized ${plan.dequantized} of ${plan.tensors.size} tensors (${plan.dropped.size} scale/marker tensors dropped), ${plan.outputBytes} bytes" +
            (if (step.keep) ", kept" else "")
        )
        true
      case Left(reason) =>
        Files.deleteIfExists(part)
        if (stateOf(id) == ConversionState.Cancelled)
          logger.info(s"Conversion $id cancelled while dequantizing")
        else fail(item, reason, Nil)
        false
    }
  }

  private def convert(item: QueuedConversion, logFile: Path): Unit = {
    val id = item.id
    if (stateOf(id) == ConversionState.Cancelled) return
    update(id)(
      _.copy(
        state = ConversionState.Converting,
        progress = None,
        detail = "starting sd-cli…"
      )
    )
    appendLog(logFile, item.command.mkString(" "))
    val tail = ArrayDeque[String]()
    var lastError = Option.empty[String]
    val builder = ProcessBuilder(item.command*)
      .directory(Path.of(item.command.head).getParent.toFile)
      .redirectErrorStream(true)
    item.launch.environment.foreach((name, value) =>
      builder.environment().put(name, value)
    )
    val process =
      try builder.start()
      catch {
        case NonFatal(err) =>
          fail(item, s"sd-cli could not start: ${err.getMessage}", Nil)
          return
      }
    running.set(Some(id -> process))
    val drained = CountDownLatch(1)
    ProcessOutput.capture(s"drift-conversion-$id", process, logFile) { raw =>
      val line = LogProgress.clean(raw)
      if (line.nonEmpty)
        LogProgress.parse(line) match {
          case Some(progress) =>
            update(id)(
              _.copy(
                progress =
                  Some(PostProcessProgress(progress.done, progress.total)),
                detail = progress.detail
              )
            )
          case None =>
            tail.synchronized {
              tail.append(line)
              if (tail.size > ConversionJobs.TailLines) tail.removeHead()
            }
            // sd-cpp pads its level names: "[ERROR  ]".
            if (lastError.isEmpty && LogProgress.looksLikeError(line))
              lastError = Some(line)
        }
    }(() => drained.countDown())
    val exit = process.waitFor()
    // The reader outlives the process by the last chunk of its output.
    drained.await()
    running.set(None)
    val lines = tail.synchronized(tail.toList)
    if (stateOf(id) == ConversionState.Cancelled) {
      Files.deleteIfExists(item.partFile)
      logger.info(s"Conversion $id cancelled")
    } else if (exit != 0)
      fail(
        item,
        s"sd-cli exited with $exit${lastError.map(e => s": $e").getOrElse("")}",
        lines
      )
    else
      lastError match {
        case Some(error) =>
          fail(item, s"sd-cli reported an error: $error", lines)
        case None =>
          update(id)(
            _.copy(state = ConversionState.Registering, detail = "registering…")
          )
          item.complete(jobs.get(id)) match {
            case Right(completed) =>
              jobs.put(
                id,
                completed.copy(
                  state = ConversionState.Completed,
                  detail = "",
                  completedAt = Some(System.currentTimeMillis()),
                  logTail = lines
                )
              )
              logger.info(
                s"Conversion $id: ${completed.sourceLabel} → ${Path.of(completed.outputPath).getFileName} as model ${completed.modelId.getOrElse("?")}"
              )
            case Left(reason) => fail(item, reason, lines)
          }
      }
  }

  private def fail(
      item: QueuedConversion,
      reason: String,
      lines: List[String]
  ): Unit = {
    try Files.deleteIfExists(item.partFile)
    catch { case NonFatal(_) => () }
    update(item.id)(
      _.copy(
        state = ConversionState.Failed,
        detail = "",
        completedAt = Some(System.currentTimeMillis()),
        error = Some(reason),
        logTail = lines
      )
    )
    logger.warn(s"Conversion ${item.id}: $reason")
  }
}

private[conversion] object ConversionJobs {
  val TailLines: Int = 30
}
