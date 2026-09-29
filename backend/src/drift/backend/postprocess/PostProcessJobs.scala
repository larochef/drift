package drift.backend.postprocess

import drift.backend.process.ProcessOutput
import drift.backend.runtime.LaunchRuntime
import drift.backend.sdserver.{GenerationHistory, NativeJobs}
import drift.backend.session.SessionLog
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.concurrent.*
import java.util.concurrent.atomic.*
import scala.jdk.CollectionConverters.*
import scala.util.Try
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{writeToString, WriterConfig}
import com.typesafe.scalalogging.Logger

/** What a running job is waiting on, so a cancel can reach it: the `sd-cli`
  * process it started, or the img_gen job one of its tiles submitted to an
  * sd-server. Both are set while they run and cleared after, so a cancel acts
  * on what is running now — and on nothing at all between two tiles, where the
  * flag alone stops the job at the next one.
  */
final private[postprocess] class JobCancellation {
  val stopped: AtomicBoolean = AtomicBoolean(false)

  /** Asked to pause (`specs/40-pause-and-resume.md`): read where `stopped` is,
    * between two tiles, and the only difference is what becomes of the work — a
    * pause keeps every tile it has.
    */
  val paused: AtomicBoolean = AtomicBoolean(false)
  val process: AtomicReference[Option[Process]] =
    AtomicReference[Option[Process]](None)
  val nativeJob: AtomicReference[Option[(Int, String)]] =
    AtomicReference[Option[(Int, String)]](None)
}

/** An image output of a recorded generation, as a job takes it. */
private[postprocess] case class PostProcessSource(
    date: String,
    fileName: String,
    file: Path,
    parent: Generation,
    output: GenerationOutput
)

/** The post-processing jobs of this drift run, and what every kind of job
  * shares: finding its source, recording its state, its log, running one
  * `sd-cli` command, and the derived entry it ends with.
  */
final private[postprocess] class PostProcessJobs(
    outputsRoot: Path,
    logsRoot: Path,
    history: GenerationHistory,
    storage: StorageService
) {

  /** Where this job's files are — its result, its log, its tiles (`JobFiles`).
    */
  val files: JobFiles = JobFiles(outputsRoot, logsRoot)
  private val logger = Logger[PostProcessJobs]
  private val jobs = ConcurrentHashMap[String, PostProcessJob]()
  private val counter = AtomicLong(0)

  /** One entry per running job, put there by `start` and removed when its
    * thread ends.
    */
  private val cancellations = ConcurrentHashMap[String, JobCancellation]()

  /** The log of each running job, the very ring buffer a session keeps
    * (`specs/13-log-streaming.md`): it collapses sd-cpp's bar redraws, reads
    * the progress out of them and remembers the last narrative line, so a job
    * shows what a session's panel shows without parsing anything of its own.
    */
  private val logs = ConcurrentHashMap[String, SessionLog]()

  /** Where a job's log progress comes from when the job does not own the
    * process writing it: tiles sent to a ready session are run by a server
    * whose output belongs to that session, so the job reads the session's own
    * progress instead, at the moment it is listed.
    */
  private val followed =
    ConcurrentHashMap[String, () => (Option[SessionProgress], Option[String])]()

  def list: List[PostProcessJob] =
    jobs.values.asScala.toList
      .map(job =>
        Option(followed.get(job.id)).filter(_ => job.state.isActive) match {
          case None         => job
          case Some(source) =>
            val (progress, activity) = source()
            job.copy(logProgress = progress, activity = activity)
        }
      )
      .sortBy(-_.startedAt)

  /** Reads this job's progress from `source` — a live session — until
    * `unfollow`.
    */
  def follows(
      job: PostProcessJob,
      source: () => (Option[SessionProgress], Option[String])
  ): Unit = followed.put(job.id, source)

  def unfollow(job: PostProcessJob): Unit = followed.remove(job.id)

  /** The picture each running tiled job is making (`LivePicture`), from the
    * start of its run until the run ends — a paused job's is on disk.
    */
  private val pictures = ConcurrentHashMap[String, LivePicture]()

  def showPicture(id: String, picture: LivePicture): Unit =
    pictures.put(id, picture)

  /** Forgets `picture` as this job's — only if it still is: a resumed run may
    * have put its own there already.
    */
  def hidePicture(id: String, picture: LivePicture): Unit =
    pictures.remove(id, picture)

  /** The picture a running or paused tiled job has made so far, as PNG bytes:
    * at most `side` px on its longest edge, or at full size.
    */
  def picture(id: String, side: Option[Int]): Option[Array[Byte]] =
    Option(pictures.get(id))
      .flatMap(live =>
        side.fold(
          live.fullSize().flatMap(file =>
            // Gone if the job ended since.
            try Some(Files.readAllBytes(file))
            catch { case NonFatal(_) => None }
          )
        )(live.forScreen)
      )
      .orElse(files.storedPicture(id, side))

  /** The persisted output and the gallery entry it belongs to — refused for
    * anything that is not an image output of a recorded generation.
    */
  def source(
      date: String,
      fileName: String
  ): Either[String, PostProcessSource] =
    if (!GenerationHistory.isDate(date)) Left(s"'$date' is not a date")
    else {
      val file = outputsRoot.resolve(date).resolve(fileName).normalize()
      if (!file.startsWith(outputsRoot) || !Files.isRegularFile(file))
        Left(s"'$fileName' is not an output of $date")
      else
        history
          .day(date)
          .flatMap(g => g.outputs.find(_.fileName == fileName).map(g -> _))
          .headOption match {
          case None => Left(s"'$fileName' belongs to no recorded generation")
          case Some((_, output)) if !output.mimeType.startsWith("image/") =>
            Left("only images can be upscaled")
          case Some((parent, output)) =>
            Right(PostProcessSource(date, fileName, file, parent, output))
        }
    }

  /** Why this image cannot take another job right now, if it cannot
    * (`specs/15-post-hoc-resize.md`): one at a time per output, so the tiles
    * drawn over a picture are one job's and stopping one is enough to start
    * another. `resuming` is the job being carried on, which is not in its own
    * way.
    */
  def busyWith(
      date: String,
      fileName: String,
      resuming: Option[String] = None
  ): Option[String] =
    list
      .filterNot(job => resuming.contains(job.id))
      .find(job =>
        job.sourceDate == date && job.sourceFileName == fileName &&
          (job.state.isActive || job.state == PostProcessState.Paused)
      )
      .map(job =>
        if (job.state == PostProcessState.Paused)
          s"a ${job.kind} of this image is paused: resume it or cancel it first"
        else s"a ${job.kind} of this image is already running: stop it first"
      )

  private def nextId(): String =
    s"g${System.currentTimeMillis()}-${1000 + counter.incrementAndGet()}"

  def refused(
      kind: String,
      date: String,
      fileName: String,
      reason: String
  ): PostProcessJob = {
    val now = System.currentTimeMillis()
    record(
      PostProcessJob(
        id = nextId(),
        kind = kind,
        sourceDate = date,
        sourceFileName = fileName,
        sourceGenerationId = "",
        state = PostProcessState.Failed,
        startedAt = now,
        completedAt = Some(now),
        error = Some(reason)
      )
    )
  }

  private def record(job: PostProcessJob): PostProcessJob = {
    jobs.put(job.id, job)
    job
  }

  /** Atomic: a job is updated from its own thread, its log reader and its
    * picture's painter, and none of them may undo another's change.
    */
  def update(id: String)(f: PostProcessJob => PostProcessJob): Unit =
    jobs.computeIfPresent(id, (_, job) => f(job))

  /** The paused jobs on disk, as the jobs they are — what drift starts with
    * after a restart (`specs/40-pause-and-resume.md`).
    */
  def loadPaused(): Unit =
    storage.list[PausedJob]("post-process-jobs").foreach { paused =>
      record(
        PostProcessJob(
          id = paused.id,
          kind = paused.kind,
          sourceDate = paused.sourceDate,
          sourceFileName = paused.sourceFileName,
          sourceGenerationId = "",
          state = PostProcessState.Paused,
          startedAt = paused.startedAt,
          progress = Some(
            PostProcessProgress(
              files.tilesDone(paused.id, paused.tiles),
              paused.tiles
            )
          ),
          // The picture it had made, kept when it paused.
          paintedTiles = Option
            .when(Files.isRegularFile(files.pictureFileOf(paused.id)))(
              files.tilesDone(paused.id, paused.tiles)
            )
        )
      )
    }

  /** What a paused job is, while it is paused: its work and its id. */
  def pausedWork(id: String): Option[PausedJob] =
    storage.get[PausedJob]("post-process-jobs", id)

  /** Records a running job of `kind` on `src` and runs it on its own thread — a
    * new one, or `resuming` the paused job of that id, which keeps its id, its
    * log and the tiles it already has.
    */
  def start(
      kind: String,
      src: PostProcessSource,
      progress: Option[PostProcessProgress] = None,
      resuming: Option[String] = None,
      tiles: List[ImageRegion] = List.empty
  )(run: PostProcessJob => Unit): PostProcessJob = {
    val job = record(
      PostProcessJob(
        id = resuming.getOrElse(nextId()),
        kind = kind,
        sourceDate = src.date,
        sourceFileName = src.fileName,
        sourceGenerationId = src.parent.id,
        state = PostProcessState.Running,
        startedAt = System.currentTimeMillis(),
        progress = progress,
        tiles = tiles
      )
    )
    cancellations.put(job.id, JobCancellation())
    logs.put(job.id, SessionLog())
    val thread = Thread(
      () =>
        try run(job)
        finally {
          cancellations.remove(job.id)
          logs.remove(job.id)
          work.remove(job.id)
        },
      s"drift-$kind-${job.id}"
    )
    thread.setDaemon(true)
    thread.start()
    job
  }

  /** Takes one line of a job's output — from the `sd-cli` it spawned or from
    * the sd-server a tile runs on — into its log, and mirrors what the log now
    * says onto the job itself, for the status socket to carry. The socket
    * samples twice a second, so a bar redrawing far faster than that costs
    * nothing beyond this record.
    */
  def noteLine(job: PostProcessJob, line: String): Unit =
    Option(logs.get(job.id)).foreach { log =>
      log.append(line)
      val progress = log.progress
      val activity = log.activity
      update(job.id)(current =>
        if (current.logProgress == progress && current.activity == activity)
          current
        else current.copy(logProgress = progress, activity = activity)
      )
    }

  /** Whether the job was cancelled: what the tile loop checks between tiles,
    * and what tells `fail` that a killed process is a cancel and not a failure.
    */
  /** Whether the job has been asked to pause — read between two tiles, where a
    * cancel is read (`specs/40-pause-and-resume.md`).
    */
  def isPaused(job: PostProcessJob): Boolean =
    Option(cancellations.get(job.id)).exists(_.paused.get())

  /** What a running tiled job is doing, for the record a pause writes: put
    * there when the job starts, gone when its thread ends.
    */
  private val work = ConcurrentHashMap[String, (PausedWork, Int)]()

  def remember(id: String, doing: PausedWork, tiles: Int): Unit =
    work.put(id, (doing, tiles))

  /** Asks a running job to pause after the tile in flight, and writes what it
    * takes to finish it later. False when no running tiled job has that id.
    */
  def pause(id: String, force: Boolean = false): Boolean =
    (Option(jobs.get(id)), Option(work.get(id))) match {
      case (Some(job), Some((doing, tiles))) if job.state.isActive =>
        Option(cancellations.get(id)).exists { cancellation =>
          cancellation.paused.set(true)
          // Forced: the tile in flight goes the way a cancel takes it — the
          // sd-cli killed, the img_gen job dropped on its server — and the
          // loop, seeing the pause flag, records a pause rather than a
          // failure. That tile is run again on resume.
          if (force) {
            cancellation.process.get().foreach(_.destroyForcibly())
            cancellation.nativeJob
              .get()
              .foreach((port, nativeJobId) =>
                NativeJobs.cancel(port, nativeJobId)
              )
          }
          update(id)(_.copy(pauseRequested = true))
          storage.save(
            "post-process-jobs",
            id,
            PausedJob(
              id = id,
              kind = job.kind,
              sourceDate = job.sourceDate,
              sourceFileName = job.sourceFileName,
              work = doing,
              tiles = tiles,
              startedAt = job.startedAt
            )
          )
          logger.info(
            if (force) s"${job.kind.capitalize} $id pausing, dropping its tile"
            else s"${job.kind.capitalize} $id pausing after this tile"
          )
          true
        }
      case _ => false
    }

  /** The job as it stands once its thread has stopped between two tiles. */
  def recordPaused(job: PostProcessJob, done: Int, tiles: Int): Unit = {
    update(job.id)(
      _.copy(
        state = PostProcessState.Paused,
        progress = Some(PostProcessProgress(done, tiles)),
        pauseRequested = false,
        error = None
      )
    )
    logger.info(s"${job.kind.capitalize} ${job.id} paused at $done/$tiles")
  }

  /** Forgets a paused job's record — it resumed, ended, or was cancelled. */
  def forgetPaused(id: String): Unit =
    storage.delete("post-process-jobs", id)

  def isCancelled(job: PostProcessJob): Boolean =
    Option(cancellations.get(job.id)).exists(_.stopped.get())

  /** The img_gen job a tile is now waiting for, so a cancel can stop it on the
    * server running it.
    */
  def waitingFor(job: PostProcessJob, port: Int, nativeJobId: String): Unit =
    Option(cancellations.get(job.id))
      .foreach(_.nativeJob.set(Some((port, nativeJobId))))

  def doneWaiting(job: PostProcessJob): Unit =
    Option(cancellations.get(job.id)).foreach(_.nativeJob.set(None))

  /** Cancels a running job: the `sd-cli` it waits on is killed, the img_gen job
    * a tile submitted is cancelled on its server, and a job between two tiles
    * stops at the next one. The state is recorded here so the answer is
    * immediate; the job's own thread then unwinds through `fail`, which removes
    * what it had written and records the cancel again rather than a failure. A
    * job that finished in the meantime keeps its result.
    */
  def cancel(id: String): Boolean = Option(jobs.get(id)) match {
    // A paused job has no thread to stop: its tiles and its record go, and it
    // is cancelled where it stands (`specs/40-pause-and-resume.md`).
    case Some(job) if job.state == PostProcessState.Paused =>
      pausedWork(job.id).foreach(paused =>
        (0 until paused.tiles).foreach(index =>
          files.tileOutputsOf(job.id, index).foreach(Files.deleteIfExists)
        )
      )
      files.deletePicture(job.id)
      forgetPaused(job.id)
      recordCancelled(job)
      true
    case Some(job) if job.state.isActive =>
      Option(cancellations.get(id)).foreach { cancellation =>
        cancellation.stopped.set(true)
        cancellation.process.get().foreach(_.destroyForcibly())
        cancellation.nativeJob
          .get()
          .foreach((port, nativeJobId) => NativeJobs.cancel(port, nativeJobId))
      }
      // A pause asked for and not yet reached, or a resumed job, has its
      // record on disk already: left there, it comes back paused at startup.
      forgetPaused(id)
      recordCancelled(job)
      true
    case _ => false
  }

  private def recordCancelled(job: PostProcessJob): Unit = {
    update(job.id)(
      _.copy(
        state = PostProcessState.Cancelled,
        completedAt = Some(System.currentTimeMillis()),
        error = None
      )
    )
    logger.info(s"${job.kind.capitalize} ${job.id} cancelled")
  }

  /** Starts a job's log: the job, the runtime, and notes on what drift did
    * before spawning. Every `spawn` of the job appends to it.
    */
  /** The job's log, opened with what it is about to do. A resumed job appends
    * to the log it already has (`specs/40-pause-and-resume.md`): what its first
    * run said is how it got here.
    */
  def startLog(
      job: PostProcessJob,
      launch: LaunchRuntime,
      notes: List[String],
      resumed: Boolean = false
  ): Unit = {
    Files.createDirectories(logsRoot)
    val header =
      (s"# drift ${job.kind} ${job.id} on runtime '${launch.runtime.id}'"
        :: notes.map("# " + _)).mkString("", "\n", "\n")
    if (resumed)
      Files.writeString(
        files.logFileOf(job),
        header,
        StandardOpenOption.CREATE,
        StandardOpenOption.APPEND
      )
    else Files.writeString(files.logFileOf(job), header)
  }

  def appendLog(job: PostProcessJob, line: String): Unit =
    Files.writeString(
      files.logFileOf(job),
      s"# $line\n",
      StandardOpenOption.CREATE,
      StandardOpenOption.APPEND
    )

  /** Runs one `sd-cli` command to completion, its command line and output
    * appended to the job log, and judges the run: a timeout, a non-zero exit or
    * an `[ERROR]` line is the reason returned. The exit code alone proves
    * nothing (see `EsrganUpscale`), and the log may already hold earlier runs
    * of the same job, so only this run's lines are read.
    */
  def spawn(
      job: PostProcessJob,
      command: List[String],
      launch: LaunchRuntime
  ): Option[String] = {
    val logFile = files.logFileOf(job)
    Files.writeString(
      logFile,
      s"# ${command.mkString(" ")}\n",
      StandardOpenOption.APPEND
    )
    val builder = ProcessBuilder(command*)
      .directory(Path.of(command.head).getParent.toFile)
      .redirectErrorStream(true)
    launch.environment.foreach((name, value) =>
      builder.environment().put(name, value)
    )
    val process = builder.start()
    Option(cancellations.get(job.id)).foreach(_.process.set(Some(process)))
    // The output is drained here rather than redirected to the file by the OS:
    // the file is written just the same (`ProcessOutput` mirrors it verbatim),
    // and the lines pass through the job's log on the way, which is what puts
    // a bar on screen while sd-cli draws it.
    val error = AtomicReference(Option.empty[String])
    val drained = CountDownLatch(1)
    ProcessOutput.capture(s"postprocess-${job.id}", process, logFile) { line =>
      noteLine(job, line)
      // sd-cpp pads its level names: "[ERROR  ]". Only this run's lines are
      // looked at, since a job's log may already hold earlier ones.
      if (line.contains("[ERROR"))
        error.compareAndSet(None, Some(LogProgress.clean(line)))
    }(() => drained.countDown())
    try judge(process, error, drained)
    finally Option(cancellations.get(job.id)).foreach(_.process.set(None))
  }

  /** How a finished `sd-cli` run is judged: the exit code alone proves nothing
    * (see `EsrganUpscale`), so the run's own `[ERROR]` lines decide — read once
    * the reader has drained what was still in flight when the process ended.
    */
  private def judge(
      process: Process,
      error: AtomicReference[Option[String]],
      drained: CountDownLatch
  ): Option[String] =
    if (!process.waitFor(PostProcessJobs.TimeoutMinutes, TimeUnit.MINUTES)) {
      process.destroyForcibly()
      Some(
        s"sd-cli did not finish within ${PostProcessJobs.TimeoutMinutes} minutes"
      )
    } else {
      drained.await(10, TimeUnit.SECONDS)
      if (process.exitValue() != 0)
        Some(s"sd-cli exited with ${process.exitValue()}")
      else error.get().map(line => s"sd-cli reported an error: $line")
    }

  /** A cancelled job ends here too — its process was killed, or its tile's job
    * cancelled on the server, so the run reports a failure the user asked for.
    * It is recorded as the cancel it is, without a reason or a log tail.
    */
  def fail(job: PostProcessJob, reason: String): Unit = {
    forgetPaused(job.id)
    if (isCancelled(job)) recordCancelled(job)
    else {
      // The log ends with why: read later, it must not look as if the job
      // simply stopped.
      Try(appendLog(job, s"failed: $reason"))
      update(job.id)(
        _.copy(
          state = PostProcessState.Failed,
          completedAt = Some(System.currentTimeMillis()),
          error = Some(reason),
          outputTail = tailOf(files.logFileOf(job))
        )
      )
      logger.warn(s"${job.kind} ${job.id}: $reason")
    }
  }

  /** Writes the derived entry's sidecar beside the output and completes the job
    * with it. The entry keeps the parent's run configuration so the gallery's
    * filters still find it; it has no session and no request.
    */
  def complete(
      job: PostProcessJob,
      src: PostProcessSource,
      outputFile: Path,
      derivation: Derivation
  ): Unit = {
    val now = System.currentTimeMillis()
    val generation = Generation(
      id = job.id,
      sessionId = "",
      runConfigurationId = src.parent.runConfigurationId,
      kind = job.kind,
      status = GenerationStatus.Completed,
      submittedAt = job.startedAt,
      startedAt = Some(job.startedAt),
      completedAt = Some(now),
      outputs = List(
        GenerationOutput(
          date = src.date,
          fileName = outputFile.getFileName.toString,
          url = s"/api/outputs/${src.date}/${outputFile.getFileName}",
          mimeType = "image/png",
          format = "png"
        )
      ),
      derivation = Some(derivation),
      importedFileName = None,
      projectId = src.parent.projectId,
      promptVersionId = src.parent.promptVersionId
    )
    Files.write(
      outputsRoot.resolve(src.date).resolve(s"${job.id}.json"),
      writeToString(generation, WriterConfig.withIndentionStep(2))
        .getBytes(StandardCharsets.UTF_8)
    )
    update(job.id)(
      _.copy(
        state = PostProcessState.Completed,
        completedAt = Some(now),
        result = Some(generation)
      )
    )
    logger.info(
      s"${job.kind.capitalize} ${job.id}: ${src.fileName} → ${outputFile.getFileName} (${derivation.width.getOrElse(0)}×${derivation.height.getOrElse(0)})"
    )
  }

  private def tailOf(logFile: Path): List[String] =
    try
      String(
        Files.readAllBytes(logFile),
        StandardCharsets.UTF_8
      ).linesIterator.toList
        .takeRight(30)
    catch { case NonFatal(_) => List.empty }
}

private[postprocess] object PostProcessJobs {

  val TimeoutMinutes: Long = 30
}
