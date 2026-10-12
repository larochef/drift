package drift.runner.server

import drift.protocol.*
import drift.runner.diffusion.ProgressReport
import drift.runner.lora.LoraReport
import drift.runner.server.JobAnswers.{failed, Answer}

import java.util.UUID
import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue}
import scala.jdk.CollectionConverters.*

/** One job of `kind`: `run` makes its result document. */
final private[server] class Job(
    val id: String,
    val created: Long,
    val kind: String,
    val run: () => NativeJobResult
) {
  @volatile var status: String = "queued"
  @volatile var started: Option[Long] = None
  @volatile var completed: Option[Long] = None
  @volatile var result: Option[NativeJobResult] = None
  @volatile var error: Option[String] = None

  /** Asked to stop while it generates: read between two steps. */
  @volatile var cancelRequested: Boolean = false

  /** Whether its request asked for step previews (`preview`, anything but
    * `none`, as sd-server's): only then are they drawn.
    */
  @volatile var previewWanted: Boolean = false

  /** What the job could not do as asked and went on without (a LoRA left
    * unapplied), for whoever shows the result.
    */
  val warnings = new java.util.concurrent.CopyOnWriteArrayList[String]()

  /** The last step's preview of the image or video being made, as the job's
    * document gives it.
    */
  @volatile var preview: Option[NativeJobPreview] = None

  /** What the job says it is doing, while it generates. */
  @volatile var progress: Option[JobProgress] = None
}

/** What a step throws once its job is asked to stop: the generation unwinds the
  * way a failed one does, and the job ends `cancelled`.
  */
final private[server] class JobCancelled extends RuntimeException("cancelled")

/** What a job throws when `reason` ended it with part of its work done and
  * kept: the job ends `cancelled` or `failed` as `reason` says, and carries
  * `result` all the same (a video's upscale, `VideoUpscale`).
  */
final private[server] class PartialResult(
    val result: NativeJobResult,
    reason: Throwable
) extends RuntimeException(reason)

/** The server's jobs: queued as they are submitted, run one at a time by the
  * worker, polled, and cancelled — while queued, or between two steps once
  * generating.
  */
final private[server] class JobQueue {

  /** The job the worker is generating, if any. */
  @volatile private var generatingJob: Option[Job] = None

  /** That job, as what makes its result reads it. */
  def generating: Option[Job] = generatingJob

  private val jobs = new ConcurrentHashMap[String, Job]()
  private val queue = new LinkedBlockingQueue[Job]()

  def enqueue(
      kind: String,
      run: () => NativeJobResult,
      previews: Boolean = false,
      warnings: Seq[String] = Nil
  ): Answer = {
    val job = new Job(
      UUID.randomUUID().toString,
      System.currentTimeMillis() / 1000,
      kind,
      run
    )
    job.previewWanted = previews
    warnings.foreach { warning =>
      println(s"[WARN] $warning")
      job.warnings.add(warning)
    }
    jobs.put(job.id, job)
    queue.put(job)
    Right(
      NativeJob(
        id = job.id,
        kind = kind,
        status = "queued",
        created = job.created,
        pollUrl = Some(JobRoutes.job(job.id))
      )
    )
  }

  /** `GET /sdcpp/v1/jobs/{id}`: the job, with its preview while it samples. */
  def polled(id: String): Answer =
    Option(jobs.get(id)) match {
      case None      => failed(404, "no such job")
      case Some(job) =>
        Right(
          describe(job).copy(preview = job.preview, progress = job.progress)
        )
    }

  /** `POST /sdcpp/v1/jobs/{id}/cancel`. */
  def cancelled(id: String): Answer =
    Option(jobs.get(id)) match {
      case None      => failed(404, "no such job")
      case Some(job) =>
        if (job.status == "queued" && queue.remove(job)) {
          job.status = "cancelled"
          job.completed = Some(System.currentTimeMillis() / 1000)
          Right(describe(job))
        } else if (job.status == "generating") {
          // Stopped at the next step, not here: the answer says it still
          // generates, and the next poll finds it cancelled.
          job.cancelRequested = true
          Right(describe(job))
        } else failed(409, s"the job is ${job.status}")
    }

  /** The job's document, without its preview: only a poll carries that. */
  private def describe(job: Job): NativeJob =
    NativeJob(
      id = job.id,
      kind = job.kind,
      status = job.status,
      created = job.created,
      started = job.started,
      completed = job.completed,
      queuePosition = math.max(queue.asScala.toSeq.indexOf(job), 0),
      // a completed job's, or what a stopped one keeps (`PartialResult`)
      result = job.result,
      error = job.error.map(NativeJobError("generation_failed", _)),
      warnings = job.warnings.asScala.toList
    )

  /** The worker: runs the queue in order. */
  def work(): Unit =
    while (true) {
      val job = queue.take()
      job.status = "generating"
      job.started = Some(System.currentTimeMillis() / 1000)
      generatingJob = Some(job)
      LoraReport.listener = warning => job.warnings.add(warning)
      ProgressReport.listener = told => job.progress = Some(told)
      ProgressReport.begin()
      def ended(reason: Throwable): Unit = reason match {
        case partial: PartialResult =>
          job.result = Some(partial.result)
          ended(partial.getCause)
        case _: JobCancelled =>
          println("generation cancelled")
          job.status = "cancelled"
        case error =>
          System.err.println(s"[ERROR] generation failed: $error")
          job.error = Some(String.valueOf(error.getMessage))
          job.status = "failed"
      }
      try {
        job.result = Some(job.run())
        job.status = "completed"
      } catch { case reason: Throwable => ended(reason) }
      LoraReport.listener = _ => ()
      ProgressReport.listener = _ => ()
      job.progress = None
      generatingJob = None
      job.completed = Some(System.currentTimeMillis() / 1000)
    }
}
