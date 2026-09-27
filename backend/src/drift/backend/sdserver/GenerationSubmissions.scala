package drift.backend.sdserver

import drift.backend.projects.ProjectManager
import drift.shared.*

import java.net.http.*
import java.util.concurrent.atomic.AtomicLong
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.typesafe.scalalogging.Logger

/** Submits a generation to its session's sd-server. A negative seed never
  * reaches sd-server: drift draws one here, so the forwarded request and the
  * sidecar always carry the seed actually used (the UI draws its own before
  * submitting; this covers direct API callers). sd-server at the commit in use
  * answered a -1 with its default 42, every time — not random.
  *
  * The forwarded body keeps the base64 images sd-server needs, its reference
  * images prepared for the session's build (`ReferenceImages`); the *recorded*
  * parameters carry URLs of the inputs persisted beside the outputs instead,
  * the references as they were given — a poll of the generation list must not
  * ship megabytes of base64 every second, and neither should the sidecar.
  */
final private[sdserver] class GenerationSubmissions(
    registry: GenerationRegistry,
    files: GenerationFiles,
    monitor: GenerationMonitor,
    projectManager: ProjectManager,
    /** The architecture and runtime of a session, while they resolve. */
    architectureAndRuntimeOf: String => Option[(Architecture, Runtime)]
) {
  private val logger = Logger[GenerationSubmissions]
  private val counter = AtomicLong(0)

  def image(
      sessionId: String,
      requested: ImageGenerationParameters,
      context: SubmitContext,
      scratch: Boolean
  ): Generation = {
    val parameters =
      if (requested.seed < 0)
        requested.copy(seed = GenerationManager.randomSeed())
      else requested
    val submittedAt = System.currentTimeMillis()
    val generationId = nextGenerationId(submittedAt)
    val recorded = parameters.copy(
      initImage = parameters.initImage.map(
        files.externalizeInput(generationId, submittedAt, "init", _, scratch)
      ),
      maskImage = parameters.maskImage.map(
        files.externalizeInput(generationId, submittedAt, "mask", _, scratch)
      ),
      refImages = parameters.refImages.zipWithIndex.map((image, index) =>
        files.externalizeInput(
          generationId,
          submittedAt,
          s"ref$index",
          image,
          scratch
        )
      )
    )
    submit(
      sessionId,
      generationId,
      submittedAt,
      kind = "img_gen",
      path = "/sdcpp/v1/img_gen",
      body = writeToString(
        architectureAndRuntimeOf(sessionId).fold(parameters)(
          (architecture, runtime) =>
            ReferenceImages.prepared(parameters, architecture, runtime)
        )
      ),
      context = context,
      scratch = scratch,
      attachParameters = _.copy(imageParameters = Some(recorded))
    )
  }

  def video(
      sessionId: String,
      requested: VideoGenerationParameters,
      context: SubmitContext,
      scratch: Boolean
  ): Generation = {
    val parameters =
      if (requested.seed < 0)
        requested.copy(seed = GenerationManager.randomSeed())
      else requested
    val submittedAt = System.currentTimeMillis()
    val generationId = nextGenerationId(submittedAt)
    val recorded = parameters.copy(
      initImage = parameters.initImage.map(
        files.externalizeInput(generationId, submittedAt, "init", _, scratch)
      ),
      endImage = parameters.endImage.map(
        files.externalizeInput(generationId, submittedAt, "end", _, scratch)
      ),
      controlFrames =
        parameters.controlFrames.zipWithIndex.map((image, index) =>
          files.externalizeInput(
            generationId,
            submittedAt,
            s"frame$index",
            image,
            scratch
          )
        )
    )
    submit(
      sessionId,
      generationId,
      submittedAt,
      kind = "vid_gen",
      path = "/sdcpp/v1/vid_gen",
      body = writeToString(parameters),
      context = context,
      scratch = scratch,
      attachParameters = _.copy(videoParameters = Some(recorded))
    )
  }

  private def nextGenerationId(submittedAt: Long): String =
    s"g$submittedAt-${counter.incrementAndGet()}"

  private def record(
      generation: Generation,
      nativeJobId: String,
      port: Int
  ): Generation = {
    registry.record(GenerationEntry(generation, nativeJobId, port))
    generation
  }

  /** A submission that cannot proceed still answers with a generation —
    * `Failed`, with the reason — and is kept in the list so the refusal is
    * visible wherever generations are shown (the launch-refusal pattern).
    */
  private def submit(
      sessionId: String,
      generationId: String,
      submittedAt: Long,
      kind: String,
      path: String,
      body: String,
      attachParameters: Generation => Generation,
      context: SubmitContext,
      scratch: Boolean
  ): Generation = {

    def blank(runConfigurationId: String): Generation = attachParameters(
      Generation(
        id = generationId,
        sessionId = sessionId,
        runConfigurationId = runConfigurationId,
        kind = kind,
        status = GenerationStatus.Queued,
        submittedAt = submittedAt,
        scratch = scratch
      )
    )

    registry.readySession(sessionId) match {
      case Left(reason) =>
        record(
          blank(runConfigurationId = "").copy(
            status = GenerationStatus.Failed,
            error = Some(reason)
          ),
          nativeJobId = "",
          port = 0
        )
      case Right((session, port)) =>
        val untagged = blank(session.runConfigurationId)
        // Inside a project the recorded request decides the version — the same
        // request the sidecar stores — before anything reaches sd-server.
        projectManager.versionFor(context, untagged) match {
          case Left(reason) =>
            record(
              untagged.copy(
                status = GenerationStatus.Failed,
                error = Some(reason)
              ),
              nativeJobId = "",
              port = port
            )
          case Right(tags) =>
            forward(
              untagged.copy(
                projectId = tags.map(_._1),
                promptVersionId = tags.map(_._2)
              ),
              path,
              body,
              port
            )
        }
    }
  }

  /** Hands the native request to sd-server and follows the job it answers. */
  private def forward(
      generation: Generation,
      path: String,
      body: String,
      port: Int
  ): Generation =
    try {
      val response = registry.send(
        HttpRequest
          .newBuilder(registry.uri(port, path))
          .timeout(java.time.Duration.ofSeconds(15))
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(body))
          .build()
      )
      if (response.statusCode == 202) {
        val job = readFromString[NativeJob](response.body)
        logger.info(
          s"Generation ${generation.id}: submitted ${generation.kind} as job ${job.id} on :$port"
        )
        val recorded = record(generation, job.id, port)
        registry.get(generation.id).foreach(monitor.watch)
        recorded
      } else
        record(
          generation.copy(
            status = GenerationStatus.Failed,
            error = Some(
              s"sd-server refused the job (${response.statusCode}): ${response.body.trim.take(500)}"
            )
          ),
          nativeJobId = "",
          port = port
        )
    } catch {
      case NonFatal(err) =>
        record(
          generation.copy(
            status = GenerationStatus.Failed,
            error = Some(s"submitting the job failed: ${err.getMessage}")
          ),
          nativeJobId = "",
          port = port
        )
    }
}
