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
  * The forwarded body keeps the base64 inputs sd-server needs, its reference
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
    val seeded =
      if (requested.seed < 0)
        requested.copy(seed = GenerationManager.randomSeed())
      else requested
    val inputs = ServedInputs(files)
    val parameters = seeded.copy(
      initImage = seeded.initImage.map(inputs.inlined("init", _)),
      maskImage = seeded.maskImage.map(inputs.inlined("mask", _)),
      refImages = seeded.refImages.zipWithIndex.map((image, index) =>
        inputs.inlined(s"ref$index", image)
      )
    )
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
      inputs = inputs,
      attachParameters = _.copy(imageParameters = Some(recorded))
    )
  }

  def video(
      sessionId: String,
      requested: VideoGenerationParameters,
      context: SubmitContext,
      scratch: Boolean
  ): Generation = {
    val seeded =
      if (requested.seed < 0)
        requested.copy(seed = GenerationManager.randomSeed())
      else requested
    val inputs = ServedInputs(files)
    val parameters = seeded.copy(
      initImage = seeded.initImage.map(inputs.inlined("init", _)),
      endImage = seeded.endImage.map(inputs.inlined("end", _)),
      controlFrames = seeded.controlFrames.zipWithIndex.map((image, index) =>
        inputs.inlined(s"frame$index", image)
      ),
      references = seeded.references.zipWithIndex.map((media, index) =>
        inputs.inlined(s"reference$index", media)
      ),
      guides = seeded.guides.zipWithIndex.map((guide, index) =>
        guide.copy(media = inputs.inlined(s"guide$index", guide.media))
      ),
      controlVideo = seeded.controlVideo.map(inputs.inlined("control", _)),
      controlMask = seeded.controlMask.map(inputs.inlined("control-mask", _)),
      sourceVideo = seeded.sourceVideo.map(inputs.inlined("source", _))
    )
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
        ),
      references = parameters.references.zipWithIndex.map((media, index) =>
        files.externalizeInput(
          generationId,
          submittedAt,
          s"reference$index",
          media,
          scratch
        )
      ),
      guides = parameters.guides.zipWithIndex.map((guide, index) =>
        guide.copy(media =
          files.externalizeInput(
            generationId,
            submittedAt,
            s"guide$index",
            guide.media,
            scratch
          )
        )
      ),
      controlVideo = parameters.controlVideo.map(
        files.externalizeInput(generationId, submittedAt, "control", _, scratch)
      ),
      controlMask = parameters.controlMask.map(
        files.externalizeInput(
          generationId,
          submittedAt,
          "control-mask",
          _,
          scratch
        )
      ),
      sourceVideo = parameters.sourceVideo.map(
        files.externalizeInput(generationId, submittedAt, "source", _, scratch)
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
      inputs = inputs,
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
      scratch: Boolean,
      inputs: ServedInputs
  ): Generation = {

    def blank(runConfigurationId: String): Generation = attachParameters(
      Generation(
        id = generationId,
        sessionId = sessionId,
        runConfigurationId = runConfigurationId,
        kind = kind,
        status = GenerationStatus.Queued,
        submittedAt = submittedAt,
        importedFileName = None,
        inputSources = inputs.sources,
        scratch = scratch
      )
    )

    registry
      .readySession(sessionId)
      .flatMap(ready => inputs.checked(ready)) match {
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

/** The inputs of one submission that arrive as the URL of a file drift serves —
  * a gallery output picked in the form (`specs/50-inputs-from-the-gallery.md`)
  * — turned into the bytes sd-server takes, each remembered with the slot it
  * fills, which is what the generation records as where its inputs came from.
  */
final private[sdserver] class ServedInputs(files: GenerationFiles) {
  private var found = List.empty[InputSource]
  private var missing = Option.empty[String]

  /** `value` as bytes: itself, unless it names a served file. */
  def inlined(slot: String, value: String): String =
    files.servedInput(value) match {
      case None                => value
      case Some(Right(served)) =>
        found = found ++ served.source.map(_.copy(slot = slot))
        served.data
      case Some(Left(reason)) =>
        missing = missing.orElse(Some(reason))
        value
    }

  /** The gallery entries the inputs came from, in the request's order. */
  def sources: List[InputSource] = found

  /** `ready` unless an input named a file that is gone, which refuses the
    * submission rather than sending the server a URL.
    */
  def checked[A](ready: A): Either[String, A] = missing.toLeft(ready)
}
