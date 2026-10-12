package drift.runner.server

import drift.protocol.*
import drift.protocol.JobEndpoints.Asked
import drift.runner.diffusion.*
import drift.runner.server.JobAnswers.{failed, Answer}
import drift.runner.server.JobRequests.*

import java.nio.file.{Files, Paths}
import scala.util.Random

/** `img_gen` requests, read into jobs on `queue`. */
final private[server] class ImageJobs(
    options: ImageOptions,
    launch: LaunchDefaults,
    queue: JobQueue,
    results: JobResults
) {

  /** `POST /sdcpp/v1/img_gen`: a job from the request over the launch flags'
    * values; what the runner cannot do yet is refused (400).
    */
  def submit(
      pipeline: ImagePipeline,
      asked: Asked[ImageJobRequest]
  ): Answer = {
    val body = asked match {
      case Right(read)   => read
      case Left(problem) =>
        return failed(400, problem)
    }
    val sample = body.sampleParams.getOrElse(SampleRequest.none)
    val guidance = sample.guidance
    val initImage = body.initImage.filter(_.nonEmpty)
    val maskImage = body.maskImage.filter(_.nonEmpty)
    val referenceImages = body.refImages.getOrElse(Nil).filter(_.nonEmpty)
    val loras = body.lora.getOrElse(Nil)
    val unsupported = Seq(
      Option.when(initImage.nonEmpty && !pipeline.takesInitImage)(
        "init images"
      ),
      Option.when(maskImage.nonEmpty && !pipeline.takesMask)("masks"),
      Option.when(maskImage.nonEmpty && initImage.isEmpty)(
        "a mask without an init image"
      ),
      Option.when(referenceImages.nonEmpty && !pipeline.takesReferences)(
        "reference images"
      ),
      Option.when(loras.nonEmpty && !pipeline.takesLoras)("LoRAs"),
      Option.when(sample.sigmas.nonEmpty && !pipeline.takesSigmas)(
        "custom sigmas"
      ),
      Option.when(body.hires.exists(_.enabled.contains(true)))("hires fix"),
      Option.when(body.vaeTilingParams.exists(_.enabled.contains(true)))(
        "VAE tiling"
      )
    ).flatten
    if (unsupported.nonEmpty)
      return failed(
        400,
        s"not supported by the drift runner for ${pipeline.family} yet: ${unsupported.mkString(", ")}"
      )
    val seed = body.seed.getOrElse(options.seed)
    val width = body.width.getOrElse(options.width)
    val height = body.height.getOrElse(options.height)
    val decoded =
      try
        Right(
          (
            initImage.map(decode),
            maskImage.map(decode),
            referenceImages.map(decode)
          )
        )
      catch { case error: IllegalArgumentException => Left(error.getMessage) }
    val (init, mask, references) = decoded match {
      case Right(images) => images
      case Left(problem) =>
        return failed(400, problem)
    }
    val stretch = body.autoResizeRefImage.getOrElse(true)
    // a sampler the family does not run is not refused: it runs its own
    val sampler = sample.sampleMethod
      .filter(pipeline.samplers.contains)
      .orElse(launch.launchSampler(pipeline))
    sample.sampleMethod
      .filterNot(name =>
        pipeline.samplers.contains(name) || name == pipeline.defaultSampler
      )
      .foreach(name =>
        println(
          s"[WARN] ${pipeline.family} has no sampler '$name' (${pipeline.samplers
              .mkString(", ")}): running its own"
        )
      )
    // the request's levels, else the launch's: a list is its own step count
    val sigmas = Some(sample.sigmas)
      .filter(_.nonEmpty)
      .orElse(Some(launch.launchSigmas).filter(_.nonEmpty))
      .map(FlowSchedule.custom) match {
      case Some(Left(problem)) =>
        return failed(400, problem)
      case Some(Right(schedule)) => Some(schedule)
      case None                  => None
    }
    val request = ImageRequest(
      prompt = body.prompt.getOrElse(options.prompt),
      negativePrompt = body.negativePrompt.getOrElse(options.negativePrompt),
      width = width,
      height = height,
      steps =
        sigmas.fold(sample.sampleSteps.getOrElse(launch.steps))(_.size - 1),
      cfgScale = guidance
        .flatMap(_.txtCfg)
        .map(_.toFloat)
        .getOrElse(options.cfgScale.toFloat),
      guidance = guidance
        .flatMap(_.distilledGuidance)
        .map(_.toFloat)
        .getOrElse(
          options.guidance.getOrElse(ImageOptions.DefaultGuidance).toFloat
        ),
      seed = if (seed < 0) Random.nextLong(Long.MaxValue) else seed,
      // A request's own noise levels come without the launch's shift, where
      // the family derives one: `--flow-shift` was set for the schedule the
      // levels replace, and only a shift sent with them is meant for them.
      shift = sample.flowShift
        .orElse(
          launch.defaultShift
            .filterNot(_ => pipeline.ownShift && sample.sigmas.nonEmpty)
        ),
      loras = loras.map { lora =>
        val path = Paths.get(lora.path)
        val resolved =
          if (path.isAbsolute) path
          else options.loraDirectory.fold(path)(_.resolve(path))
        resolved -> lora.multiplier.fold(1f)(_.toFloat)
      },
      initImage = init.map(Images.resized(_, width, height)),
      strength = body.strength.fold(0.75f)(_.toFloat),
      mask = mask.map(Images.resized(_, width, height)),
      references =
        if (stretch) references.map(Images.resized(_, width, height))
        else references,
      sigmas = sigmas,
      sampler = sampler
    )
    val missing = request.loras.map(_._1).filterNot(Files.isRegularFile(_))
    if (missing.nonEmpty)
      return failed(400, s"no such LoRA: ${missing.mkString(", ")}")
    if (request.shift.exists(_ <= 0))
      return failed(400, s"a flow shift of ${request.shift.get}: above 0")
    if (request.width % 16 != 0 || request.height % 16 != 0)
      return failed(
        400,
        s"${request.width} × ${request.height}: ${pipeline.family} takes multiples of 16"
      )
    val count = body.batchCount.getOrElse(1).max(1)
    queue.enqueue(
      "img_gen",
      () => results.images(pipeline, request, count),
      previewAsked(body.preview),
      unread(sample, None, body.clipSkip, body.outputFormat, "png", Nil)
    )
  }
}
