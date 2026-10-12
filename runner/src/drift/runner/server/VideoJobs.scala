package drift.runner.server

import drift.protocol.*
import drift.protocol.JobEndpoints.Asked
import drift.runner.diffusion.*
import drift.runner.server.JobAnswers.{failed, Answer}
import drift.runner.server.JobRequests.*

import java.nio.file.{Files, Paths}
import scala.util.Random

/** `vid_gen` requests, read into jobs on `queue`. */
final private[server] class VideoJobs(
    options: ImageOptions,
    launch: LaunchDefaults,
    queue: JobQueue,
    results: JobResults
) {

  /** `POST /sdcpp/v1/vid_gen`: a job from the request over the launch flags'
    * values, its frames and sides rounded up to what the model takes; what the
    * runner cannot do yet is refused (400).
    */
  def submitVideo(
      pipeline: VideoPipeline,
      asked: Asked[VideoJobRequest]
  ): Answer = {
    val body = asked match {
      case Right(read)   => read
      case Left(problem) =>
        return failed(400, problem)
    }
    def text(value: Option[String]) = value.filter(_.nonEmpty)
    def some[A](values: Option[List[A]]) = values.getOrElse(Nil)
    val sample = body.sampleParams.getOrElse(SampleRequest.none)
    val guidance = sample.guidance
    val unsupported = Seq(
      Option.when(text(body.initImage).nonEmpty && !pipeline.takesInitImage)(
        "init images"
      ),
      Option.when(text(body.endImage).nonEmpty && !pipeline.takesEndImage)(
        "end images"
      ),
      Option.when(some(body.controlFrames).nonEmpty)("control frames"),
      Option.when(
        some(body.references).nonEmpty && !pipeline.takesReferences
      )("reference media"),
      Option.when(some(body.guides).nonEmpty && !pipeline.takesGuides)(
        "guides"
      ),
      Option.when(
        (text(body.controlVideo).nonEmpty ||
          text(body.controlMask).nonEmpty) &&
          !pipeline.takesControl
      )("control videos"),
      Option.when(some(body.lora).nonEmpty && !pipeline.takesLoras)("LoRAs"),
      Option.when(sample.sigmas.nonEmpty)("custom sigmas"),
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
    val highNoise = body.highNoiseSampleParams
    val width = body.width.getOrElse(options.width)
    val height = body.height.getOrElse(options.height)
    def frames(media: Media): Seq[java.awt.image.BufferedImage] = media match {
      case Media.Still(image)       => Seq(image)
      case Media.Clip(frames, _, _) => frames
      case Media.Sound(_)           =>
        throw new IllegalArgumentException("a sound where frames are expected")
    }
    val inputs =
      try
        Right {
          // a mask alone (with its source) inpaints without a control video
          val control = text(body.controlVideo)
            .map(media)
            .orElse(
              text(body.controlMask)
                .map(_ => Media.Clip(Nil, pipeline.fps, None))
            )
          (
            text(body.initImage).map(decode),
            text(body.endImage).map(decode),
            some(body.references).filter(_.nonEmpty).map(media),
            some(body.guides).map { guide =>
              VideoGuide(media(guide.media), guide.frameIndex.getOrElse(0))
            },
            control.map { video =>
              VideoControl(
                frames(video),
                video match {
                  case Media.Clip(_, fps, _) => fps
                  case _                     => pipeline.fps.toDouble
                },
                body.controlStrength.fold(1f)(_.toFloat),
                body.controlStart.fold(0f)(_.toFloat),
                body.controlEnd.fold(1f)(_.toFloat),
                text(body.controlMask).map(mask => frames(media(mask))),
                text(body.sourceVideo).map(source => frames(media(source)))
              )
            }
          )
        }
      catch { case error: IllegalArgumentException => Left(error.getMessage) }
    val (initImage, endImage, references, guides, control) = inputs match {
      case Right(decoded) => decoded
      case Left(problem)  =>
        return failed(400, problem)
    }
    def rounded(side: Int) =
      (side + pipeline.sizeMultiple - 1) / pipeline.sizeMultiple * pipeline.sizeMultiple
    val request = VideoRequest(
      prompt = body.prompt.getOrElse(options.prompt),
      negativePrompt = body.negativePrompt.getOrElse(options.negativePrompt),
      width = width,
      height = height,
      frames = body.videoFrames.getOrElse(options.videoFrames),
      steps = sample.sampleSteps.getOrElse(launch.steps),
      cfgScale = guidance
        .flatMap(_.txtCfg)
        .map(_.toFloat)
        .getOrElse(options.cfgScale.toFloat),
      seed = if (seed < 0) Random.nextLong(Long.MaxValue) else seed,
      shift = sample.flowShift
        .orElse(options.flowShift)
        .getOrElse(FlowSchedule.DefaultShift),
      highNoiseSteps = highNoise
        .flatMap(_.sampleSteps)
        .filter(_ >= 0)
        .orElse(options.highNoiseSteps),
      highNoiseCfgScale = highNoise
        .flatMap(_.guidance)
        .flatMap(_.txtCfg)
        .map(_.toFloat)
        .orElse(options.highNoiseCfgScale.map(_.toFloat)),
      moeBoundary = body.moeBoundary
        .map(_.toFloat)
        .getOrElse(options.moeBoundary.toFloat),
      fps = body.fps.orElse(options.fps),
      audioCfgScale = body.audioCfgScale.map(_.toFloat),
      ancestral = sample.sampleMethod.contains("euler_a"),
      modalityScale = body.modalityScale.map(_.toFloat).getOrElse(1f),
      audioModalityScale = body.audioModalityScale.map(_.toFloat).getOrElse(1f),
      initImage = initImage.map(
        pipeline.onCanvas(_, rounded(width), rounded(height), follower = false)
      ),
      endImage = endImage.map(
        pipeline.onCanvas(
          _,
          rounded(width),
          rounded(height),
          follower = initImage.nonEmpty
        )
      ),
      loras = some(body.lora).map { lora =>
        val path = Paths.get(lora.path)
        VideoLora(
          if (path.isAbsolute) path
          else options.loraDirectory.fold(path)(_.resolve(path)),
          lora.multiplier.fold(1f)(_.toFloat),
          lora.isHighNoise.contains(true)
        )
      },
      references = references,
      guides = guides,
      control = control
    )
    val missing = request.loras.map(_.path).filterNot(Files.isRegularFile(_))
    if (missing.nonEmpty)
      return failed(400, s"no such LoRA: ${missing.mkString(", ")}")
    queue.enqueue(
      "vid_gen",
      () => results.video(pipeline, request),
      previewAsked(body.preview),
      unread(
        sample,
        highNoise,
        body.clipSkip,
        body.outputFormat,
        "webm",
        Seq("vace_strength" -> body.vaceStrength.nonEmpty)
      )
    )
  }
}
