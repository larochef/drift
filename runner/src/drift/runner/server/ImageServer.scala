package drift.runner.server

import drift.protocol.*
import drift.protocol.JobEndpoints.Asked
import drift.runner.diffusion.*
import drift.runner.lora.LoraReport

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.file.*
import java.util.{Base64, UUID}
import java.util.concurrent.*
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Random
import scala.util.control.NonFatal

import ox.*
import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.NettySyncServer

/** `sd-server`'s native API (`/sdcpp/v1`) on the runner's image or video
  * pipeline, as drift drives it (`specs/42`, steps 12 and 14): the capabilities
  * document (the form's defaults, from the launch flags), `img_gen` or
  * `vid_gen` jobs queued and run one at a time, polled, and cancelled — while
  * queued, or between two steps once generating. A request starts from the
  * launch flags' values and overrides what it sends. Progress goes to the log
  * as sd-cpp prints it (`| i/n - Xs/it`), which drift parses; an image or video
  * job whose request asks for it (`preview`, sd-server's field) also carries
  * the last step's estimate of its picture, or of its video as a picture that
  * moves (`specs/54-step-preview.md`), and any job what it went on without
  * (`warnings`: a LoRA left unapplied). Images come as base64 or data URLs;
  * references are stretched to the output's size first unless
  * `auto_resize_ref_image` is false, as sd-server does. A generated video comes
  * back as one webm (`VideoFiles`); an upscaled one is written to the file its
  * request names (`VideoUpscale`).
  */
final class ImageServer(
    options: ImageOptions,
    pipeline: ImagePipeline | VideoPipeline | SeedVr2Pipeline,
    ops: drift.runner.ops.Ops
) {

  /** One job of `kind`: `run` makes its result document. */
  final private class Job(
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
  }

  /** What a step throws once its job is asked to stop: the generation unwinds
    * the way a failed one does, and the job ends `cancelled`.
    */
  final private class JobCancelled extends RuntimeException("cancelled")

  /** What a job throws when `reason` ended it with part of its work done and
    * kept: the job ends `cancelled` or `failed` as `reason` says, and carries
    * `result` all the same (a video's upscale, `VideoUpscale`).
    */
  final private class PartialResult(
      val result: NativeJobResult,
      reason: Throwable
  ) extends RuntimeException(reason)

  /** The job the worker is generating, if any. */
  @volatile private var generating: Option[Job] = None

  private val mode = pipeline match {
    case _: ImagePipeline   => "img_gen"
    case _: VideoPipeline   => "vid_gen"
    case _: SeedVr2Pipeline => "upscale"
  }

  /** The steps a request takes when neither it nor the flags say: 4 for the
    * turbo image models the runner draws, sd-cpp's 20 for a video.
    */
  private val steps = options.steps.getOrElse(pipeline match {
    case _: ImagePipeline   => 4
    case _: VideoPipeline   => 20
    case _: SeedVr2Pipeline => 1
  })

  private val family = pipeline match {
    case image: ImagePipeline      => image.family
    case video: VideoPipeline      => video.family
    case upscaler: SeedVr2Pipeline => upscaler.family
  }

  private val jobs = new ConcurrentHashMap[String, Job]()
  private val queue = new LinkedBlockingQueue[Job]()

  /** An answer to a job's submission, poll or cancel: its document, or the
    * status and reason it is refused with.
    */
  private type Answer = Either[(StatusCode, JobError), NativeJob]

  private def failed(status: Int, reason: String): Answer =
    Left(StatusCode(status) -> JobError(reason))

  /** Whatever a request throws is the answer's reason, not a silent 500. */
  private def answered(answer: => Answer): Answer =
    try answer
    catch {
      case NonFatal(error) => failed(500, String.valueOf(error.getMessage))
    }

  private val endpoints: List[ServerEndpoint[Any, Identity]] = List(
    JobEndpoints.capabilities
      .serverLogicSuccess[Identity](_ => ujson.write(capabilities)),
    JobEndpoints.submitImage.serverLogic[Identity](asked =>
      answered(pipeline match {
        case image: ImagePipeline => submit(image, asked)
        case _                    => wrongMode
      })
    ),
    JobEndpoints.submitVideo.serverLogic[Identity](asked =>
      answered(pipeline match {
        case video: VideoPipeline => submitVideo(video, asked)
        case _                    => wrongMode
      })
    ),
    JobEndpoints.submitUpscale.serverLogic[Identity](asked =>
      answered(pipeline match {
        case upscaler: SeedVr2Pipeline => submitUpscale(upscaler, asked)
        case _                         => wrongMode
      })
    ),
    JobEndpoints.job.serverLogic[Identity](id => answered(polled(id))),
    JobEndpoints.cancel.serverLogic[Identity](id => answered(cancelled(id)))
  )

  /** Serves until the process ends: the jobs' worker and the server are forks
    * of one scope, so neither outlives the other.
    */
  def serve(): Unit = supervised {
    forkUser(work()).discard
    NettySyncServer()
      .addEndpoints(endpoints)
      .host(options.host)
      .port(options.port)
      .start()
      .discard
    println(s"listening on http://${options.host}:${options.port}")
    never
  }

  /** The LoRA files under `--lora-model-dir`, relative to it. */
  private def loraFiles: Seq[Path] =
    options.loraDirectory.filter(Files.isDirectory(_)).toSeq.flatMap {
      directory =>
        val walk = Files.walk(directory)
        try
          walk.iterator.asScala
            .filter(p =>
              Files.isRegularFile(p) && p.toString.endsWith(".safetensors")
            )
            .map(directory.relativize)
            .toSeq
            .sortBy(_.toString)
        finally walk.close()
    }

  private def wrongMode: Answer =
    failed(400, s"$family generates with $mode only")

  /** The flow shift a request starts from: the launch's, else sd-server's
    * default — or none for a family that derives its own.
    */
  private val defaultShift: Option[Double] = options.flowShift.orElse(
    pipeline match {
      case image: ImagePipeline if image.ownShift => None
      case _ => Some(FlowSchedule.DefaultShift)
    }
  )

  /** `--sigmas`, for a model that takes them. */
  private val launchSigmas: Seq[Double] = pipeline match {
    case image: ImagePipeline if image.takesSigmas => options.sigmas
    case _                                         => Nil
  }

  /** `--sampling-method`, when it is one of the model's samplers. */
  private def launchSampler(pipeline: ImagePipeline): Option[String] =
    options.sampler.filter(pipeline.samplers.contains)

  /** A request's `sample_params.custom_sigmas`. */
  /** What a request sets that the runner does not read: said in the job's
    * warnings rather than dropped. A field at the value that means "nothing
    * asked" is not one.
    */
  private def unread(
      sample: SampleRequest,
      highNoise: Option[SampleRequest],
      clipSkip: Option[Int],
      outputFormat: Option[String],
      written: String,
      others: Seq[(String, Boolean)]
  ): Seq[String] = {
    val samples = sample +: highNoise.toSeq
    val set = Seq(
      "scheduler" -> samples.exists(
        _.scheduler.exists(name => name.nonEmpty && name != "default")
      ),
      "eta" -> samples.exists(_.eta.exists(_ != 0)),
      "shifted_timestep" -> samples.exists(_.shiftedTimestep.exists(_ != 0)),
      "img_cfg" -> samples.exists(_.guidance.exists(_.imgCfg.nonEmpty)),
      "slg" -> samples.exists(
        _.guidance.exists(_.slg.exists(_.scale.exists(_ != 0)))
      ),
      "clip_skip" -> clipSkip.exists(_ > 0),
      "output_format" -> outputFormat.exists(format =>
        format.nonEmpty && format != written
      )
    ) ++ others
    val names = set.collect { case (name, true) => s"`$name`" }
    Option
      .when(names.nonEmpty)(
        s"the drift runner does not read ${names.mkString(", ")}: " +
          (if (names.size == 1) "it was" else "they were") + " ignored"
      )
      .toSeq
  }

  private def capabilities: ujson.Value = {
    val name = options.diffusionModel.getFileName.toString
    val sampleParams = ujson.Obj(
      "scheduler" -> "default",
      "sample_method" -> (pipeline match {
        case image: ImagePipeline =>
          launchSampler(image).getOrElse(image.defaultSampler)
        case _ => "euler"
      }),
      "sample_steps" -> steps,
      "eta" -> 0.0,
      "shifted_timestep" -> 0,
      "custom_sigmas" -> ujson.Arr.from(launchSigmas),
      "flow_shift" -> defaultShift.fold[ujson.Value](ujson.Null)(ujson.Num(_)),
      "guidance" -> ujson.Obj(
        "txt_cfg" -> options.cfgScale,
        "img_cfg" -> options.cfgScale,
        "distilled_guidance" -> options.guidance.getOrElse(
          ImageOptions.DefaultGuidance
        )
      )
    )
    val (defaults, features, formats) = pipeline match {
      case image: ImagePipeline =>
        (imageDefaults(sampleParams), imageFeatures(image), ujson.Arr("png"))
      case video: VideoPipeline =>
        val videoDefaults = ujson.Obj(
          "prompt" -> options.prompt,
          "negative_prompt" -> options.negativePrompt,
          "clip_skip" -> -1,
          "width" -> options.width,
          "height" -> options.height,
          "strength" -> 0.75,
          "seed" -> ujson.Num(options.seed.toDouble),
          "video_frames" -> video.alignedFrames(options.videoFrames),
          "fps" -> options.fps.getOrElse(video.fps),
          "sample_params" -> sampleParams,
          "output_format" -> "webm",
          "output_compression" -> 100
        )
        // a second expert announces its own steps and CFG, as sd-cpp does:
        // drift's form only offers them when the defaults carry them
        if (video.hasHighNoiseExpert) {
          val highNoise = ujson.copy(sampleParams)
          highNoise("sample_steps") =
            ujson.Num(options.highNoiseSteps.getOrElse(-1).toDouble)
          highNoise("guidance")("txt_cfg") =
            ujson.Num(options.highNoiseCfgScale.getOrElse(options.cfgScale))
          videoDefaults("high_noise_sample_params") = highNoise
        }
        (
          videoDefaults,
          ujson.Obj(
            "init_image" -> video.takesInitImage,
            "end_image" -> video.takesEndImage,
            "control_frames" -> false,
            "references" -> video.takesReferences,
            "guides" -> video.takesGuides,
            "control_video" -> video.takesControl,
            "lora" -> (video.takesLoras && options.loraDirectory.isDefined),
            "vae_tiling" -> false,
            "cancel_queued" -> true,
            "cancel_generating" -> true,
            "preview" -> true
          ),
          ujson.Arr("webm")
        )
      case _: SeedVr2Pipeline =>
        (
          ujson.Obj(
            "scale" -> SeedVr2Pipeline.DefaultScale,
            "seed" -> ujson.Num(options.seed.toDouble),
            "batch" -> SeedVr2Options.Default.batch,
            "overlap" -> SeedVr2Options.Default.overlap
          ),
          ujson.Obj("cancel_queued" -> true, "cancel_generating" -> true),
          ujson.Arr("png", "webm")
        )
    }
    ujson.Obj(
      // which protocol this server speaks, for the drift that drives it
      "drift" -> ujson.Obj("protocol" -> JobRoutes.Version),
      "model" -> ujson.Obj(
        "name" -> name,
        "stem" -> name.stripSuffix(".safetensors").stripSuffix(".gguf"),
        "path" -> options.diffusionModel.toString
      ),
      "current_mode" -> mode,
      "supported_modes" -> ujson.Arr(mode),
      "defaults_by_mode" -> ujson.Obj(mode -> defaults),
      "features_by_mode" -> ujson.Obj(mode -> features),
      "output_formats_by_mode" -> ujson.Obj(mode -> formats),
      "preview_modes" -> (pipeline match {
        case _: SeedVr2Pipeline => ujson.Arr("none")
        case _                  => ujson.Arr("none", "proj")
      }),
      "samplers" -> ujson.Arr.from(pipeline match {
        case image: ImagePipeline => image.samplers
        case _                    => Seq("euler")
      }),
      "schedulers" -> ujson.Arr("default"),
      "loras" -> ujson.Arr.from(loraFiles.map { path =>
        ujson.Obj(
          "name" -> path.getFileName.toString.stripSuffix(".safetensors"),
          "path" -> path.toString
        )
      }),
      "upscalers" -> ujson.Arr(),
      "limits" -> ujson.Obj(
        "min_width" -> 64,
        "max_width" -> 4096,
        "min_height" -> 64,
        "max_height" -> 4096,
        "max_batch_count" -> 8,
        "max_queue_size" -> 64
      )
    )
  }

  private def imageDefaults(sampleParams: ujson.Obj): ujson.Obj = {
    ujson.Obj(
      "prompt" -> options.prompt,
      "negative_prompt" -> options.negativePrompt,
      "clip_skip" -> -1,
      "width" -> options.width,
      "height" -> options.height,
      "strength" -> 0.75,
      "seed" -> ujson.Num(options.seed.toDouble),
      "batch_count" -> 1,
      "sample_params" -> sampleParams,
      "output_format" -> "png",
      "output_compression" -> 100
    )
  }

  private def imageFeatures(pipeline: ImagePipeline): ujson.Obj =
    ujson.Obj(
      "init_image" -> pipeline.takesInitImage,
      "mask_image" -> pipeline.takesMask,
      "ref_images" -> pipeline.takesReferences,
      "lora" -> (pipeline.takesLoras && options.loraDirectory.isDefined),
      "hires" -> false,
      "vae_tiling" -> false,
      "cancel_queued" -> true,
      "cancel_generating" -> true,
      "preview" -> true
    )

  /** `POST /sdcpp/v1/img_gen`: a job from the request over the launch flags'
    * values; what the runner cannot do yet is refused (400).
    */
  private def submit(
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
      .orElse(launchSampler(pipeline))
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
      .orElse(Some(launchSigmas).filter(_.nonEmpty))
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
      steps = sigmas.fold(sample.sampleSteps.getOrElse(steps))(_.size - 1),
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
          defaultShift
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
    enqueue(
      "img_gen",
      () => images(pipeline, request, count),
      previewAsked(body.preview),
      unread(sample, None, body.clipSkip, body.outputFormat, "png", Nil)
    )
  }

  /** Whether a request asks for step previews: sd-server's `preview`, a mode
    * (`proj`, `tae`, `vae`) or `none`. The runner has one way to draw one,
    * whatever the mode asked.
    */
  private def previewAsked(preview: Option[String]): Boolean =
    preview.exists(_ != "none")

  /** `POST /sdcpp/v1/vid_gen`: a job from the request over the launch flags'
    * values, its frames and sides rounded up to what the model takes; what the
    * runner cannot do yet is refused (400).
    */
  private def submitVideo(
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
      steps = sample.sampleSteps.getOrElse(steps),
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
    enqueue(
      "vid_gen",
      () => video(pipeline, request),
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

  /** `POST /sdcpp/v1/upscale`, the runner's own job, restored at
    * `width × height`, or at `scale` times the source's size (4 unless said);
    * `seed`, `batch` and `overlap` as `SeedVr2Options` has them.
    *   - **A picture** is `source`, base64 or a data URL, and comes back as
    *     `img_gen`'s result does.
    *   - **A video** is a file, `source_file`, and is written to another,
    *     `output_file` (a webm, its frame rate and soundtrack the source's):
    *     drift and the runner share the machine, and a video of any length goes
    *     through without being held (`VideoUpscale`). The result names the file
    *     (`file`), its frames (`frame_count`, of `source_frames`) and its rate
    *     (`frame_rate`, exact; `fps`, rounded). A job stopped or failed after
    *     its first frames ends `cancelled` or `failed` with that result all the
    *     same: the file is the video of the frames made.
    */
  private def submitUpscale(
      pipeline: SeedVr2Pipeline,
      asked: Asked[UpscaleJobRequest]
  ): Answer = {
    val body = asked match {
      case Right(read)   => read
      case Left(problem) =>
        return failed(400, problem)
    }
    def text(value: Option[String]) = value.filter(_.nonEmpty)
    val scale = body.scale.getOrElse(SeedVr2Pipeline.DefaultScale)
    def even(side: Int) = side / 2 * 2
    def target(width: Int, height: Int) = (
      even(body.width.getOrElse(width * scale)),
      even(body.height.getOrElse(height * scale))
    )
    val seed = body.seed
      .filter(_ >= 0)
      .getOrElse(
        if (options.seed >= 0) options.seed
        else Random.nextInt(Int.MaxValue).toLong
      )
    val upscaleOptions =
      try
        SeedVr2Options.Default.copy(
          batch = body.batch.getOrElse(SeedVr2Options.Default.batch),
          overlap = body.overlap.getOrElse(SeedVr2Options.Default.overlap)
        )
      catch {
        case error: IllegalArgumentException =>
          return failed(400, error.getMessage)
      }
    (text(body.source), text(body.sourceFile), text(body.outputFile)) match {
      case (Some(encoded), None, None) =>
        submitPictureUpscale(
          pipeline,
          encoded,
          target,
          seed,
          upscaleOptions
        )
      case (None, Some(source), Some(output)) =>
        submitVideoUpscale(
          pipeline,
          Paths.get(source),
          Paths.get(output),
          target,
          seed,
          upscaleOptions
        )
      case _ =>
        failed(
          400,
          "a picture as `source`, or a video as `source_file` and `output_file`"
        )
    }
  }

  /** A video's `upscale` job, from a file into a file (`VideoUpscale`). */
  private def submitVideoUpscale(
      pipeline: SeedVr2Pipeline,
      source: Path,
      output: Path,
      target: (Int, Int) => (Int, Int),
      seed: Long,
      upscaleOptions: SeedVr2Options
  ): Answer = {
    if (!Files.isReadable(source))
      return failed(400, s"no readable file at $source")
    if (!Option(output.getParent).exists(Files.isDirectory(_)))
      return failed(400, s"no folder to write $output in")
    val pass = new UpscalePass {
      def steps(frames: Int, width: Int, height: Int): Int =
        pipeline.steps(frames, width, height, upscaleOptions)
      def upscaled(
          frames: Iterator[java.awt.image.BufferedImage],
          count: Int,
          width: Int,
          height: Int,
          tick: () => Unit
      ): Iterator[java.awt.image.BufferedImage] =
        pipeline.upscaled(
          frames,
          count,
          width,
          height,
          seed,
          upscaleOptions,
          tick
        )
    }
    enqueue(
      "upscale",
      () => {
        println("sampling using Euler method")
        val startedAt = System.nanoTime()
        val made = VideoUpscale.run(
          source,
          output,
          target,
          pass,
          stepPrinter(),
          (clip, size, passes) =>
            println(
              s"upscaling ${clip.count} frames of ${clip.width} × ${clip.height} to ${size._1} × ${size._2}${
                  if (passes > 1) s" in $passes passes" else ""
                } (seed $seed)"
            )
        )
        val result = NativeJobResult(
          outputFormat = "webm",
          mimeType = Some("video/webm"),
          fps = Some(made.rate.rounded),
          frameRate = Some(made.rate.toString),
          frameCount = Some(made.frames),
          sourceFrames = Some(made.sourceFrames),
          file = Some(output.toString)
        )
        made.stopped match {
          case Some(reason) =>
            println(
              s"stopped after ${made.frames} frames: the video of those is kept"
            )
            throw new PartialResult(result, reason)
          case None =>
            println(
              f"generate_image completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
            )
            result
        }
      }
    )
  }

  /** A picture's `upscale` job: `encoded` is base64 or a data URL. */
  private def submitPictureUpscale(
      pipeline: SeedVr2Pipeline,
      encoded: String,
      target: (Int, Int) => (Int, Int),
      seed: Long,
      upscaleOptions: SeedVr2Options
  ): Answer = {
    val bytes = Base64.getMimeDecoder.decode(
      if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
      else encoded
    )
    // a picture as it is: `VideoFiles` would bring a large one down
    val frames = Option(ImageIO.read(new ByteArrayInputStream(bytes)))
      .fold(VideoFiles.decode(bytes))(Media.Still(_)) match {
      case Media.Still(image) => Seq(image)
      case _                  =>
        return failed(
          400,
          "`source` is a picture: a video goes as `source_file` and `output_file`"
        )
    }
    val (width, height) = target(frames.head.getWidth, frames.head.getHeight)
    enqueue(
      "upscale",
      () => {
        println("sampling using Euler method")
        println(
          s"upscaling 1 frame of ${frames.head.getWidth} × ${frames.head.getHeight} to $width × $height (seed $seed)"
        )
        val startedAt = System.nanoTime()
        val sizes = SeedVr2Pipeline.passes(
          (frames.head.getWidth, frames.head.getHeight),
          (width, height)
        )
        val restored = sizes.zipWithIndex.foldLeft(frames) {
          case (current, ((passWidth, passHeight), index)) =>
            if (sizes.size > 1)
              println(
                s"pass ${index + 1} of ${sizes.size}: to $passWidth × $passHeight"
              )
            pipeline.upscale(
              current,
              passWidth,
              passHeight,
              seed,
              upscaleOptions,
              stepPrinter()
            )
        }
        println(
          f"generate_image completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
        )
        val png = new ByteArrayOutputStream()
        ImageIO.write(restored.head, "png", png)
        NativeJobResult(
          outputFormat = "png",
          images = List(
            NativeJobImage(0, Base64.getEncoder.encodeToString(png.toByteArray))
          )
        )
      }
    )
  }

  private def enqueue(
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

  /** An image sent as base64 or a data URL. */
  private def decode(encoded: String): java.awt.image.BufferedImage = {
    val data =
      if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
      else encoded
    Option(
      ImageIO.read(new ByteArrayInputStream(Base64.getMimeDecoder.decode(data)))
    ).getOrElse(
      throw new IllegalArgumentException("an image that does not decode")
    )
  }

  /** A media input (base64 or a data URL), decoded. */
  private def media(encoded: String): Media =
    VideoFiles.decode(
      Base64.getMimeDecoder.decode(
        if (encoded.startsWith("data:")) encoded.drop(encoded.indexOf(',') + 1)
        else encoded
      )
    )

  /** `GET /sdcpp/v1/jobs/{id}` and `POST /sdcpp/v1/jobs/{id}/cancel`. */
  /** `GET /sdcpp/v1/jobs/{id}`: the job, with its preview while it samples. */
  private def polled(id: String): Answer =
    Option(jobs.get(id)) match {
      case None      => failed(404, "no such job")
      case Some(job) => Right(describe(job).copy(preview = job.preview))
    }

  /** `POST /sdcpp/v1/jobs/{id}/cancel`. */
  private def cancelled(id: String): Answer =
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

  /** A step's progress line, as sd-cpp prints it: step 0 is the empty bar a
    * pass starts with, which says the sampling has begun (and how many steps it
    * is) while the first step, minutes long on a large video, is under way.
    */
  private def stepPrinter(): (Int, Int) => Unit = (new StepBar).step

  /** A video's last stage, after the pipeline's own. */
  private val VideoFileStage = "writing the file"

  /** A pass's bar: a line a step, and while a step lasts, the same line redrawn
    * with the fraction of it done (`3.42/8`, at the last step's rate, or during
    * the first step at the rate it is heading for). The fraction is the
    * pipeline's own when it tells one (`within`), else the work the backend was
    * given since the step began over what the last whole step took (`work`):
    * any model's, from its second step on.
    */
  final private class StepBar {
    private var last = System.nanoTime()
    private var drawn = last
    private var at = (0, 0)
    private var rate = 0.0
    // the backend's work since the step began, and the last whole step's
    private var worked = 0L
    private var perStep = 0L
    private var told = false
    private var shown = 0.0

    def step(step: Int, steps: Int): Unit = {
      // Between two steps is where a generation can be left: every pipeline
      // reports them here, and none holds a kernel across the call.
      if (generating.exists(_.cancelRequested)) throw new JobCancelled
      if (steps > 0) {
        val now = System.nanoTime()
        val filled = 50 * step / steps
        // a call may count several steps at once (a pass on more pixels)
        rate =
          if (step == 0) 0.0
          else (now - last) / 1e9 / math.max(1, step - at._1)
        println(
          f"  |${"=" * filled}${" " * (50 - filled)}| $step/$steps - $rate%.2fs/it"
        )
        last = now
        drawn = now
        at = (step, steps)
        if (step > 0) perStep = worked
        worked = 0
        shown = 0.0
      }
    }

    /** The fraction done of the step under way, as the pipeline tells it. */
    def within(fraction: Double): Unit = {
      told = true
      draw(fraction)
    }

    /** A piece of work given to the backend (`Ops.onWork`). */
    def work(): Unit = {
      worked += 1
      if (!told && perStep > 0 && (worked & 15) == 0)
        draw(math.min(0.99, worked.toDouble / perStep))
    }

    /** Redraws the bar at `fraction` of the step under way, a second apart. */
    private def draw(fraction: Double): Unit = {
      val (step, steps) = at
      val now = System.nanoTime()
      if (
        step < steps && fraction > shown && fraction < 1 && now - drawn >= 1e9
      ) {
        shown = fraction
        val done = step + fraction
        val filled = (50 * done / steps).toInt
        val seconds = if (rate > 0) rate else (now - last) / 1e9 / fraction
        println(
          f"  |${"=" * filled}${" " * (50 - filled)}| $done%.2f/$steps - $seconds%.2fs/it"
        )
        drawn = now
      }
    }
  }

  /** An `img_gen` job's result: `count` images of `request`, seeds in sequence.
    */
  private def images(
      pipeline: ImagePipeline,
      request: ImageRequest,
      count: Int
  ): NativeJobResult = {
    println("sampling using Euler method")
    val startedAt = System.nanoTime()
    val pngs = (0 until count).map { index =>
      val seeded = request.copy(seed = request.seed + index)
      println(s"generating image ${index + 1}/$count (seed ${seeded.seed})")
      val bar = new StepBar
      ops.onWork = bar.work
      // the picture being made has no estimate yet, whatever the last one had
      generating.foreach(_.preview = None)
      val publish = previews()
      pipeline.preview = (step, steps, picture) =>
        publish(
          index + 1,
          step,
          steps,
          () => {
            val jpeg = new ByteArrayOutputStream()
            ImageIO.write(picture(), "jpg", jpeg)
            jpeg.toByteArray
          }
        )
      val image =
        try pipeline.generate(seeded, bar.step)
        finally {
          ops.onWork = () => ()
          pipeline.preview = (_, _, _) => ()
          generating.foreach(_.preview = None)
        }
      val png = new ByteArrayOutputStream()
      ImageIO.write(image, "png", png)
      png.toByteArray
    }
    // sd-cpp's closing line, which is what drift ends the batch count on.
    val seconds = (System.nanoTime() - startedAt) / 1e9
    println(f"generate_image completed in $seconds%.2fs")
    NativeJobResult(
      outputFormat = "png",
      images = pngs.zipWithIndex.map { (png, index) =>
        NativeJobImage(index, Base64.getEncoder.encodeToString(png))
      }.toList
    )
  }

  /** The least time between two previews of a picture. */
  private val PreviewIntervalNanos = 1_000_000_000L

  /** What a pipeline's `preview` hook hands a step's preview to, for one
    * picture or video: `publish(pass, step, steps, picture)` puts it in the
    * document of the job under way when its request asked, as sd-server gives
    * one — the sampling pass from 1 (an image of the batch, a video's expert),
    * the step it is the estimate of, the pass' steps and the picture's file, a
    * JPEG or a GIF here, a PNG there. One a second at most, and a pass' last
    * step always: a poll comes once a second, and steps shorter than that would
    * pay for pictures nobody sees.
    */
  private def previews(): (Int, Int, Int, () => Array[Byte]) => Unit = {
    var drawnAt = 0L
    (pass, step, steps, picture) =>
      generating.filter(_.previewWanted).foreach { job =>
        val now = System.nanoTime()
        if (step == steps || now - drawnAt >= PreviewIntervalNanos) {
          val file = picture()
          job.preview = Some(
            NativeJobPreview(
              pass,
              step,
              steps,
              Base64.getEncoder.encodeToString(file)
            )
          )
          drawnAt = now
        }
      }
  }

  /** A `vid_gen` job's result: one webm. */
  private def video(
      pipeline: VideoPipeline,
      request: VideoRequest
  ): NativeJobResult = {
    println("sampling using Euler method")
    println(
      s"generating ${pipeline.alignedFrames(request.frames)} frames of ${request.width} × ${request.height} (seed ${request.seed})"
    )
    val startedAt = System.nanoTime()
    // the pipeline's stages and the file's, each announced as it starts
    val stages = pipeline.stages(request) :+ VideoFileStage
    val enter = (name: String) => {
      val index = stages.indexOf(name)
      if (index >= 0) println(s"stage ${index + 1}/${stages.size}: $name")
      // the estimate stays from a sampling pass to the next, and leaves when
      // the sampling is over: the decode shows none
      if (!VideoPipeline.samples(name)) generating.foreach(_.preview = None)
    }
    val bar = new StepBar
    val publish = previews()
    pipeline.stage = enter
    pipeline.withinStep = bar.within
    pipeline.preview = (pass, step, steps, clip) =>
      publish(pass, step, steps, () => PreviewGif.encode(clip()))
    ops.onWork = bar.work
    val video =
      try pipeline.generate(request, bar.step)
      finally {
        pipeline.stage = _ => ()
        pipeline.withinStep = _ => ()
        pipeline.preview = (_, _, _, _) => ()
        ops.onWork = () => ()
        generating.foreach(_.preview = None)
      }
    enter(VideoFileStage)
    val webm = VideoFiles.webm(video)
    println(
      f"generate_video completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
    )
    NativeJobResult(
      outputFormat = "webm",
      mimeType = Some("video/webm"),
      fps = Some(video.fps),
      frameCount = Some(video.frames.size),
      b64Json = Some(Base64.getEncoder.encodeToString(webm))
    )
  }

  /** The worker: runs the queue in order. */
  private def work(): Unit =
    while (true) {
      val job = queue.take()
      job.status = "generating"
      job.started = Some(System.currentTimeMillis() / 1000)
      generating = Some(job)
      LoraReport.listener = warning => job.warnings.add(warning)
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
      generating = None
      job.completed = Some(System.currentTimeMillis() / 1000)
    }
}
