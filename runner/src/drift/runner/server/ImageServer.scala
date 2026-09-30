package drift.runner.server

import drift.runner.diffusion.*

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.{Base64, UUID}
import java.util.concurrent.*
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Random

import com.sun.net.httpserver.{HttpExchange, HttpServer}

/** `sd-server`'s native API (`/sdcpp/v1`) on the runner's image or video
  * pipeline, as drift drives it (`specs/42`, steps 12 and 14): the capabilities
  * document (the form's defaults, from the launch flags), `img_gen` or
  * `vid_gen` jobs queued and run one at a time, polled, and cancelled while
  * queued. A request starts from the launch flags' values and overrides what it
  * sends. Progress goes to the log as sd-cpp prints it (`| i/n - Xs/it`), which
  * drift parses. Images come as base64 or data URLs; references are stretched
  * to the output's size first unless `auto_resize_ref_image` is false, as
  * sd-server does. A video comes back as one webm (`VideoFiles`).
  */
final class ImageServer(
    options: ImageOptions,
    pipeline: ImagePipeline | VideoPipeline
) {

  /** One job of `kind`: `run` makes its result document. */
  final private class Job(
      val id: String,
      val created: Long,
      val kind: String,
      val run: () => ujson.Value
  ) {
    @volatile var status: String = "queued"
    @volatile var started: Option[Long] = None
    @volatile var completed: Option[Long] = None
    @volatile var result: Option[ujson.Value] = None
    @volatile var error: Option[String] = None
  }

  private val mode = pipeline match {
    case _: ImagePipeline => "img_gen"
    case _: VideoPipeline => "vid_gen"
  }

  /** The steps a request takes when neither it nor the flags say: 4 for the
    * turbo image models the runner draws, sd-cpp's 20 for a video.
    */
  private val steps = options.steps.getOrElse(pipeline match {
    case _: ImagePipeline => 4
    case _: VideoPipeline => 20
  })

  private val family = pipeline match {
    case image: ImagePipeline => image.family
    case video: VideoPipeline => video.family
  }

  private val jobs = new ConcurrentHashMap[String, Job]()
  private val queue = new LinkedBlockingQueue[Job]()

  private val server =
    HttpServer.create(new InetSocketAddress(options.host, options.port), 64)
  server.setExecutor(Executors.newFixedThreadPool(4))

  def start(): Unit = {
    route("/sdcpp/v1/capabilities")(exchange =>
      json(exchange, 200, capabilities)
    )
    route("/sdcpp/v1/img_gen")(exchange =>
      pipeline match {
        case image: ImagePipeline => submit(exchange, image)
        case _                    => wrongMode(exchange)
      }
    )
    route("/sdcpp/v1/vid_gen")(exchange =>
      pipeline match {
        case video: VideoPipeline => submitVideo(exchange, video)
        case _                    => wrongMode(exchange)
      }
    )
    route("/sdcpp/v1/jobs/")(job)
    server.start()
    val worker = new Thread(() => work(), "image-worker")
    worker.setDaemon(true)
    worker.start()
  }

  private def route(path: String)(handle: HttpExchange => Unit): Unit =
    server.createContext(
      path,
      exchange =>
        try handle(exchange)
        catch {
          case error: Throwable =>
            json(
              exchange,
              500,
              ujson.Obj("error" -> String.valueOf(error.getMessage))
            )
        } finally exchange.close()
    )

  private def json(
      exchange: HttpExchange,
      status: Int,
      body: ujson.Value
  ): Unit = {
    val bytes = ujson.write(body).getBytes(StandardCharsets.UTF_8)
    exchange.getResponseHeaders.set("Content-Type", "application/json")
    exchange.sendResponseHeaders(status, bytes.length)
    exchange.getResponseBody.write(bytes)
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

  private def wrongMode(exchange: HttpExchange): Unit =
    json(
      exchange,
      400,
      ujson.Obj("error" -> s"$family generates with $mode only")
    )

  private def capabilities: ujson.Value = {
    val name = options.diffusionModel.getFileName.toString
    val sampleParams = ujson.Obj(
      "scheduler" -> "default",
      "sample_method" -> "euler",
      "sample_steps" -> steps,
      "eta" -> 0.0,
      "shifted_timestep" -> 0,
      "custom_sigmas" -> ujson.Arr(),
      "flow_shift" -> options.flowShift,
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
        (
          ujson.Obj(
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
          ),
          ujson.Obj(
            "init_image" -> video.takesInitImage,
            "end_image" -> false,
            "control_frames" -> false,
            "lora" -> false,
            "vae_tiling" -> false,
            "cancel_queued" -> true,
            "cancel_generating" -> false
          ),
          ujson.Arr("webm")
        )
    }
    ujson.Obj(
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
      "samplers" -> ujson.Arr("euler"),
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
      "mask_image" -> false,
      "ref_images" -> pipeline.takesReferences,
      "lora" -> (pipeline.takesLoras && options.loraDirectory.isDefined),
      "hires" -> false,
      "vae_tiling" -> false,
      "cancel_queued" -> true,
      "cancel_generating" -> false
    )

  /** `POST /sdcpp/v1/img_gen`: a job from the request over the launch flags'
    * values; what the runner cannot do yet is refused (400).
    */
  private def submit(exchange: HttpExchange, pipeline: ImagePipeline): Unit = {
    if (exchange.getRequestMethod != "POST")
      return json(exchange, 405, ujson.Obj("error" -> "POST only"))
    val body = ujson.read(exchange.getRequestBody.readAllBytes())
    def field(name: String) = body.obj.get(name).filterNot(_.isNull)
    val sample: collection.Map[String, ujson.Value] =
      field("sample_params").map(_.obj).getOrElse(Map.empty)
    val guidance = sample.get("guidance").filterNot(_.isNull).map(_.obj)
    val initImage = field("init_image").map(_.str).filter(_.nonEmpty)
    val referenceImages =
      field("ref_images").toSeq.flatMap(_.arr).map(_.str).filter(_.nonEmpty)
    val loras = field("lora").toSeq.flatMap(_.arr)
    val unsupported = Seq(
      Option.when(initImage.nonEmpty && !pipeline.takesInitImage)(
        "init images"
      ),
      Option.when(field("mask_image").exists(_.str.nonEmpty))("masks"),
      Option.when(referenceImages.nonEmpty && !pipeline.takesReferences)(
        "reference images"
      ),
      Option.when(loras.nonEmpty && !pipeline.takesLoras)("LoRAs"),
      Option.when(
        field("hires").exists(h => h.obj.get("enabled").exists(_.bool))
      )("hires fix"),
      Option.when(
        field("vae_tiling_params").exists(t =>
          t.obj.get("enabled").exists(_.bool)
        )
      )("VAE tiling")
    ).flatten
    if (unsupported.nonEmpty)
      return json(
        exchange,
        400,
        ujson.Obj(
          "error" -> s"not supported by the drift runner for ${pipeline.family} yet: ${unsupported.mkString(", ")}"
        )
      )
    val seed = field("seed").map(_.num.toLong).getOrElse(options.seed)
    val width = field("width").map(_.num.toInt).getOrElse(options.width)
    val height = field("height").map(_.num.toInt).getOrElse(options.height)
    val decoded =
      try Right((initImage.map(decode), referenceImages.map(decode)))
      catch { case error: IllegalArgumentException => Left(error.getMessage) }
    val (init, references) = decoded match {
      case Right(images) => images
      case Left(problem) =>
        return json(exchange, 400, ujson.Obj("error" -> problem))
    }
    val stretch = field("auto_resize_ref_image").forall(_.bool)
    val request = ImageRequest(
      prompt = field("prompt").map(_.str).getOrElse(options.prompt),
      negativePrompt =
        field("negative_prompt").map(_.str).getOrElse(options.negativePrompt),
      width = width,
      height = height,
      steps = sample.get("sample_steps").map(_.num.toInt).getOrElse(steps),
      cfgScale = guidance
        .flatMap(_.get("txt_cfg"))
        .map(_.num.toFloat)
        .getOrElse(options.cfgScale.toFloat),
      guidance = guidance
        .flatMap(_.get("distilled_guidance"))
        .filterNot(_.isNull)
        .map(_.num.toFloat)
        .getOrElse(
          options.guidance.getOrElse(ImageOptions.DefaultGuidance).toFloat
        ),
      seed = if (seed < 0) Random.nextLong(Long.MaxValue) else seed,
      shift = sample
        .get("flow_shift")
        .filterNot(_.isNull)
        .map(_.num)
        .getOrElse(options.flowShift),
      loras = loras.map { lora =>
        val path = Paths.get(lora("path").str)
        val resolved =
          if (path.isAbsolute) path
          else options.loraDirectory.fold(path)(_.resolve(path))
        resolved -> lora.obj.get("multiplier").fold(1f)(_.num.toFloat)
      },
      initImage = init.map(Images.resized(_, width, height)),
      strength = field("strength").fold(0.75f)(_.num.toFloat),
      references =
        if (stretch) references.map(Images.resized(_, width, height))
        else references
    )
    val missing = request.loras.map(_._1).filterNot(Files.isRegularFile(_))
    if (missing.nonEmpty)
      return json(
        exchange,
        400,
        ujson.Obj("error" -> s"no such LoRA: ${missing.mkString(", ")}")
      )
    if (request.width % 16 != 0 || request.height % 16 != 0)
      return json(
        exchange,
        400,
        ujson.Obj(
          "error" -> s"${request.width} × ${request.height}: ${pipeline.family} takes multiples of 16"
        )
      )
    val count = field("batch_count").map(_.num.toInt).getOrElse(1).max(1)
    enqueue(exchange, "img_gen", () => images(pipeline, request, count))
  }

  /** `POST /sdcpp/v1/vid_gen`: a job from the request over the launch flags'
    * values, its frames and sides rounded up to what the model takes; what the
    * runner cannot do yet is refused (400).
    */
  private def submitVideo(
      exchange: HttpExchange,
      pipeline: VideoPipeline
  ): Unit = {
    if (exchange.getRequestMethod != "POST")
      return json(exchange, 405, ujson.Obj("error" -> "POST only"))
    val body = ujson.read(exchange.getRequestBody.readAllBytes())
    def field(name: String) = body.obj.get(name).filterNot(_.isNull)
    def present(name: String) =
      field(name).exists(value =>
        value.strOpt.exists(_.nonEmpty) || value.arrOpt.exists(_.nonEmpty)
      )
    val sample: collection.Map[String, ujson.Value] =
      field("sample_params").map(_.obj).getOrElse(Map.empty)
    val guidance = sample.get("guidance").filterNot(_.isNull).map(_.obj)
    val unsupported = Seq(
      Option.when(present("init_image") && !pipeline.takesInitImage)(
        "init images"
      ),
      Option.when(present("end_image"))("end images"),
      Option.when(present("control_frames"))("control frames"),
      Option.when(present("lora"))("LoRAs"),
      Option.when(
        field("vae_tiling_params").exists(t =>
          t.obj.get("enabled").exists(_.bool)
        )
      )("VAE tiling")
    ).flatten
    if (unsupported.nonEmpty)
      return json(
        exchange,
        400,
        ujson.Obj(
          "error" -> s"not supported by the drift runner for ${pipeline.family} yet: ${unsupported.mkString(", ")}"
        )
      )
    val seed = field("seed").map(_.num.toLong).getOrElse(options.seed)
    val highNoise: collection.Map[String, ujson.Value] =
      field("high_noise_sample_params").map(_.obj).getOrElse(Map.empty)
    val width = field("width").map(_.num.toInt).getOrElse(options.width)
    val height = field("height").map(_.num.toInt).getOrElse(options.height)
    val init =
      try Right(field("init_image").map(_.str).filter(_.nonEmpty).map(decode))
      catch { case error: IllegalArgumentException => Left(error.getMessage) }
    val initImage = init match {
      case Right(image)  => image
      case Left(problem) =>
        return json(exchange, 400, ujson.Obj("error" -> problem))
    }
    val request = VideoRequest(
      prompt = field("prompt").map(_.str).getOrElse(options.prompt),
      negativePrompt =
        field("negative_prompt").map(_.str).getOrElse(options.negativePrompt),
      width = width,
      height = height,
      frames =
        field("video_frames").map(_.num.toInt).getOrElse(options.videoFrames),
      steps = sample.get("sample_steps").map(_.num.toInt).getOrElse(steps),
      cfgScale = guidance
        .flatMap(_.get("txt_cfg"))
        .map(_.num.toFloat)
        .getOrElse(options.cfgScale.toFloat),
      seed = if (seed < 0) Random.nextLong(Long.MaxValue) else seed,
      shift = sample
        .get("flow_shift")
        .filterNot(_.isNull)
        .map(_.num)
        .getOrElse(options.flowShift),
      highNoiseSteps = highNoise
        .get("sample_steps")
        .filterNot(_.isNull)
        .map(_.num.toInt)
        .filter(_ >= 0)
        .orElse(options.highNoiseSteps),
      highNoiseCfgScale = highNoise
        .get("guidance")
        .filterNot(_.isNull)
        .flatMap(_.obj.get("txt_cfg"))
        .map(_.num.toFloat)
        .orElse(options.highNoiseCfgScale.map(_.toFloat)),
      moeBoundary = field("moe_boundary")
        .map(_.num.toFloat)
        .getOrElse(options.moeBoundary.toFloat),
      fps = field("fps").map(_.num.toInt).orElse(options.fps),
      audioCfgScale = field("audio_cfg_scale").map(_.num.toFloat),
      ancestral = sample.get("sample_method").exists(_.strOpt.contains("euler_a")),
      modalityScale = field("modality_scale").map(_.num.toFloat).getOrElse(1f),
      audioModalityScale =
        field("audio_modality_scale").map(_.num.toFloat).getOrElse(1f),
      initImage = initImage.map { image =>
        def rounded(side: Int) =
          (side + pipeline.sizeMultiple - 1) / pipeline.sizeMultiple * pipeline.sizeMultiple
        Images.resized(image, rounded(width), rounded(height))
      }
    )
    enqueue(exchange, "vid_gen", () => video(pipeline, request))
  }

  private def enqueue(
      exchange: HttpExchange,
      kind: String,
      run: () => ujson.Value
  ): Unit = {
    val job = new Job(
      UUID.randomUUID().toString,
      System.currentTimeMillis() / 1000,
      kind,
      run
    )
    jobs.put(job.id, job)
    queue.put(job)
    json(
      exchange,
      202,
      ujson.Obj(
        "id" -> job.id,
        "kind" -> kind,
        "status" -> "queued",
        "created" -> ujson.Num(job.created.toDouble),
        "poll_url" -> s"/sdcpp/v1/jobs/${job.id}"
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

  /** `GET /sdcpp/v1/jobs/{id}` and `POST /sdcpp/v1/jobs/{id}/cancel`. */
  private def job(exchange: HttpExchange): Unit = {
    val parts = exchange.getRequestURI.getPath
      .stripPrefix("/sdcpp/v1/jobs/")
      .split('/')
      .toSeq
    Option(jobs.get(parts.head)) match {
      case None => json(exchange, 404, ujson.Obj("error" -> "no such job"))
      case Some(job) if parts.lift(1).contains("cancel") =>
        if (job.status == "queued" && queue.remove(job)) {
          job.status = "cancelled"
          job.completed = Some(System.currentTimeMillis() / 1000)
          json(exchange, 200, describe(job))
        } else
          json(exchange, 409, ujson.Obj("error" -> s"the job is ${job.status}"))
      case Some(job) => json(exchange, 200, describe(job))
    }
  }

  private def describe(job: Job): ujson.Value = {
    val position = queue.asScala.toSeq.indexOf(job)
    val described = ujson.Obj(
      "id" -> job.id,
      "kind" -> job.kind,
      "status" -> job.status,
      "created" -> ujson.Num(job.created.toDouble),
      "queue_position" -> math.max(position, 0)
    )
    job.started.foreach(t => described("started") = ujson.Num(t.toDouble))
    job.completed.foreach(t => described("completed") = ujson.Num(t.toDouble))
    if (job.status == "completed") job.result.foreach(described("result") = _)
    job.error.foreach(message =>
      described("error") =
        ujson.Obj("code" -> "generation_failed", "message" -> message)
    )
    described
  }

  /** A step's progress line, as sd-cpp prints it. */
  private def stepPrinter(): (Int, Int) => Unit = {
    var last = System.nanoTime()
    (step, steps) => {
      val now = System.nanoTime()
      val filled = 50 * step / steps
      println(
        f"  |${"=" * filled}${" " * (50 - filled)}| $step/$steps - ${(now - last) / 1e9}%.2fs/it"
      )
      last = now
    }
  }

  /** An `img_gen` job's result: `count` images of `request`, seeds in sequence.
    */
  private def images(
      pipeline: ImagePipeline,
      request: ImageRequest,
      count: Int
  ): ujson.Value = {
    println("sampling using Euler method")
    val pngs = (0 until count).map { index =>
      val seeded = request.copy(seed = request.seed + index)
      println(s"generating image ${index + 1}/$count (seed ${seeded.seed})")
      val image = pipeline.generate(seeded, stepPrinter())
      val png = new ByteArrayOutputStream()
      ImageIO.write(image, "png", png)
      png.toByteArray
    }
    ujson.Obj(
      "output_format" -> "png",
      "images" -> ujson.Arr.from(pngs.zipWithIndex.map { (png, index) =>
        ujson.Obj(
          "index" -> index,
          "b64_json" -> Base64.getEncoder.encodeToString(png)
        )
      })
    )
  }

  /** A `vid_gen` job's result: one webm. */
  private def video(
      pipeline: VideoPipeline,
      request: VideoRequest
  ): ujson.Value = {
    println("sampling using Euler method")
    println(
      s"generating ${pipeline.alignedFrames(request.frames)} frames of ${request.width} × ${request.height} (seed ${request.seed})"
    )
    val video = pipeline.generate(request, stepPrinter())
    val webm = VideoFiles.webm(video)
    ujson.Obj(
      "output_format" -> "webm",
      "mime_type" -> "video/webm",
      "fps" -> video.fps,
      "frame_count" -> video.frames.size,
      "b64_json" -> Base64.getEncoder.encodeToString(webm)
    )
  }

  /** The worker: runs the queue in order. */
  private def work(): Unit =
    while (true) {
      val job = queue.take()
      job.status = "generating"
      job.started = Some(System.currentTimeMillis() / 1000)
      try {
        job.result = Some(job.run())
        job.status = "completed"
      } catch {
        case error: Throwable =>
          System.err.println(s"[ERROR] generation failed: $error")
          job.error = Some(String.valueOf(error.getMessage))
          job.status = "failed"
      }
      job.completed = Some(System.currentTimeMillis() / 1000)
    }
}
