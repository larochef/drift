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

/** `sd-server`'s native API (`/sdcpp/v1`) on the runner's image pipeline, as
  * drift drives it (`specs/42`, step 12): the capabilities document (the form's
  * defaults, from the launch flags), `img_gen` jobs queued and run one at a
  * time, polled, and cancelled while queued. A request starts from the launch
  * flags' values and overrides what it sends. Progress goes to the log as
  * sd-cpp prints it (`| i/n - Xs/it`), which drift parses. Images come as
  * base64 or data URLs; references are stretched to the output's size first
  * unless `auto_resize_ref_image` is false, as sd-server does.
  */
final class ImageServer(options: ImageOptions, pipeline: ImagePipeline) {

  /** One `img_gen` job: `count` images of `request`, seeds in sequence. */
  final private class Job(
      val id: String,
      val created: Long,
      val request: ImageRequest,
      val count: Int
  ) {
    @volatile var status: String = "queued"
    @volatile var started: Option[Long] = None
    @volatile var completed: Option[Long] = None
    @volatile var images: Seq[Array[Byte]] = Nil
    @volatile var error: Option[String] = None
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
    route("/sdcpp/v1/img_gen")(submit)
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

  private def capabilities: ujson.Value = {
    val name = options.diffusionModel.getFileName.toString
    val defaults = ujson.Obj(
      "prompt" -> options.prompt,
      "negative_prompt" -> options.negativePrompt,
      "clip_skip" -> -1,
      "width" -> options.width,
      "height" -> options.height,
      "strength" -> 0.75,
      "seed" -> ujson.Num(options.seed.toDouble),
      "batch_count" -> 1,
      "sample_params" -> ujson.Obj(
        "scheduler" -> "default",
        "sample_method" -> "euler",
        "sample_steps" -> options.steps,
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
      ),
      "output_format" -> "png",
      "output_compression" -> 100
    )
    ujson.Obj(
      "model" -> ujson.Obj(
        "name" -> name,
        "stem" -> name.stripSuffix(".safetensors").stripSuffix(".gguf"),
        "path" -> options.diffusionModel.toString
      ),
      "current_mode" -> "img_gen",
      "supported_modes" -> ujson.Arr("img_gen"),
      "defaults_by_mode" -> ujson.Obj("img_gen" -> defaults),
      "features_by_mode" -> ujson.Obj(
        "img_gen" -> ujson.Obj(
          "init_image" -> pipeline.takesInitImage,
          "mask_image" -> false,
          "ref_images" -> pipeline.takesReferences,
          "lora" -> (pipeline.takesLoras && options.loraDirectory.isDefined),
          "hires" -> false,
          "vae_tiling" -> false,
          "cancel_queued" -> true,
          "cancel_generating" -> false
        )
      ),
      "output_formats_by_mode" -> ujson.Obj("img_gen" -> ujson.Arr("png")),
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

  /** `POST /sdcpp/v1/img_gen`: a job from the request over the launch flags'
    * values; what the runner cannot do yet is refused (400).
    */
  private def submit(exchange: HttpExchange): Unit = {
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
      steps =
        sample.get("sample_steps").map(_.num.toInt).getOrElse(options.steps),
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
    val job = new Job(
      UUID.randomUUID().toString,
      System.currentTimeMillis() / 1000,
      request,
      field("batch_count").map(_.num.toInt).getOrElse(1).max(1)
    )
    jobs.put(job.id, job)
    queue.put(job)
    json(
      exchange,
      202,
      ujson.Obj(
        "id" -> job.id,
        "kind" -> "img_gen",
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
      "kind" -> "img_gen",
      "status" -> job.status,
      "created" -> ujson.Num(job.created.toDouble),
      "queue_position" -> math.max(position, 0)
    )
    job.started.foreach(t => described("started") = ujson.Num(t.toDouble))
    job.completed.foreach(t => described("completed") = ujson.Num(t.toDouble))
    if (job.status == "completed")
      described("result") = ujson.Obj(
        "output_format" -> "png",
        "images" -> ujson.Arr.from(job.images.zipWithIndex.map { (png, index) =>
          ujson.Obj(
            "index" -> index,
            "b64_json" -> Base64.getEncoder.encodeToString(png)
          )
        })
      )
    job.error.foreach(message =>
      described("error") =
        ujson.Obj("code" -> "generation_failed", "message" -> message)
    )
    described
  }

  /** The worker: runs the queue in order. */
  private def work(): Unit =
    while (true) {
      val job = queue.take()
      job.status = "generating"
      job.started = Some(System.currentTimeMillis() / 1000)
      try {
        println("sampling using Euler method")
        job.images = (0 until job.count).map { index =>
          val request = job.request.copy(seed = job.request.seed + index)
          var last = System.nanoTime()
          val image = pipeline.generate(
            request,
            (step, steps) => {
              val now = System.nanoTime()
              val filled = 50 * step / steps
              println(
                f"  |${"=" * filled}${" " * (50 - filled)}| $step/$steps - ${(now - last) / 1e9}%.2fs/it"
              )
              last = now
            }
          )
          val png = new ByteArrayOutputStream()
          ImageIO.write(image, "png", png)
          png.toByteArray
        }
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
