package drift.runner.server

import drift.protocol.JobRoutes
import drift.runner.diffusion.*

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** The capabilities document (`GET /sdcpp/v1/capabilities`): what the launched
  * model takes and the form's defaults, from the launch flags.
  */
final private[server] class ServerCapabilities(
    options: ImageOptions,
    pipeline: Pipeline.Opened,
    launch: LaunchDefaults
) {

  private val mode = pipeline.mode

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

  def document: ujson.Value = {
    val name = options.diffusionModel.getFileName.toString
    val sampleParams = ujson.Obj(
      "scheduler" -> "default",
      "sample_method" -> (pipeline match {
        case image: ImagePipeline =>
          launch.launchSampler(image).getOrElse(image.defaultSampler)
        case _ => "euler"
      }),
      "sample_steps" -> launch.steps,
      "eta" -> 0.0,
      "shifted_timestep" -> 0,
      "custom_sigmas" -> ujson.Arr.from(launch.launchSigmas),
      "flow_shift" -> launch.defaultShift
        .fold[ujson.Value](ujson.Null)(ujson.Num(_)),
      "guidance" -> ujson.Obj(
        // no `img_cfg`: the runner reads none, and a default the form sends
        // back would be answered with a warning that it was ignored
        "txt_cfg" -> options.cfgScale,
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
}
