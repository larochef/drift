package drift.frontend.pages.gallery

import drift.shared.*

import scala.scalajs.js

/** Turns a recorded generation into rows the detail view lists — every
  * parameter the sidecar carries, in words, so a result can be read back
  * without knowing the native field names. Pure functions, no DOM.
  */
object RecordedParameters {

  def promptOf(generation: Generation): String =
    generation.imageParameters
      .map(_.prompt)
      .orElse(generation.videoParameters.map(_.prompt))
      .getOrElse("")

  /** The prompt an entry stands for: its own, else the one its PiD request
    * used, else its parent's — a resize or ESRGAN upscale records none, so the
    * chain is walked through `find` back to the generation that had one.
    */
  def sourcePromptOf(
      generation: Generation,
      find: String => Option[Generation]
  ): String = {
    @annotation.tailrec
    def walk(current: Generation, depth: Int): String =
      Some(promptOf(current))
        .filter(_.nonEmpty)
        .orElse(current.derivation.flatMap(_.prompt).filter(_.nonEmpty)) match {
        case Some(prompt) => prompt
        case None         =>
          current.derivation.flatMap(d => find(d.parentId)) match {
            case Some(parent) if depth < 16 => walk(parent, depth + 1)
            case _                          => ""
          }
      }
    walk(generation, 0)
  }

  def kindLabel(generation: Generation): String = generation.kind match {
    case "vid_gen"                  => "Video"
    case "upscale"                  => "Upscale"
    case "pid"                      => "PiD upscale"
    case SeedVr2UpscaleRequest.Kind => "SeedVR2 upscale"
    case "redraw"                   => "Redraw"
    case "edit"                     => "Edit"
    case "resize"                   => "Resize"
    case "import"                   => "Imported image"
    case _                          => "Image"
  }

  /** What was done to the parent, in words. */
  def operationOf(derivation: Derivation): String = {
    val size = (derivation.width, derivation.height) match {
      case (Some(w), Some(h)) => s" to ${w} × ${h}"
      case _                  => ""
    }
    derivation.operation match {
      case "upscale" =>
        val times =
          derivation.repeats.filter(_ > 1).map(r => s" ×$r").getOrElse("")
        s"Upscaled$size with ${derivation.upscalerId.getOrElse("?")}$times"
      case "pid" =>
        s"PiD-upscaled$size with ${derivation.configurationId.getOrElse("?")}"
      case SeedVr2UpscaleRequest.Kind =>
        val times = derivation.repeats.map(r => s" ×$r").getOrElse("")
        s"SeedVR2-upscaled$times$size with ${derivation.configurationId.getOrElse("?")}"
      case "redraw" =>
        s"Redrawn with ${derivation.configurationId.getOrElse("?")}" +
          derivation.strength.map(s => s" at strength $s").getOrElse("")
      case "edit" =>
        s"Edited with ${derivation.configurationId.getOrElse("?")}" +
          derivation.instructions.map(i => s": $i").getOrElse("")
      case "resize" =>
        s"Resized$size (${derivation.fit.getOrElse("fit")})"
      case other => other
    }
  }

  /** The card's first line: the prompt, or for a derived entry the operation.
    */
  def titleOf(generation: Generation): String =
    Some(promptOf(generation))
      .filter(_.nonEmpty)
      .orElse(generation.derivation.map(operationOf))
      .orElse(generation.importedFileName)
      .getOrElse("")

  def timeOf(millis: Long): String =
    new js.Date(millis.toDouble).toLocaleTimeString()

  def dateTimeOf(millis: Long): String =
    new js.Date(millis.toDouble).toLocaleString()

  def durationOf(generation: Generation): Option[String] =
    generation.completedAt.map { completed =>
      val from = generation.startedAt.getOrElse(generation.submittedAt)
      s"${((completed - from) / 1000).max(0)}s"
    }

  /** Label → value, in reading order — the seed the output at `outputIndex` of
    * a batch was made with.
    */
  def rows(generation: Generation, outputIndex: Int): List[(String, String)] =
    generation.derivation
      .map(derivedRows)
      .orElse(
        generation.imageParameters.map(p =>
          imageRows(
            p.copy(seed = generation.seedOf(outputIndex).getOrElse(p.seed))
          )
        )
      )
      .orElse(generation.videoParameters.map(videoRows))
      .orElse(
        generation.importedFileName.map(name => List("Imported file" -> name))
      )
      .getOrElse(List.empty)

  private def derivedRows(d: Derivation): List[(String, String)] =
    List("Operation" -> operationOf(d)) ++
      d.upscalerId.map("Upscaler" -> _) ++
      d.repeats.map(r => "Repeats" -> r.toString) ++
      d.fit.map("Fit" -> _) ++
      d.configurationId.map("Configuration" -> _) ++
      d.prompt.map("Prompt" -> _) ++
      d.instructions.map("Instructions" -> _) ++
      d.strength.map(v => "Strength" -> v.toString) ++
      d.steps.map(v => "Steps" -> v.toString) ++
      d.seed.map(v => "Seed" -> v.toString) ++
      List(
        "Original" -> d.parentId,
        "Original file" -> d.parentFileName
      )

  /** The persisted inputs: label → URL — images, and a video model's clips and
    * sounds (`specs/42`, step 14), a guide's frame in its label. Inputs whose
    * persistence failed carry a placeholder instead of a URL and are left out.
    */
  def inputMedia(generation: Generation): List[(String, String)] = {
    val fromImage = generation.imageParameters.toList.flatMap { p =>
      p.initImage.map("Init image" -> _).toList ++
        p.maskImage.map("Mask" -> _).toList ++
        p.refImages.zipWithIndex.map((url, i) => s"Reference ${i + 1}" -> url)
    }
    val fromVideo = generation.videoParameters.toList.flatMap { p =>
      p.initImage.map("Start image" -> _).toList ++
        p.endImage.map("End image" -> _).toList ++
        p.controlFrames.zipWithIndex.map((url, i) =>
          s"Control frame ${i + 1}" -> url
        ) ++
        p.references.zipWithIndex.map((url, i) =>
          s"Reference ${i + 1}" -> url
        ) ++
        p.guides.zipWithIndex.map((guide, i) =>
          s"Guide ${i + 1} · frame ${guide.frameIndex}" -> guide.media
        ) ++
        p.controlVideo.map("Control video" -> _).toList ++
        p.controlMask.map("Control mask" -> _).toList ++
        p.sourceVideo.map("Source video" -> _).toList
    }
    (fromImage ++ fromVideo).filter(_._2.startsWith("/"))
  }

  private def seedText(seed: Long): String =
    if (seed < 0) "random" else seed.toString

  /** Doubles as the user typed them, not as float32 round-tripped them: sd-cpp
    * echoes `0.01` back as `0.009999999776482582`.
    */
  private def number(value: Double): String = {
    val rounded = math.round(value * 10000) / 10000.0
    if (rounded == rounded.toLong.toDouble) rounded.toLong.toString
    else rounded.toString
  }

  private def sampleRows(
      sample: SampleParameters,
      prefix: String
  ): List[(String, String)] = {
    val guidance = sample.guidance
    List(
      s"${prefix}Steps" -> sample.sampleSteps.toString,
      s"${prefix}CFG" -> number(guidance.txtCfg)
    ) ++
      guidance.imgCfg.map(v => s"${prefix}Image CFG" -> number(v)) ++
      List(
        s"${prefix}Distilled guidance" -> number(guidance.distilledGuidance)
      ) ++
      sample.sampleMethod.map(s"${prefix}Sampler" -> _) ++
      sample.scheduler.map(s"${prefix}Scheduler" -> _) ++
      sample.eta.map(v => s"${prefix}Eta" -> number(v)) ++
      sample.flowShift.map(v => s"${prefix}Flow shift" -> number(v)) ++
      Option(sample.shiftedTimestep)
        .filter(_ != 0)
        .map(v => s"${prefix}Shifted timestep" -> v.toString) ++
      Option(sample.customSigmas)
        .filter(_.nonEmpty)
        .map(v => s"${prefix}Custom sigmas" -> v.map(number).mkString(", ")) ++
      guidance.slg
        .filter(slg => slg.layers.nonEmpty || slg.scale != 0.0)
        .map(slg =>
          s"${prefix}SLG" ->
            s"layers ${slg.layers.mkString(",")} · ${number(slg.layerStart)}–${number(slg.layerEnd)} · scale ${number(slg.scale)}"
        )
  }

  private def loraRows(loras: List[LoraSelection]): List[(String, String)] =
    loras.zipWithIndex.map { (lora, i) =>
      s"LoRA ${i + 1}" ->
        s"${lora.path} × ${number(lora.multiplier)}${
            if (lora.isHighNoise) " (high noise)"
            else ""
          }"
    }

  private def vaeRows(
      tiling: Option[VaeTilingParameters]
  ): List[(String, String)] =
    tiling.filter(_.enabled).toList.flatMap { vae =>
      val size =
        if (vae.relSizeX != 0.0 || vae.relSizeY != 0.0)
          s"relative ${number(vae.relSizeX)} × ${number(vae.relSizeY)}"
        else if (vae.tileSizeX != 0 || vae.tileSizeY != 0)
          s"${vae.tileSizeX} × ${vae.tileSizeY} latent"
        else "server default"
      List(
        "VAE tiling" ->
          s"$size · overlap ${number(vae.targetOverlap)}${
              if (vae.temporalTiling)
                " · temporal"
              else ""
            }${
              if (vae.extraTilingArgs.nonEmpty)
                s" · ${vae.extraTilingArgs}"
              else ""
            }"
      )
    }

  private def imageRows(p: ImageGenerationParameters): List[(String, String)] =
    List(
      "Prompt" -> p.prompt,
      "Negative prompt" -> p.negativePrompt,
      "Size" -> s"${p.width} × ${p.height}",
      "Seed" -> seedText(p.seed)
    ) ++
      sampleRows(p.sampleParams, "") ++
      List("Clip skip" -> p.clipSkip.toString) ++
      p.initImage.map(_ => "Strength" -> number(p.strength)) ++
      Option(p.batchCount)
        .filter(_ > 1)
        .map(v => "Batch count" -> v.toString) ++
      p.hires
        .filter(_.enabled)
        .map { hires =>
          val target =
            if (hires.targetWidth > 0 || hires.targetHeight > 0)
              s"${hires.targetWidth} × ${hires.targetHeight}"
            else s"× ${number(hires.scale)}"
          "Hires fix" ->
            s"${hires.upscaler} $target · ${
                if (hires.steps == 0) "derived"
                else hires.steps.toString
              } steps · denoise ${number(hires.denoisingStrength)} · tile ${hires.upscaleTileSize}"
        } ++
      vaeRows(p.vaeTilingParams) ++
      loraRows(p.lora) ++
      List("Output" -> s"${p.outputFormat} · quality ${p.outputCompression}")

  private def videoRows(p: VideoGenerationParameters): List[(String, String)] =
    List(
      "Prompt" -> p.prompt,
      "Negative prompt" -> p.negativePrompt,
      "Size" -> s"${p.width} × ${p.height}",
      "Frames" -> s"${p.videoFrames} @ ${p.fps} fps",
      "Seed" -> seedText(p.seed)
    ) ++
      sampleRows(p.sampleParams, "") ++
      p.highNoiseSampleParams.toList
        .flatMap(sampleRows(_, "High-noise ")) ++
      List("Clip skip" -> p.clipSkip.toString) ++
      p.initImage.map(_ => "Strength" -> number(p.strength)) ++
      p.moeBoundary.map(v => "MoE boundary" -> number(v)) ++
      p.vaceStrength.map(v => "VACE strength" -> number(v)) ++
      p.controlVideo.map(_ =>
        "Control" ->
          s"strength ${number(p.controlStrength.getOrElse(1.0))} · steps ${number(
              p.controlStart.getOrElse(0.0)
            )}–${number(p.controlEnd.getOrElse(1.0))}"
      ) ++
      vaeRows(p.vaeTilingParams) ++
      loraRows(p.lora) ++
      List("Output" -> s"${p.outputFormat} · quality ${p.outputCompression}")
}
