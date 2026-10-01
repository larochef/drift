package drift.frontend.pages.generate

import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Builds a request from the form and submits it: the form's fields over a base
  * — the session's defaults or, after a reuse, the recorded request — so what
  * the form never shows rides along unchanged.
  */
class GenerationSubmission(
    state: GenerationFormState,
    sessionId: String,
    service: GenerationService,
    loraService: LoraService,
    /** Free play (`specs/22-free-play-and-scratch-generations.md`). */
    scratch: Boolean,
    /** What the submission says about its project, as it stands at submit. */
    submitContext: () => SubmitContext
) {
  import state.*
  import GenerationFormState.cleanse

  private def intOf(v: Var[String], fallback: Int): Int =
    v.now().trim.toIntOption.getOrElse(fallback)

  private def doubleOf(v: Var[String], fallback: Double): Double =
    v.now().trim.toDoubleOption.getOrElse(fallback)

  private def chosen(v: Var[String]): Option[String] =
    Some(v.now().trim).filter(_.nonEmpty)

  /** The VAE-tiling block to send, or None when the feature is off or the box
    * is unchecked — None means the server's own default, same convention as
    * hires. Applies to both modes, so it is built once.
    */
  private def vaeTilingToSend(
      features: Map[String, Boolean],
      defaults: VaeTilingParameters
  ): Option[VaeTilingParameters] =
    if (!features.getOrElse("vae_tiling", false) || !vaeTilingEnabledVar.now())
      None
    else
      Some(
        defaults.copy(
          enabled = true,
          tileSizeX = intOf(vaeTileSizeXVar, defaults.tileSizeX),
          tileSizeY = intOf(vaeTileSizeYVar, defaults.tileSizeY),
          targetOverlap = doubleOf(vaeTargetOverlapVar, defaults.targetOverlap),
          relSizeX = doubleOf(vaeRelSizeXVar, defaults.relSizeX),
          relSizeY = doubleOf(vaeRelSizeYVar, defaults.relSizeY),
          temporalTiling = vaeTemporalTilingVar.now(),
          extraTilingArgs = vaeExtraTilingArgsVar.now().trim
        )
      )

  /** The form's guidance over a base: txt CFG always, the rest as typed —
    * which, untouched, is exactly what the base carried, so a recipe that never
    * opened the section round-trips unchanged (`specs/14`).
    */
  private def guidanceOver(base: GuidanceParameters): GuidanceParameters = {
    val baseSlg = base.slg.getOrElse(SlgParameters())
    val slg = SlgParameters(
      layers = slgLayersVar
        .now()
        .split(',')
        .toList
        .map(_.trim)
        .filter(_.nonEmpty)
        .flatMap(_.toIntOption),
      layerStart = doubleOf(slgLayerStartVar, baseSlg.layerStart),
      layerEnd = doubleOf(slgLayerEndVar, baseSlg.layerEnd),
      scale = doubleOf(slgScaleVar, baseSlg.scale)
    )
    base.copy(
      txtCfg = doubleOf(cfgVar, base.txtCfg),
      imgCfg = imgCfgVar.now().trim match {
        case ""    => None
        case typed => typed.toDoubleOption.orElse(base.imgCfg)
      },
      distilledGuidance =
        doubleOf(distilledGuidanceVar, base.distilledGuidance),
      // No SLG block unless there is something in it: a base that carried
      // none and a form nobody touched must send none, or the recipe changes
      // and a version is appended for nothing (`specs/19`).
      slg =
        if (base.slg.isEmpty && slg == SlgParameters()) None else Some(slg)
    )
  }

  /** Submits the form as one request of its current mode. */
  def submit(capabilities: SessionCapabilities): Unit = {
    val currentMode = mode.now()
    val defaults = capabilities.defaultsByMode
      .getOrElse(currentMode, GenerationDefaults())
    // drift picks the random seed itself, one per submission, and shows it
    // in the (disabled) seed input: a seed drift chose is one it can display
    // and record. Sending -1 and leaving the choice to sd-server was not
    // random at all at the commit in use — every such request landed on the
    // server's default seed 42.
    val seedValue =
      if (randomSeedVar.now()) {
        val picked = GenerationSubmission.randomSeed()
        seedVar.set(picked.toString)
        picked
      } else
        seedVar
          .now()
          .trim
          .toLongOption
          .getOrElse(GenerationSubmission.randomSeed())

    val features = capabilities.featuresByMode
      .getOrElse(currentMode, Map.empty)
    def attached(name: String, image: Option[String]): Option[String] =
      image.filter(_ => features.getOrElse(name, false))

    // The form's sampling fields over a base — the session's defaults or,
    // after a reuse, the recorded request — so what the form never shows
    // (SLG, eta) rides along unchanged. An empty sampler, scheduler or flow
    // shift is "(model default)": the field is omitted. Custom sigmas
    // are their own step count; a list that does not read keeps the base's.
    def sampleOver(base: SampleParameters): SampleParameters = {
      val cleansed = cleanse(base)
      val sigmas = GenerationFormState
        .sigmasOf(sigmasVar.now())
        .getOrElse(cleansed.customSigmas)
      cleansed.copy(
        sampleSteps =
          if (sigmas.isEmpty) intOf(stepsVar, cleansed.sampleSteps)
          else GenerationFormState.sigmaSteps(sigmas),
        customSigmas = sigmas,
        flowShift = flowShiftVar.now().trim.toDoubleOption,
        guidance = guidanceOver(cleansed.guidance),
        sampleMethod = chosen(samplerVar),
        scheduler = chosen(schedulerVar)
      )
    }

    // The high-noise expert keeps its own steps and CFG; the sampler and
    // scheduler are the model's business, so they ride along unchanged.
    def highNoiseOver(base: SampleParameters): SampleParameters = {
      val cleansed = cleanse(base)
      cleansed.copy(
        sampleSteps = intOf(highNoiseStepsVar, cleansed.sampleSteps),
        guidance = cleansed.guidance.copy(
          txtCfg = doubleOf(highNoiseCfgVar, cleansed.guidance.txtCfg)
        )
      )
    }

    // Each selected LoRA contributes all its files: a wan 2.2 pair becomes
    // two entries at the same multiplier, the high-noise one flagged.
    val pendingLoras = pendingLoraSelections.now()
    val loraSelections =
      if (!features.getOrElse("lora", false)) List.empty
      else if (pendingLoras.nonEmpty)
        // A recipe named LoRAs the collection had not loaded yet, so the
        // picker is still empty: the recorded selections are the truth, and
        // resending them keeps the recipe whole. Submitting without them
        // would appear to be a changed recipe and append a version for it
        // (François, 2026-09-10).
        pendingLoras
      else {
        val byId = loraService.lorasNow.map(l => l.id -> l).toMap
        val strengths = loraStrengthsVar.now()
        val (known, gone) =
          selectedLoraIdsVar.now().partition(byId.contains)
        if (gone.nonEmpty)
          warnAboutRecipe(
            s"LoRA no longer installed, not sent: ${gone.mkString(", ")}."
          )
        known.flatMap { loraId =>
          byId.get(loraId).toList.flatMap { lora =>
            val multiplier = strengths
              .get(loraId)
              .flatMap(_.trim.toDoubleOption)
              .getOrElse(lora.defaultStrength)
            lora.selections(multiplier)
          }
        }
      }

    if (currentMode == "vid_gen") {
      val control = attached("control_video", controlVideoVar.now())
      val base = reusedVideoBase
        .now()
        .getOrElse(
          VideoGenerationParameters(
            prompt = "",
            clipSkip = defaults.clipSkip,
            moeBoundary = defaults.moeBoundary,
            sampleParams = defaults.sampleParams,
            highNoiseSampleParams = defaults.highNoiseSampleParams,
            outputFormat = Some(defaults.outputFormat)
              .filter(_.nonEmpty)
              .getOrElse("webm"),
            outputCompression = defaults.outputCompression
          )
        )
      service.push(
        GenerationService.Command.SubmitVideo(
          sessionId,
          context = submitContext(),
          scratch = scratch,
          parameters = base.copy(
            prompt = promptVar.now(),
            negativePrompt = negativePromptVar.now(),
            width = intOf(widthVar, defaults.width),
            height = intOf(heightVar, defaults.height),
            strength = doubleOf(strengthVar, defaults.strength),
            seed = seedValue,
            videoFrames = intOf(videoFramesVar, defaults.videoFrames),
            fps = intOf(fpsVar, defaults.fps),
            initImage = attached("init_image", initImageVar.now()),
            endImage = attached("end_image", endImageVar.now()),
            // No control-frame picker yet (spec 14); a reused base carries
            // URLs there, not payloads, so they cannot be resent.
            controlFrames = List.empty,
            references =
              if (features.getOrElse("references", false))
                referencesVar.now()
              else List.empty,
            guides =
              if (!features.getOrElse("guides", false)) List.empty
              else
                guidesVar
                  .now()
                  .map(guide =>
                    VideoGuide(
                      guide.media,
                      guide.frameIndex.trim.toIntOption.getOrElse(-1)
                    )
                  ),
            // The rest of the control block means nothing without the video,
            // so it is sent only beside one; blanks are the runner's
            // defaults, spelled out so the recipe records what ran.
            controlVideo = control,
            controlStrength = control.map(_ => doubleOf(controlStrengthVar, 1)),
            controlStart = control.map(_ => doubleOf(controlStartVar, 0)),
            controlEnd = control.map(_ => doubleOf(controlEndVar, 1)),
            controlMask = control.flatMap(_ => controlMaskVar.now()),
            sourceVideo = control.flatMap(_ => sourceVideoVar.now()),
            lora = loraSelections,
            sampleParams = sampleOver(base.sampleParams),
            highNoiseSampleParams =
              base.highNoiseSampleParams.map(highNoiseOver),
            vaeTilingParams = vaeTilingToSend(
              features,
              base.vaeTilingParams.getOrElse(
                defaults.vaeTilingParams.getOrElse(VaeTilingParameters())
              )
            )
          )
        )
      )
    } else {
      val base = reusedImageBase
        .now()
        .getOrElse(
          ImageGenerationParameters(
            prompt = "",
            clipSkip = defaults.clipSkip,
            sampleParams = defaults.sampleParams,
            outputFormat =
              Some(defaults.outputFormat).filter(_.nonEmpty).getOrElse("png"),
            outputCompression = defaults.outputCompression
          )
        )
      // Only an enabled hires block is sent: omitted means the server's own
      // default, which is off unless the launch flags said otherwise.
      val hiresDefaults =
        base.hires.getOrElse(defaults.hires.getOrElse(HiresParameters()))
      val hires =
        if (!features.getOrElse("hires", false) || !hiresEnabledVar.now())
          None
        else
          Some(
            hiresDefaults.copy(
              enabled = true,
              upscaler =
                chosen(hiresUpscalerVar).getOrElse(hiresDefaults.upscaler),
              scale = doubleOf(hiresScaleVar, hiresDefaults.scale),
              targetWidth =
                intOf(hiresTargetWidthVar, hiresDefaults.targetWidth),
              targetHeight =
                intOf(hiresTargetHeightVar, hiresDefaults.targetHeight),
              steps = intOf(hiresStepsVar, hiresDefaults.steps),
              denoisingStrength = doubleOf(
                hiresDenoisingStrengthVar,
                hiresDefaults.denoisingStrength
              ),
              upscaleTileSize = intOf(
                hiresUpscaleTileSizeVar,
                hiresDefaults.upscaleTileSize
              )
            )
          )
      service.push(
        GenerationService.Command.SubmitImage(
          sessionId,
          context = submitContext(),
          scratch = scratch,
          parameters = base.copy(
            prompt = promptVar.now(),
            negativePrompt = negativePromptVar.now(),
            width = intOf(widthVar, defaults.width),
            height = intOf(heightVar, defaults.height),
            strength = doubleOf(strengthVar, defaults.strength),
            seed = seedValue,
            initImage = attached("init_image", initImageVar.now()),
            refImages =
              if (features.getOrElse("ref_images", false))
                refImagesVar.now()
              else List.empty,
            // A mask means nothing without the image it masks, so it is
            // sent only beside an init image (`specs/14`).
            maskImage = attached("mask_image", maskImageVar.now())
              .filter(_ => initImageVar.now().isDefined),
            batchCount = intOf(batchCountVar, base.batchCount)
              .max(1)
              .min(capabilities.limits.maxBatchCount),
            lora = loraSelections,
            hires = hires,
            vaeTilingParams = vaeTilingToSend(
              features,
              base.vaeTilingParams.getOrElse(
                defaults.vaeTilingParams.getOrElse(VaeTilingParameters())
              )
            ),
            sampleParams = sampleOver(base.sampleParams)
          )
        )
      )
    }
  }
}

object GenerationSubmission {

  /** A fresh seed in sd-cpp's comfortable range (a non-negative 31-bit int,
    * what its own `rand()` produced).
    */
  def randomSeed(): Long = scala.util.Random.nextInt(Int.MaxValue).toLong
}
