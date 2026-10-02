package drift.frontend.pages.generate

import drift.frontend.components.LoraPicker
import drift.frontend.services.*
import drift.frontend.services.GenerationService.Reuse
import drift.shared.*

import com.raquo.laminar.api.L.Var

/** Lays values over the form: a mode's defaults, a reused generation
  * (`specs/12-gallery.md`), a project version (`specs/19-…`) and the
  * configuration's default LoRAs (`specs/28-configuration-loras.md`). It only
  * writes — see `applyLoras` for why nothing written here is read back in the
  * same pass.
  */
class RecipeSeeding(
    state: GenerationFormState,
    /** The launched configuration — a full parameter reuse is honoured only
      * when the recorded generation names this very one.
      */
    configurationId: String,
    loraService: LoraService,
    /** The configuration's default LoRAs as they stand now. */
    configurationLoras: () => List[ConfiguredLora]
) {
  import state.*
  import GenerationFormState.{cleanse, modeLabel}

  /** Fills the mode-dependent fields from that mode's defaults. The prompt
    * survives a mode switch; everything numeric follows the mode.
    */
  def seedModeFields(
      capabilities: SessionCapabilities,
      newMode: String
  ): Unit = {
    val defaults =
      capabilities.defaultsByMode.getOrElse(newMode, GenerationDefaults())
    val sample = cleanse(defaults.sampleParams)
    untouchSampling()
    seedFor(newMode)
    widthVar.set(defaults.width.toString)
    heightVar.set(defaults.height.toString)
    stepsVar.set(sample.sampleSteps.toString)
    defaults.highNoiseSampleParams.foreach(applyHighNoise)
    cfgVar.set(sample.guidance.txtCfg.toString)
    applyGuidance(sample.guidance)
    samplerVar.set(sample.sampleMethod.getOrElse(""))
    schedulerVar.set(sample.scheduler.getOrElse(""))
    sigmasVar.set(GenerationFormState.sigmasText(sample.customSigmas))
    flowShiftVar.set(sample.flowShift.fold("")(_.toString))
    seedVar.set(if (defaults.seed < 0) "" else defaults.seed.toString)
    videoFramesVar.set(defaults.videoFrames.toString)
    fpsVar.set(defaults.fps.toString)
    strengthVar.set(defaults.strength.toString)
    batchCountVar.set(defaults.batchCount.toString)
    applyHires(defaults.hires.getOrElse(HiresParameters()))
    applyVaeTiling(defaults.vaeTilingParams.getOrElse(VaeTilingParameters()))
    if (promptVar.now().isEmpty && defaults.prompt.nonEmpty)
      promptVar.set(defaults.prompt)
    if (negativePromptVar.now().isEmpty && defaults.negativePrompt.nonEmpty)
      negativePromptVar.set(defaults.negativePrompt)
  }

  private def applyHires(hires: HiresParameters): Unit = {
    hiresEnabledVar.set(hires.enabled)
    hiresUpscalerVar.set(hires.upscaler)
    hiresScaleVar.set(hires.scale.toString)
    hiresTargetWidthVar.set(
      if (hires.targetWidth == 0) "" else hires.targetWidth.toString
    )
    hiresTargetHeightVar.set(
      if (hires.targetHeight == 0) "" else hires.targetHeight.toString
    )
    hiresStepsVar.set(if (hires.steps == 0) "" else hires.steps.toString)
    hiresDenoisingStrengthVar.set(hires.denoisingStrength.toString)
    hiresUpscaleTileSizeVar.set(hires.upscaleTileSize.toString)
  }

  private def applyVaeTiling(vae: VaeTilingParameters): Unit = {
    vaeTilingEnabledVar.set(vae.enabled)
    vaeTileSizeXVar.set(if (vae.tileSizeX == 0) "" else vae.tileSizeX.toString)
    vaeTileSizeYVar.set(if (vae.tileSizeY == 0) "" else vae.tileSizeY.toString)
    vaeTargetOverlapVar.set(vae.targetOverlap.toString)
    vaeRelSizeXVar.set(if (vae.relSizeX == 0.0) "" else vae.relSizeX.toString)
    vaeRelSizeYVar.set(if (vae.relSizeY == 0.0) "" else vae.relSizeY.toString)
    vaeTemporalTilingVar.set(vae.temporalTiling)
    vaeExtraTilingArgsVar.set(vae.extraTilingArgs)
  }

  // ----------------------------------------------------------------- reuse

  /** Applies what the gallery asked for (`specs/12-gallery.md`), right after
    * the defaults seeded the form. A full reuse is honoured only for this very
    * configuration; on another one the recorded request becomes a *task* — the
    * form's fields as recorded over this configuration's own defaults — which
    * is how two models get compared on the same job.
    */
  def applyReuse(
      capabilities: SessionCapabilities,
      reuse: Reuse
  ): Unit = {
    // Whatever the last recipe could not carry over is stale from here on;
    // what this one cannot carry is raised during the seeding below.
    reuseWarnings.set(List.empty)
    // The configuration's LoRAs, unless a recipe of this configuration brings
    // its own below.
    applyConfigurationLoras()
    reuse match {
      case Reuse.Full(generation)
          if generation.runConfigurationId == configurationId &&
            capabilities.supportedModes.contains(generation.kind) =>
        seedFromRecord(capabilities, generation, withLoras = true)
        reusedImageBase.set(generation.imageParameters)
        reusedVideoBase.set(generation.videoParameters)
        reuseNoticeText.set(
          Some(
            s"Form seeded from generation ${generation.id}: every recorded " +
              "parameter, seed included."
          )
        )
      case Reuse.Full(generation) =>
        applyReuse(capabilities, Reuse.Task(generation))
      case Reuse.Task(generation)
          if capabilities.supportedModes.contains(generation.kind) =>
        seedFromRecord(capabilities, generation, withLoras = false)
        // The record's fields cleared the picker; this configuration's own
        // LoRAs go back in.
        applyConfigurationLoras()
        reusedImageBase.set(None)
        reusedVideoBase.set(None)
        reuseNoticeText.set(
          Some(
            s"Task of generation ${generation.id} on this configuration: " +
              "prompt, images, seed, size and the sampling settings shown " +
              "are as recorded; the LoRAs are this configuration's own, and " +
              "anything the form " +
              "does not show is this configuration's own default. Adjust " +
              "steps and CFG if this model wants others."
          )
        )
      case Reuse.Task(generation) =>
        // An image task on a video-only model, or the reverse: only the
        // prompt side means anything here.
        val (prompt, negativePrompt, seed, width, height) =
          generation.imageParameters
            .map(p => (p.prompt, p.negativePrompt, p.seed, p.width, p.height))
            .orElse(
              generation.videoParameters
                .map(p =>
                  (p.prompt, p.negativePrompt, p.seed, p.width, p.height)
                )
            )
            .getOrElse(("", "", -1L, 512, 512))
        promptVar.set(prompt)
        negativePromptVar.set(negativePrompt)
        widthVar.set(width.toString)
        heightVar.set(height.toString)
        applySeed(seed)
        reuseNoticeText.set(
          Some(
            s"Generation ${generation.id} was a ${modeLabel(generation.kind).toLowerCase}; " +
              "this model does not make those, so only its prompt, negative " +
              "prompt, seed and size were reused."
          )
        )
    }
  }

  /** Seeds the form from a version exactly as the gallery's reuse would from
    * the generation that made it: field for field on the version's own
    * configuration, best effort elsewhere.
    */
  def applyVersion(
      capabilities: SessionCapabilities,
      version: PromptVersion
  ): Unit = {
    // How many to make is not part of a recipe either (`specs/14`): picking a
    // version keeps the batch count the user has, like it rolls a new seed.
    val batchCount = batchCountVar.now()
    applyReuse(
      capabilities,
      Reuse.Full(
        Generation(
          id = version.id,
          sessionId = "",
          runConfigurationId = version.runConfigurationId,
          kind = version.kind,
          status = GenerationStatus.Completed,
          submittedAt = version.createdAt,
          imageParameters = version.imageParameters,
          videoParameters = version.videoParameters,
          importedFileName = None,
          inputSources = List.empty
        )
      )
    )
    // The seed is not part of a recipe: a version's seed is one that ran,
    // and the next run of the same version should roll its own.
    applySeed(-1)
    batchCountVar.set(
      if (batchCount.nonEmpty) batchCount
      else
        capabilities.defaultsByMode
          .get("img_gen")
          .map(_.batchCount.toString)
          .getOrElse("1")
    )
    reuseNoticeText.set(
      Some(
        if (version.runConfigurationId == configurationId)
          s"Form follows v${version.number}: its recipe, LoRAs included; the seed rolls anew."
        else
          s"Form follows v${version.number} on another configuration: prompts, size and " +
            "the sampling settings shown as recorded, this configuration's LoRAs and defaults for the rest."
      )
    )
  }

  /** The record's mode and every field the form shows, over that mode's
    * defaults. LoRAs come along only onto the configuration the record ran on;
    * elsewhere they are left out and named, since a LoRA suits a model rather
    * than a task — another architecture cannot load it, and a turbo LoRA is
    * dead weight on a model that is turbo already (François, 2026-09-12).
    */
  private def seedFromRecord(
      capabilities: SessionCapabilities,
      generation: Generation,
      withLoras: Boolean
  ): Unit = {
    mode.set(generation.kind)
    seedModeFields(capabilities, generation.kind)
    generation.imageParameters
      .map(p => if (withLoras) p else p.copy(lora = List.empty))
      .foreach(applyImageParameters(capabilities, _))
    generation.videoParameters
      .map(p => if (withLoras) p else p.copy(lora = List.empty))
      .foreach(applyVideoParameters(capabilities, _))
    val leftOut =
      if (withLoras) List.empty
      else
        generation.imageParameters.toList.flatMap(_.lora) ++
          generation.videoParameters.toList.flatMap(_.lora)
    if (leftOut.nonEmpty)
      warnAboutRecipe(
        "LoRAs left out, the recipe ran on another model: " +
          leftOut
            .map(selection => loraName(selection.path))
            .distinct
            .mkString(", ") +
          ". This configuration's own LoRAs apply; add back any that suit it."
      )
  }

  /** A recorded LoRA path as the user knows it: the installed LoRA's label,
    * else the path itself.
    */
  private def loraName(path: String): String =
    loraService.loadedLorasNow
      .flatMap(loraOf(_, path))
      .map(_.label)
      .getOrElse(path)

  private def applySeed(seed: Long): Unit = {
    randomSeedVar.set(seed < 0)
    seedVar.set(if (seed < 0) "" else seed.toString)
  }

  /** Guidance beyond txt CFG into the form, from defaults or a recording. */
  private def applyGuidance(guidance: GuidanceParameters): Unit = {
    imgCfgVar.set(guidance.imgCfg.map(_.toString).getOrElse(""))
    distilledGuidanceVar.set(guidance.distilledGuidance.toString)
    val slg = guidance.slg.getOrElse(SlgParameters())
    slgLayersVar.set(slg.layers.mkString(","))
    slgLayerStartVar.set(slg.layerStart.toString)
    slgLayerEndVar.set(slg.layerEnd.toString)
    slgScaleVar.set(slg.scale.toString)
  }

  /** A sampler or scheduler this model does not list falls back to "(model
    * default)" — the select could not show it anyway.
    */
  private def applySampling(
      capabilities: SessionCapabilities,
      sample: SampleParameters
  ): Unit = {
    val cleansed = cleanse(sample)
    // a recording's values are the user's: no LoRA's settings replace them
    touchSampling()
    stepsVar.set(cleansed.sampleSteps.toString)
    cfgVar.set(cleansed.guidance.txtCfg.toString)
    applyGuidance(cleansed.guidance)
    samplerVar.set(
      cleansed.sampleMethod.filter(capabilities.samplers.contains).getOrElse("")
    )
    schedulerVar.set(
      cleansed.scheduler.filter(capabilities.schedulers.contains).getOrElse("")
    )
    sigmasVar.set(GenerationFormState.sigmasText(cleansed.customSigmas))
    flowShiftVar.set(cleansed.flowShift.fold("")(_.toString))
    // Dropping one is the rule, not a fault — the select could not show it —
    // but doing it silently changes the recipe behind the user's back, and the
    // next run then differs from the version it came from.
    val dropped =
      cleansed.sampleMethod
        .filterNot(capabilities.samplers.contains)
        .map(name => s"sampler $name")
        .toList :::
        cleansed.scheduler
          .filterNot(capabilities.schedulers.contains)
          .map(name => s"scheduler $name")
          .toList
    if (dropped.nonEmpty)
      warnAboutRecipe(
        s"${dropped.mkString(", ")} — this model does not offer that, so it " +
          "runs on the model's own default."
      )
  }

  /** The high-noise expert's own fields, from a session's defaults or from a
    * recorded request being reused.
    */
  private def applyHighNoise(highNoise: SampleParameters): Unit = {
    val cleansed = cleanse(highNoise)
    highNoiseStepsVar.set(cleansed.sampleSteps.toString)
    highNoiseCfgVar.set(cleansed.guidance.txtCfg.toString)
  }

  private def applyImageParameters(
      capabilities: SessionCapabilities,
      p: ImageGenerationParameters
  ): Unit = {
    promptVar.set(p.prompt)
    negativePromptVar.set(p.negativePrompt)
    widthVar.set(p.width.toString)
    heightVar.set(p.height.toString)
    strengthVar.set(p.strength.toString)
    applySeed(p.seed)
    applySampling(capabilities, p.sampleParams)
    // No block recorded means the pass was off when the recipe ran, and
    // `seedModeFields` has just put the session's own default back in: without
    // clearing the box, turning hires or VAE tiling off would never survive
    // the recipe being reapplied, and the next run would differ from the
    // version it came from (François, 2026-09-10).
    p.hires match {
      case Some(hires) => applyHires(hires)
      case None        => hiresEnabledVar.set(false)
    }
    p.vaeTilingParams match {
      case Some(tiling) => applyVaeTiling(tiling)
      case None         => vaeTilingEnabledVar.set(false)
    }
    applyLoras(p.lora)
    // The recorded inputs are URLs of the persisted files; the request needs
    // their bytes back, so they are fetched into the pickers.
    initImageVar.set(None)
    refImagesVar.set(List.empty)
    loadRecordedMedia(p.initImage.toList)(images =>
      initImageVar.set(images.headOption)
    )
    loadRecordedMedia(p.refImages)(refImagesVar.set)
    maskImageVar.set(None)
    loadRecordedMedia(p.maskImage.toList)(images =>
      maskImageVar.set(images.headOption)
    )
    batchCountVar.set(p.batchCount.toString)
  }

  private def applyVideoParameters(
      capabilities: SessionCapabilities,
      p: VideoGenerationParameters
  ): Unit = {
    promptVar.set(p.prompt)
    negativePromptVar.set(p.negativePrompt)
    widthVar.set(p.width.toString)
    heightVar.set(p.height.toString)
    strengthVar.set(p.strength.toString)
    videoFramesVar.set(p.videoFrames.toString)
    fpsVar.set(p.fps.toString)
    applySeed(p.seed)
    applySampling(capabilities, p.sampleParams)
    p.highNoiseSampleParams.foreach(applyHighNoise)
    // As above: no block recorded means the tiling was off.
    p.vaeTilingParams match {
      case Some(tiling) => applyVaeTiling(tiling)
      case None         => vaeTilingEnabledVar.set(false)
    }
    applyLoras(p.lora)
    initImageVar.set(None)
    endImageVar.set(None)
    loadRecordedMedia(p.initImage.toList)(images =>
      initImageVar.set(images.headOption)
    )
    loadRecordedMedia(p.endImage.toList)(images =>
      endImageVar.set(images.headOption)
    )
    applyVideoInputs(p)
  }

  /** The video models' media inputs, fetched back like the images. Every Var is
    * cleared first, so a recipe without them leaves none behind; the loads land
    * later.
    */
  private def applyVideoInputs(p: VideoGenerationParameters): Unit = {
    referencesVar.set(List.empty)
    guidesVar.set(List.empty)
    controlVideoVar.set(None)
    controlMaskVar.set(None)
    sourceVideoVar.set(None)
    controlStrengthVar.set(p.controlStrength.fold("")(_.toString))
    controlStartVar.set(p.controlStart.fold("")(_.toString))
    controlEndVar.set(p.controlEnd.fold("")(_.toString))
    // References are read by position: one that failed to load shifts every
    // later one, which the notice has to say.
    loadRecordedMediaAligned(p.references) { loaded =>
      if (loaded.contains(None))
        warnAboutRecipe(
          "A reference could not be loaded; the later ones moved up a place."
        )
      referencesVar.set(loaded.flatten)
    }
    loadRecordedMediaAligned(p.guides.map(_.media)) { loaded =>
      if (loaded.contains(None))
        warnAboutRecipe("A guide could not be loaded and was left out.")
      guidesVar.set(p.guides.zip(loaded).collect { case (guide, Some(media)) =>
        guideInput(media, guide.frameIndex.toString)
      })
    }
    loadRecordedMedia(p.controlVideo.toList)(videos =>
      controlVideoVar.set(videos.headOption)
    )
    loadRecordedMedia(p.controlMask.toList)(masks =>
      controlMaskVar.set(masks.headOption)
    )
    loadRecordedMedia(p.sourceVideo.toList)(videos =>
      sourceVideoVar.set(videos.headOption)
    )
  }

  // ------------------------------------------------------------------ loras

  /** The selected LoRAs' sampling settings over the mode's defaults
    * (`specs/49-lora-sampling-settings.md`), into the fields that still hold a
    * default — and, when `force`d, into the ones the LoRAs set whatever they
    * hold, which makes those defaults again. With no setting a field goes back
    * to the session's own, so removing a LoRA takes its values away with it.
    *
    * Only for the mode the form was seeded for: asked for another — the mode a
    * form held before the seeding that is still under way — it would write that
    * mode's defaults, or drift's own where the session has none, over the ones
    * just laid down (a video form came up at 20 steps and CFG 7).
    */
  def applyLoraSampling(
      capabilities: SessionCapabilities,
      newMode: String,
      sampling: LoraSampling,
      force: Boolean
  ): Unit = if (seededFor(newMode)) {
    val defaults =
      capabilities.defaultsByMode.getOrElse(newMode, GenerationDefaults())
    val takesLoras = capabilities.featuresByMode
      .getOrElse(newMode, Map.empty)
      .getOrElse("lora", false)
    val layer = if (takesLoras) sampling else LoraSampling()
    val sample = layer.over(cleanse(defaults.sampleParams))
    if (force) untouch(layer.fields)
    def write(field: String, target: Var[String], value: String): Unit =
      if (!touched(field)) target.set(value)
    write("steps", stepsVar, sample.sampleSteps.toString)
    write("cfg", cfgVar, sample.guidance.txtCfg.toString)
    write("flowShift", flowShiftVar, sample.flowShift.fold("")(_.toString))
    write(
      "sigmas",
      sigmasVar,
      GenerationFormState.sigmasText(sample.customSigmas)
    )
    write(
      "sampler",
      samplerVar,
      sample.sampleMethod.filter(capabilities.samplers.contains).getOrElse("")
    )
    write(
      "scheduler",
      schedulerVar,
      sample.scheduler.filter(capabilities.schedulers.contains).getOrElse("")
    )
    write(
      "distilledGuidance",
      distilledGuidanceVar,
      sample.guidance.distilledGuidance.toString
    )
    defaults.highNoiseSampleParams.foreach { highNoise =>
      val expert = layer.overHighNoise(cleanse(highNoise))
      write("highNoiseSteps", highNoiseStepsVar, expert.sampleSteps.toString)
      write("highNoiseCfg", highNoiseCfgVar, expert.guidance.txtCfg.toString)
    }
  }

  /** The configuration's default LoRAs (`specs/28-configuration-loras.md`) as
    * the picker's whole selection, ids and strengths straight from the
    * configuration, so nothing waits for the collection; one no longer
    * installed shows as such in the picker and is not sent. Written whole and
    * never read back, like `applyLoras`.
    */
  def applyConfigurationLoras(): Unit = {
    val loras = configurationLoras()
    pendingLoraSelections.set(List.empty)
    selectedLoraIdsVar.set(loras.map(_.loraId))
    loraStrengthsVar.set(LoraPicker.strengthsOf(loras))
  }

  /** Lays a recipe's LoRAs over the picker — at once when the collection is
    * listed, else once it is. Every Var is written whole and never read back:
    * this runs inside the observer that seeds a new panel, and a Var set inside
    * an Airstream transaction still reads its old value until the transaction
    * ends. Reading the pending selections back right after setting them is how
    * a recipe's LoRAs stayed pending for good — absent from the picker, sent
    * with every generation, beyond the user's reach (François, 2026-09-12).
    */
  private def applyLoras(selections: List[LoraSelection]): Unit =
    loraService.loadedLorasNow match {
      case Some(loras) => resolveLoras(selections, loras)
      case None        =>
        selectedLoraIdsVar.set(List.empty)
        loraStrengthsVar.set(Map.empty)
        pendingLoraSelections.set(selections)
    }

  /** The LoRA a recorded `lora[].path` names. The exact path first, then the id
    * the path carries: `requestPathOf` is `<sfw|nsfw>/<loraId>/<fileName>`, so
    * flipping a LoRA's NSFW flag moves its folder and would orphan every recipe
    * that used it — matching on the id in the path survives that (François,
    * 2026-09-10), and so do recipes recorded while the path still began with
    * the architecture.
    */
  private def loraOf(loras: List[Lora], path: String): Option[Lora] =
    loras
      .find(lora => lora.files.exists(file => lora.requestPathOf(file) == path))
      .orElse(loras.find(lora => path.split('/').contains(lora.id)))

  /** Maps recorded `lora[].path`s back to installed LoRAs — a pair's two files
    * resolve to the one entity — and makes them the picker's whole selection. A
    * path no LoRA owns (uninstalled) is named in the notice and skipped.
    */
  private def resolveLoras(
      selections: List[LoraSelection],
      loras: List[Lora]
  ): Unit = {
    val resolved =
      selections.map(selection => selection -> loraOf(loras, selection.path))
    val selected = resolved
      .collect { case (selection, Some(lora)) =>
        lora.id -> selection.multiplier.toString
      }
      .distinctBy(_._1)
    pendingLoraSelections.set(List.empty)
    selectedLoraIdsVar.set(selected.map(_._1))
    loraStrengthsVar.set(selected.toMap)
    val unresolved = resolved.collect { case (selection, None) =>
      selection.path
    }
    if (unresolved.nonEmpty)
      warnAboutRecipe(
        "LoRA not installed here, dropped from the form: " +
          unresolved.mkString(", ") +
          ". Generating now records a recipe without it."
      )
  }

  /** A recipe held back for the collection resolves when the listing lands. */
  def resolvePendingLoras(collection: Option[List[Lora]]): Unit =
    collection.foreach { loras =>
      val pending = pendingLoraSelections.now()
      if (pending.nonEmpty) resolveLoras(pending, loras)
    }
}
