package drift.frontend.pages.generate

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the generation form holds (`specs/08-inference-ui.md`, `specs/14`):
  * every field, the attached images, the LoRA selection, the recipe a reuse
  * laid down and what it could not carry. The panel owns one; seeding writes
  * it, the sections edit it, the submit reads it.
  *
  * Numeric fields stay strings until submit: a half-typed number must not fight
  * the input, and the defaults fill the gaps anyway.
  */
class GenerationFormState {
  val mode = Var("img_gen")
  val promptVar = Var("")
  val negativePromptVar = Var("")
  val widthVar = Var("")
  val heightVar = Var("")
  val stepsVar = Var("")

  /** wan 2.2 and its kind run two experts, and the high-noise pass takes its
    * own step count and CFG (`--high-noise-steps`, `--high-noise-cfg-scale`).
    * Shown only when the session reports high-noise defaults, which is how a
    * two-expert model announces itself (François, 2026-09-09).
    */
  val highNoiseStepsVar = Var("")
  val highNoiseCfgVar = Var("")
  val cfgVar = Var("")
  val samplerVar = Var("")
  val schedulerVar = Var("")
  val seedVar = Var("")
  val randomSeedVar = Var(true)
  val videoFramesVar = Var("")
  val fpsVar = Var("")
  val strengthVar = Var("")

  // Highres fix (`specs/10-generation-time-upscaling.md`) — img_gen only. Target
  // dimensions stay empty by default: empty means "use the scale factor".
  val hiresEnabledVar = Var(false)
  val hiresUpscalerVar = Var("")
  val hiresScaleVar = Var("")
  val hiresTargetWidthVar = Var("")
  val hiresTargetHeightVar = Var("")
  val hiresStepsVar = Var("")
  val hiresDenoisingStrengthVar = Var("")
  val hiresUpscaleTileSizeVar = Var("")

  // VAE tiling (`specs/10-generation-time-upscaling.md`) — decode the VAE in tiles,
  // which is what keeps a large hires pass from aborting sd-server on ROCm.
  val vaeTilingEnabledVar = Var(false)
  val vaeTileSizeXVar = Var("")
  val vaeTileSizeYVar = Var("")
  val vaeTargetOverlapVar = Var("")
  val vaeRelSizeXVar = Var("")
  val vaeRelSizeYVar = Var("")
  val vaeTemporalTilingVar = Var(false)
  val vaeExtraTilingArgsVar = Var("")

  // Attached images, as data URLs. The init image is shared between modes on
  // purpose: an img2img source is usually also the img2vid source.
  val initImageVar = Var(Option.empty[String])
  val endImageVar = Var(Option.empty[String])
  val refImagesVar = Var(List.empty[String])

  /** The inpaint mask (`specs/14`): meaningful only beside an init image, so it
    * is offered, and sent, only when one is attached.
    */
  val maskImageVar = Var(Option.empty[String])

  /** How many images one run makes (`specs/14`) — the same recipe on that many
    * seeds. img_gen only; the native video request has no batch count.
    */
  val batchCountVar = Var("")

  // The rest of guidance (`specs/14`), seeded from the session's defaults or
  // a recording and written back over the base, so leaving them untouched
  // sends exactly what the base already carried. A blank image CFG means the
  // model's own; SLG layers are typed as a comma-separated list.
  val imgCfgVar = Var("")
  val distilledGuidanceVar = Var("")
  val slgLayersVar = Var("")
  val slgLayerStartVar = Var("")
  val slgLayerEndVar = Var("")
  val slgScaleVar = Var("")

  /** Selected LoRA ids and their strengths-as-typed, in two Vars on purpose: a
    * strength keystroke must not rebuild the row list (and steal the input's
    * focus), so the rows re-render on id changes only.
    */
  val selectedLoraIdsVar = Var(List.empty[String])
  val loraStrengthsVar = Var(Map.empty[String, String])
  // Default checked, per François: the checkbox exists to hide, not to gate.
  val includeNsfwLorasVar = Var(true)

  // "Reuse these parameters" (`specs/12-gallery.md`): the recorded request
  // becomes the base the submit copies the form over, so fields the form never
  // exposes (SLG, eta, flow shift, custom sigmas, …) round-trip too.
  val reusedImageBase = Var(Option.empty[ImageGenerationParameters])
  val reusedVideoBase = Var(Option.empty[VideoGenerationParameters])
  val pendingLoraSelections = Var(List.empty[LoraSelection])

  /** The reuse notice in two parts: what was seeded, and what could not be. A
    * warning is raised while the recipe is being applied — before the headline
    * is set, and sometimes long after, when the LoRA collection finally arrives
    * — so the element composes the two instead of either overwriting the other,
    * which is how a skipped LoRA used to vanish unannounced (François,
    * 2026-09-10).
    */
  val reuseNoticeText = Var(Option.empty[String])
  val reuseWarnings = Var(List.empty[String])

  def warnAboutRecipe(text: String): Unit =
    reuseWarnings.update(list =>
      if (list.contains(text)) list else list :+ text
    )
}

object GenerationFormState {

  /** The capabilities report the literal string "default" for an unset sampler
    * or scheduler; sending it back is not worth trusting, so it maps to "omit
    * the field" everywhere.
    */
  def cleanse(parameters: SampleParameters): SampleParameters =
    parameters.copy(
      scheduler = parameters.scheduler.filter(_ != "default"),
      sampleMethod = parameters.sampleMethod.filter(_ != "default")
    )

  def modeLabel(m: String): String = m match {
    case "img_gen" => "Image"
    case "vid_gen" => "Video"
    case other     => other
  }
}
