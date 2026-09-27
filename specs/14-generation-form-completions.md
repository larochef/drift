# 14 — Generation form completions

**Status:** done
**Depends on:** 08, 12

The generation form carries around 45 controls. This spec is the presentation
that keeps that usable — a short always-visible core and folded sections —
plus the last fields 08 plumbed but never surfaced: batches, mask, guidance
and SLG, and the high-noise pass of two-expert video models.

## What it does

- The core, always visible: prompt, negative prompt, Generate (directly under
  the prompts — that is where the loop happens), size, steps, CFG, high-noise
  steps and CFG when the session reports high-noise defaults (wan 2.2's two
  experts), frames on video, seed with Random, and the LoRA picker (09).
- Folded sections, each a `CollapsibleSection`: **Sampling** (sampler,
  scheduler), **Inputs** (init image, reference images for edit, end image on
  video, mask shown only once an init image is attached, strength),
  **Hires** (10), **VAE tiling** (10), **Guidance & SLG** (image CFG, blank
  = the model's own; distilled guidance; SLG layers, scale, start, end),
  **Batch** (count bounded to `limits.maxBatchCount`, format, compression;
  image mode only — the native video request has no batch).
- A closed section says so when it is doing something: Hires and VAE tiling
  when enabled, Inputs when an image is attached, Sampling only when sampler
  or scheduler differ from the session's defaults.
- Open/closed state is per user, not per project: it lives on
  `CollapsibleSection`'s companion, so it survives the panel being rebuilt for
  another configuration and a version being applied. It does not survive a
  browser reload.
- A batch's images lay out side by side in the result; the gallery shows one
  tile per output (12) and the workspace tiles per output (19).
- Guidance untouched sends exactly what the base carried: no SLG block is
  invented where the base had none, so a version is never appended for
  nothing (19).

## Shape

- `pages/generate/`: `GenerationForm` assembles `CoreFields` and the
  sections `SamplingSection`, `InputsSection`, `HiresSection`,
  `VaeTilingSection`, `GuidanceSection`, `BatchSection`; `GenerationFormState`
  holds the Vars, `RecipeSeeding` seeds them from capabilities or a recording,
  `GenerationSubmission` writes them back over the base.
- Shapes already in `shared/.../Generation.scala`: `batchCount`, `maskImage`,
  `GuidanceParameters(txtCfg, imgCfg, distilledGuidance, slg)`,
  `VideoGenerationParameters.highNoiseSampleParams`.
- The mask key is `mask_image` in `features_by_mode.img_gen`; video reports
  none. The backend externalizes the mask beside the outputs like every input.
- `ProjectManager.recipe` blanks `batchCount` beside the seed: four images in
  one run are four re-rolls of one recipe, not a new version.

## Notes

- Folded sections over a drawer or Basic/Advanced tabs: they extend what the
  panel already did for hires and VAE tiling, hide nothing behind a mode, and
  keep the whole recipe in one scroll at both widths (free play and the
  three-column workspace share the panel).
- Reuse (12) round-trips every field here because the form seeds from a full
  `ImageGenerationParameters` and submits over it as the base, never from
  individual Vars only.
- A batch runs a bit slower than n single runs (sd-cpp works through the
  images one after another inside one job) but needs nobody at the screen.
- Drawing a mask in-app is out of scope; the picker takes a file like every
  other image input.
