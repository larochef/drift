# 49 — Sampling settings on LoRAs

**Status:** done in code, not yet run live
**Depends on:** 09, 14, 28, 38, 42

A turbo LoRA only works with the sampling it was distilled for: few steps,
CFG 1, a schedule leaning toward the noisy end. On the model's own settings
its images break (Qwen Image 2.1 with Viggle's 6-step LoRA: clean at flow
shift 3 and CFG 1, debris and broken structures on the model's own schedule). Today those values are on the LoRA's page, in the user's memory,
and typed again on every project. They belong on the LoRA, as one more layer of
defaults: the architecture's, then the run configuration's, then the selected
LoRAs', then what the user types.

## What it does

- A LoRA carries **sampling settings**, each optional — a LoRA only touches
  what it declares, and most LoRAs (styles, characters) declare nothing:
  - **steps**, the count it was made for: a minimum, more is fine;
  - **CFG**;
  - **flow shift**;
  - **sigmas**, the exact noise levels when its page lists them; they
    replace the schedule, and are their own step count;
  - **sampler** and **scheduler**, for the LoRAs that need one;
  - **distilled guidance**, for the models that embed one;
  - **high-noise steps** and **high-noise CFG**, for a pair on a two-expert
    model (Wan 2.2): the pair is one LoRA, the flow shift is the pair's, and
    it takes no sigmas.
- The values are **typed by the user** on the LoRA's page, from wherever its
  author published them. drift reads none from Civitai or Hugging Face, where
  they are free text.
- The form's defaults are the **merge** of the session's (the architecture's
  parameters under the run configuration's) and the selected LoRAs' settings,
  each layer overriding the one before for the fields it declares. Selecting
  or removing a LoRA merges again: the fields the user has not changed take
  the new defaults, the ones they changed keep their value.
- The values are **defaults, not rules**: every field stays editable, and
  nothing is refused — a user may run a turbo LoRA at 30 steps and CFG 6 to
  see what happens. Under the picker the form says what each selected LoRA
  sets, and flags fewer steps than the LoRA's.
- A reused generation or a project version lays down its own values, which
  count as the user's: the LoRAs' settings do not replace them. **Apply the
  LoRA settings** puts them over whatever the fields hold.
- The request carries plain values: a generation's record is what ran, and
  reuse needs nothing new.
- Two selected LoRAs that set the same field: the last one selected
  overrides, like any later layer.
- A LoRA on a **run configuration** (28) applies its settings when the form
  is seeded, and to redraw and edit, the jobs that have no form (under the
  steps their request asks for). PiD keeps its own steps.
- The LoRA picker marks the LoRAs that carry settings.
- The **catalog**'s turbo LoRAs (09) come with their settings filled in, so
  installing one is enough: Viggle's 6-step LoRA for Qwen Image 2.1 (6 steps,
  CFG 1, flow shift 3), and the others the catalog lists. One installed
  before keeps the settings it has: none, until they are typed.

## Shape

- `Lora.sampling: LoraSampling` (shared), a record of options (`steps`, `cfg`,
  `flowShift`, `sigmas`, `sampler`, `scheduler`, `distilledGuidance`,
  `highNoiseSteps`, `highNoiseCfg`). It merges itself: over another
  `LoraSampling` (the later LoRA's over the earlier's, `LoraSampling.of`),
  over a session's `SampleParameters`, and over the high-noise expert's.
  Migration 009 adds the empty record to stored LoRAs.
- `LoraSamplingFields` on the LoRA's card, behind **⚙ Sampling**: a field a
  setting, saved as it is left; the high-noise ones on an architecture with a
  `--high-noise-diffusion-model` checkpoint, sigmas on the others.
- The generation form keeps the sampling fields the user has changed
  (`GenerationFormState.touch`, a plain set: it is read by the observer that
  writes the fields). Seeding a mode clears it, a recipe fills it.
  `RecipeSeeding.applyLoraSampling` writes the merge into the fields that are
  not in it; the panel calls it whenever the selection's settings, the mode or
  the session's defaults change. `LoraSamplingNote` is the line under the
  picker.
- `TiledJobs` hands the jobs the configuration's LoRAs with their merged
  sampling (`ConfiguredLoras`).
- Sigmas and the flow shift override reach the drift runner for Qwen Image
  2.1 only (42); the other families read the shift they already read and
  refuse sigmas. sd-cpp takes both for every model.
