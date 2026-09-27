# 28 — Default LoRAs on run configurations

**Status:** done in code, not yet run live
**Depends on:** 04, 09, 26, 27

Some checkpoints need a LoRA to behave — a turbo LoRA for a model that is not
distilled — and others must not get it. That knowledge is encoded on the run
configuration, so the generation form starts with the right LoRAs and the
jobs that have no form (PiD, redraw) apply them too.

## What it does

- The run configuration form and edit card show a "Default LoRAs" picker for
  every architecture (a chat model applies them per reply, `35`): search the architecture's installed LoRAs, add one
  (strength seeded from the LoRA's default), adjust, remove. The create form
  drops them when the architecture changes.
- A session's form starts with the configuration's LoRAs. They are defaults,
  not locks: one request can change a strength or drop one.
- A reused recipe keeps its own LoRAs (a turbo recipe must not get the turbo
  LoRA twice); a task reused on another configuration takes that
  configuration's defaults.
- PiD and redraw through the configuration apply them on every tile.
- Missing LoRAs are loud: the form marks a selected LoRA no longer installed;
  a tiled job is refused before it starts, naming the LoRA. The job log lists
  the LoRAs applied.
- The Architectures cards fold their LoRA list closed, "LoRAs (n)" in the
  header with any downloads in progress; "Add LoRA" opens it.

## Shape

- `RunConfiguration.loras: List[ConfiguredLora]`, `ConfiguredLora(loraId,
  strength)` in `shared/.../Api.scala`; existing configurations decode with
  none.
- `Lora.selections(strength): List[LoraSelection]` in `shared/.../Loras.scala`
  is the one mapping from a LoRA to request entries (every file, the path
  relative to `--lora-model-dir`, `isHighNoise` from the file's stage, so a
  Wan 2.2 pair sends both halves); the form, the configuration and the backend
  all use it.
- `components/LoraPicker`: the caller owns the selection (ids and
  strengths-as-typed); trigger words are clickable only where a prompt takes
  them. Used by `RunConfigurationForm`, `RunConfigurationEditCard` and the
  generation panel.
- `RecipeSeeding` fills the picker from the configuration's LoRAs when no
  recipe is reused; `Reuse.Full` of the same configuration replaces them with
  the recorded ones; `Reuse.Task` restores the defaults.
- `TiledJobs` resolves the configured LoRAs through `LoraManager` before a
  job is recorded. Redraw and PiD tiles send them as `lora`.
- `pages/architectures/LoraSection` folds with the generation form's
  collapsible-section pattern.

## Notes

- A ready session reused for tiles scanned `--lora-model-dir` at its own
  launch; a LoRA installed after that is not visible to it and the tile fails
  with sd-server's error.
- The sd-cli LoRA tag syntax is taken from upstream docs and not yet verified
  on a real PiD run.
- Out of scope on purpose: parameters a LoRA implies (a turbo LoRA's few steps
  and cfg 1 are the configuration's `overriddenParameters`, set beside it),
  LoRAs as a default layer on architectures or models, and launch-time
  `--lora` flags — drift sends LoRAs per request.
