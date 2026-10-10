# Bug 55 — Turbo LoRAs on the stock Qwen Image 2.1 give unfinished pictures

**Status:** fixed 2026-10-09 (raised by François the same day: "the image feels like if a few steps were
missing, with things like vertical lines, or poor details, some unfinished sensation")
**Severity:** high (the turbo LoRAs are how the model is used day to day)
**Files:** `runner/src/drift/runner/diffusion/Lora.scala`, `runner/src/drift/runner/diffusion/FlowSchedule.scala`,
`runner/src/drift/runner/diffusion/QwenImage21Pipeline.scala`, `runner/src/drift/runner/models/QwenImage21.scala`,
`backend/resources/reference/loras.json`, `backend/resources/migrations/015-qwen-turbo-lora-levels.json`

## What happened

Three causes, one per LoRA family, all on the drift runner with the official weights
(`~/dev/redraw-experiments/2026-10-09-qwen-turbo`, `FINDINGS.md`):

1. **Pruna 5-step ran at half strength.** The file is diffusers' own export: no `.alpha` tensors, and PEFT's
   settings in the safetensors metadata (`lora_adapter_metadata`: `lora_alpha` 128, `r` 64). The runner read
   only alpha tensors, so the update was applied at 1 instead of 2: smooth, plastic skin, no fine hair.
2. **alibaba-pai's Fun-Acc 4-step was not applied at all.** Its pairs are named `X.lora_down` / `X.lora_up`
   without `.weight`, and it brings four output layers, one a step. Every tensor was reported as "not a LoRA
   pair" in the runner's log — 528 warnings nobody reads — and the model ran four steps bare: a blur.
3. **Viggle's noise levels were taken as final.** Its page lists `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25` for
   diffusers' `sigmas=`, which the pipeline then shifts for the image's size (at 1024 × 1536: 1, 0.971,
   0.940, 0.869, 0.689, 0.425). Taken as written, or replaced by a flow shift of 3 (what its record held),
   the model is told there is less noise than there is: harsh, over-textured skin and speckled walls.

## Fix

(The picture without a LoRA was soft too, and striped: that one is the guidance, `bugs/49`.)

- `Lora.open` reads PEFT's alpha (per module too, and rank-stabilized), pairs without `.weight`, and whole
  weights beside the pairs; `QwenImage21` takes an output layer per step (`atStep`) and checks any other
  whole weight is its own.
- Qwen Image 2.1 shifts given levels for the image's size, unstretched; a flow shift of 1 beside them
  leaves them as written (Fun-Acc's are final).
- Migration 15 gives the three LoRAs their levels where they are installed (by checksum); the catalog's
  Viggle entry takes its nodes and v0.3 is offered.

A LoRA left unapplied is no longer only a line in the runner's log: the job carries it (`warnings`), the
generation keeps it (`Generation.warnings`) and the page says it above the picture.

## Still open

- Viggle's 4-step r64 file stays soft on every schedule tried; its page no longer documents it.
- sd-cpp takes every list as final: a Qwen Image 2.1 turbo LoRA's nodes are not shifted there.
