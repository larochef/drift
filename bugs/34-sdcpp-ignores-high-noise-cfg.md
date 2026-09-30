# Bug 34 — sd-cpp ignores Wan 2.2's high-noise CFG when the low-noise CFG is 1

**Status:** open (upstream, sd-cpp; found 2026-09-30 while benchmarking the
drift runner, `specs/42` step 14)
**Severity:** medium (silently different videos from what the configuration
says; with drift's Wan 2.2 14B defaults the high-noise expert never runs CFG)
**Files:** sd-cpp `src/pipeline/video.cpp` (~line 1192, `if (request.use_uncond)`),
drift's seed `backend/resources/reference/architectures.json` (`wan-2.2-14B`:
`--high-noise-cfg-scale 3.5`, `--cfg-scale 1.0`)

## Symptom

`sd-cli -M vid_gen` on Wan 2.2 A14B with `--cfg-scale 1.0
--high-noise-cfg-scale 3.5` (drift's defaults) runs the high-noise steps at the
cost of one pass each (63 s, like the low-noise steps), and its log shows one
text encoding only. The drift runner, which applies 3.5, spends two passes per
high-noise step and, on a step-distilled expert (Civitai's FASTMOVE V2),
gives an overcooked, overexposed video; at 1 it matches sd-cpp's look.

## Root cause

sd-cpp encodes the negative prompt only when the low-noise guidance needs it:

```cpp
if (request.use_uncond) {
    embeds.uncond = sd->cond_stage_model->get_learned_condition(...);
```

and hands the high-noise sampler
`request.use_high_noise_uncond ? embeds.uncond : SDCondition()`: an empty
condition, so no CFG, whatever `use_high_noise_uncond` says.

## Suggested fix

Upstream: encode the negative prompt when `use_uncond || use_high_noise_uncond`.
In drift: nothing to change in the runner (it follows the flags). Decide
whether the seed's `--high-noise-cfg-scale 3.5` stays (right for the base
A14B experts, wrong for step-distilled finetunes, and a no-op on sd-cpp
today); the user guide already says to use 1 for distilled experts.

## Verification

On sd-cpp: the log shows two `computing condition graph` lines and the
high-noise steps take twice the low-noise ones.
