# Bug 49 — The official Qwen Image 2.1 leaves vertical stripes every 8 px on some pictures

**Status:** fixed 2026-10-09 by the defaults (seen by François 2026-10-06 on the test subjects of the edit
study: "there is some kind of grid on them"; cause found with `bugs/55`)
**Severity:** medium (a picture that looks fine at a glance and carries a regular pattern: it spoils any upscale
or texture comparison made from it, and SeedVR2 enlarges it with the rest)
**Files:** unknown — `runner/src/drift/runner/diffusion/QwenImage21Pipeline.scala`,
`runner/src/drift/runner/models/QwenImage21Vae.scala` are where to look first

## What happened

Three 1024 × 1536 pictures made on the drift runner with the official weights
(`leejet/Qwen-Image-2.1-GGUF`, `qwen_image_2.1-Q4_K.gguf`, 25 steps, CFG 6, euler) show fine vertical lines,
most visible in hair and on plain walls: `room.png` (2026-10-05), and two `longhair.png` (2026-10-06).

Measured (`~/dev/redraw-experiments/2026-10-06-edit/common.py`, `stripes`): the spectrum of the picture minus
its 4 px mean, along x, has its strongest line at a period of exactly **8.0 px**, 6.7 to 17.5 times the median
of the spectrum. Two other pictures of the same model and settings (`terrace.png`, 1536 × 1024, and
`hands.png`, 1024 × 1536) have no such line, and neither have three 1024² pictures from the official Flux.2
Klein 9B (`hair11`–`hair13`).

## Not known

- Whether it is the Q4_K quantisation, the runner, or the model: the same seed and prompt on sd-cpp, and on
  the bf16 weights, would say. 8 px is the VAE's step (one latent column), so the decoder and the last
  denoising step are the first suspects.
- What separates the pictures that show it from those that do not (two portrait pictures do, one does not).

## Meanwhile

The edit study's subjects are generated on Flux.2 Klein 9B and checked with `stripes` before they are used.
The 2026-10-05 texture numbers measured on `room` (FINDINGS §4) carry this pattern in their source.

## Cause

Guidance. The model's own pipeline runs 40 steps with no classifier-free guidance (`true_cfg_scale` 1 by
default, and its model card passes none); drift seeded sd-cpp's documented example instead, 25 steps at CFG 6.
At CFG 6 the picture is over-saturated, its detail smeared, and the 8 px line stands 15 times above the
spectrum's median, on the Q4_K and on the bf16 weights alike (so it is not the quantisation), on the runner's
plain `uncond + scale × (cond − uncond)`, which is the official pipeline's. The same seed and prompt at CFG 1:
2.4 at 40 steps, 2.2 at 25 — nothing. The VAE alone (a clean picture encoded and decoded) adds no such line.

Measured in `~/dev/redraw-experiments/2026-10-09-qwen-turbo` (`FINDINGS.md`), 1024 × 1536, seed 42:

| steps | CFG | 8 px line / median | |
| --- | --- | --- | --- |
| 25 | 6 | 15.4 (Q4_K), 15.3 (bf16) | soft, red, smeared |
| 40 | 1 | 2.4 | clean, natural |
| 25 | 1 | 2.2 | clean |
| 40 | 2 | 2.8 | clean, a little warmer |
| 40 | 3 | 3.5 | usable; skin smoother and redder |
| 40 | 4 | 4.5 | the pattern begins to show; redder still |

## Fix

The architecture's defaults are the pipeline's: 40 steps, CFG 1 (migration 15 reseeds it), and its prompting
notes say what a CFG above 1 costs: twice the time a step, and from about 4 up the colour and the pattern.
CFG 2 to 3 is the usable range for a negative prompt. A configuration that sets its own CFG keeps it.
