# Bug 49 — The official Qwen Image 2.1 leaves vertical stripes every 8 px on some pictures

**Status:** open (seen by François 2026-10-06 on the test subjects of the edit study: "there is some kind of grid on
them"; measured the same night, cause not looked for)
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
