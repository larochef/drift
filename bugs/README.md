# drift — open bugs

One file, open bugs only, kept short, in the order of their numbers. A bug that is fixed and confirmed is removed
from here; so is one that is not drift's to fix. Code and specs do not name bugs by number: a comment says what
the code guards against. Next number: **6**.

## Open

### 1 — Redrawing a part already redrawn ruins the skin (medium)

The reported chain, all Qwen Image 2.1 at 0.4: SeedVR2 1k → 4k, full redraw, then a one-tile selection redrawn to remove
a line — the last pass made the skin much worse. Measured on a generated subject
(`~/dev/redraw-experiments/2026-10-05-edit`, `FINDINGS.md` §8):

- no redraw removes the line, at any strength, masked or not;
- the full redraw at 0.4 is what costs the texture of a SeedVR2 picture (grain 2.46 → 1.98; kept at 0.25);
- the redraw's paste moves the skin around the selection by 1.1–2.2 levels; with the ramp kept within 64 px
  (`TileBlending.paste`'s `reach`, which edit uses), 0.2–0.3.

To do:
1. **To decide:** the redraw's paste kept within 64 px instead of fading to the window's edge.
2. **"Upscale this selection again"** (designed, not built): on a picture that is an upscale, the window around
   the selection is cut from the picture it was upscaled from, upscaled by the same model, and the selection
   pasted within 64 px. Measured: the line is gone, the texture is the picture's, no join, 44 s.

### 2 — SeedVR2 leaves the GPU idle while the CPU works (low to medium, speed only)

Measured (`~/dev/redraw-experiments/2026-10-06-seam`): drift spends 6 s between two tiles (decode, checks,
colour match, paint) with the runner idle — about 5 % of a tiled upscale; inside a job the runner's own gaps
(PNG encode, VAE stitch, per-layer `hipFree`) cost 1 to 9 %.

To do:
1. drift: submit tile n + 1 as soon as tile n's answer is in and finish tile n in an ox fork — upscales only; a
   redraw or an edit cuts the next crop from the painted picture.
2. Runner: the PNG encode and the stitch off the worker thread; the next VAE tile cut while one runs; a buffer
   pool in the SeedVR2 VAE.

### 3 — Klein 9B with a 2048² reference fails on the runner (fixed 2026-10-12, to confirm)

`requirement failed: 83968 ints at once`: the positions of a 2048² picture with a 2048² reference are more ints
than the runner staged in one copy to the GPU. A write of any length now goes through the staging in several
copies (tested on the GPU). The same request completes in 158 s through drift and reads its reference (a
gradient came back as the watercolour asked for).

### 4 — LTX decode crawls on long videos (high for long videos)

A 10 s, 48 fps, 40 step video (473 frames of 512²) was not done an hour after its last step. The decode is not
computing: memory is oversubscribed, the kernel reclaims pages of the mapped weight files, each reclaim evicts
the GPU queues, and every `hipFree` in `LtxVideoVae.conv` waits for them. GPU at 1–3 %, memory pressure `full`
45 %, 60 GB on the GPU + 42 GB of mapped files on a 121 GB machine. The decode holds every frame at every stage
(about 12 GB for 473 frames).

Done: the decode line and bar for LTX; quantized weights read in place (no decoded copy, 10 GB on the GPU
instead of 51 for the transformer).

To do:
1. Measure the 17 GB of the 60 unaccounted for: `hipMemGetInfo` after the load, after the last step, at each
   decode stage.
2. Decode in a window of frames (the 3×3×3 convolutions need one frame each side), so the peak no longer grows
   with the length.
3. Reuse the buffers — done for every model at once (2026-10-11): `HipOps` keeps released buffers for the next
   allocation of their size and finds one by address. Same output to the byte on SeedVR2, decode 12 % faster with
   memory to spare; not yet run on a long LTX decode, where it should count most.
4. A memory warning when a video's own weights and decode exceed the machine (`MemoryHeadroom`).
5. Decode bars for Wan and MiniMax H3 (they have the line only).

Declined (2026-10-08): releasing Gemma between the prompt and the decode — every component stays loaded for the
next generation.

### 5 — LLaDA-Image, Nucleus and (under load) LTX 2.5 do not repeat a result for the same seed (medium for LLaDA)

Found 2026-10-11, checking the shared sampling loop: the same request three times in one session of the runner,
512², 4 steps, seed 7. Krea 2, FLUX.2 klein, Mage-Flow Turbo, Qwen Image 2.1, HiDream O1 and GRN give the same
file to the byte. LLaDA-Image Turbo gives three visibly different pictures (PSNR 22–26 dB between them), on the
code before the shared loop as on the code after, with the spare buffers on or off. Nucleus differs at rounding
level only (45–51 dB).

LTX 2.5 too, once (2026-10-12): the same request — 512², 25 frames, 8 steps with its turbo LoRA, seed 7 — in
five fresh sessions gave the same frames four times, across two builds, and other frames (32 dB) once: the first
LTX session after a 2048² FLUX.2 job on the same drift. Cause not found.

No unseeded random generator is in either family's code. A hypothesis that would fit all three: a buffer read
before anything wrote it — a spare buffer, or fresh device memory taken for zero — which shows as rare
differences tied to what ran before in the process or on the device. To look at: an operation whose result depends on the
order work finishes in (a sum over experts, a kernel that accumulates across threads), which LLaDA's semantic
tokens — an argmax over logits — would turn from a rounding difference into another picture; and what the two
families share that the six repeatable ones do not.

