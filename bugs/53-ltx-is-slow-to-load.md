# Bug 53 — LTX 2.5 takes one to three minutes to load

**Status:** open (raised by François 2026-10-08); read in the code and the logs, not timed by part yet —
`LtxPipeline` now prints each part's share of the load (text encoder, transformer, video VAE, audio VAE)
**Severity:** medium (every launch, and every restart after a stop)
**Files:** `runner/src/drift/runner/models/WeightSource.scala`, `runner/src/drift/runner/formats/ComfyQuant.scala`,
`runner/src/drift/runner/diffusion/LtxPipeline.scala`

## What the logs say

The same checkpoint (REDGraft LTX 2.5, int8 ConvRot, 15.9 GB) loaded in 37.8, 54.3, 67.7, 79.5, 154.0 and
185.9 s. A spread of five times on one file says most of a slow load is not computation: it is the files read
from disk (42 GB mapped: the checkpoint, Gemma 24.5 GB, the VAEs) when the page cache does not hold them, and
memory pressure when something else is loaded.

## What the load does

- **the transformer is decoded on the processor** (`WeightSource.toBf16` → `ComfyQuant.int8`): 14.3 GB of int8
  read into Java arrays, scaled and un-rotated (the ConvRot Hadamard) row by row, 16 million values at a time,
  each chunk uploaded as F32 and converted to BF16 on the GPU — 57 GB of floats through `fromFloats` to make
  28.7 GB of BF16. Probably most of the 38 s floor;
- **the files are read from disk** when cold: at the disk's speed, 42 GB is 15 to 40 s by itself;
- the text encoder and the VAEs are mapped and registered in place (BF16 files): no decode.

## What can be done, by what it buys

1. **Decode on the GPU.** The checkpoint is already registered with the GPU, so a kernel can read the int8 in
   place and write the BF16 (scale, un-rotate, convert): no Java arrays, no F32 upload. Small (one kernel
   beside `convert`), keeps everything else as it is. Removes the processor's share of the load.
2. **Decode once, keep the result** as a BF16 file in drift's cache, used on later launches: the load becomes a
   mapping, and the weights are no longer held twice (bugs/52). Costs 29 GB of disk per checkpoint and makes the
   weights reclaimable page cache instead of GPU memory — the very thing bugs/52 suffers from — unless pinned.
3. **Run the int8 as it is**: W x = Q (H x), so rotate the activations and multiply by the stored int8 — no
   decode at all, 14 GB on the GPU instead of 29. The largest gain (load and memory) and the largest work: an
   int8 product as fast as the BF16 one today, LoRAs as a side path. To be measured before it is chosen.
   **François wants this one tested in the night of 2026-10-08** (plan to be agreed first).
   He holds little hope for a true int8 product (no native instructions for it on the GPU), and the runner
   agrees: from `GemmMinimumRows` rows it already dequantizes GGUF weights to F16 for hipBLAS. **The variant to
   try first:** keep the int8 on the GPU and decode each layer into one reusable scratch buffer just before
   its product (the kernel of option 1), then the same float product as today — one more pass over 14 GB of
   weights per step, to be measured against the 58 s steps; LoRAs on a layer decoded on the fly to be settled.
4. **Do not load what the first job will not need at once**: nothing to gain here — Gemma is needed for the
   prompt, the transformer right after.
