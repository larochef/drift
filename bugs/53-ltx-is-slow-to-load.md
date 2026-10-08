# Bug 53 — LTX 2.5 takes one to three minutes to load

**Status:** fixed in code for a warm page cache (2026-10-08): 49.9 s → 6.4 s, the same video bit for bit, the
weights read where the file holds them (no copy, 10 GB on the GPU instead of 51). A cold load still reads 42 GB
from the disk.
**Severity:** medium (every launch, and every restart after a stop)
**Files:** `runner/src/drift/runner/tensor/ConvRot.scala`, `runner/kernels/convrot.hip`,
`runner/src/drift/runner/models/WeightSource.scala`, `runner/src/drift/runner/formats/ComfyQuant.scala`,
`runner/src/drift/runner/ops/Ops.scala`, `HipOps.scala`, `CpuOps.scala`,
`runner/src/drift/runner/models/LtxText.scala`, `runner/src/drift/runner/diffusion/LtxPipeline.scala`

## What the logs said

The same checkpoint (REDGraft LTX 2.5, int8 ConvRot, 15.9 GB) loaded in 37.8, 54.3, 67.7, 79.5, 154.0 and
185.9 s. A spread of five times on one file says most of a slow load is not computation: it is the files read
from disk (42 GB mapped: the checkpoint, Gemma 24.5 GB, the VAEs) when the page cache does not hold them, and
memory pressure when something else is loaded.

## What the load was made of

Timed by part, page cache warm (`LtxPipeline` prints each part): 49.9 s, of which **35.5 s the transformer**,
decoded on the processor (`ComfyQuant` → `WeightSource.toBf16`), and 3.3 s the text projections, regrouped one
value at a time. The checkpoint holds 20.5 G values in 1440 quantized linears (831 int8, 609 4-bit), all
rotated: 41.1 GB once decoded to BF16, not the 29 first written here.

## What was built and measured (2026-10-08, evening)

- **The decode on the GPU** (`kernels/convrot.hip`), bit for bit the processor's decode. First from a packed copy
  of the codes on the GPU (a group of 256 columns: its row's scale and 256 int8, the 4-bit codes made int8).
- **The weights kept undecoded**, each layer decoded into the GEMM's scratch buffer before its product, as GGUF
  weights are dequantized; or decoded at load (`ComfyQuantStorage`, `-Ddrift.comfyQuant`).
- **The text projections regrouped in parallel**: 3.3 s → 0.8 s.
- The load's log lists every part (model recognized, tokenizer, text encoder's file, text features).

One clip, seed 42, 512², 8 steps, page cache warm; every row gives the same video and soundtrack, bit for bit
(also with a LoRA, the turbo distill at rank 256, applied beside the weights at run time).

| weights held                       | transformer | whole load | a step, 97 frames | a step, 257 frames | GPU memory, peak |
| ---------------------------------- | ----------- | ---------- | ----------------- | ------------------ | ---------------- |
| decoded on the processor (before)  | 35.5 s      | 49.9 s     | 7.87 s            | not run            | not sampled      |
| decoded by the GPU at load         | 2.5 s       | 8.5 s      | 7.87 s            | 21.40 s            | 50.7 / 56.5 GB   |
| packed, first decode kernel        | 1.1 s       | 9.6 s (\*) | 8.50 s            | 22.02 s            | 33.5 / 39.1 GB   |
| packed, second decode kernel       | 1.1 s       | 7.2 s      | 8.01 s            | not run            | 33.2 GB          |

(\*) before the text projections' fix.

- **The first kernel** rotated one value a thread through shared memory, five synchronizations a group: every
  layer in 0.81 s, 76 GB/s moved where the GPU reads 237 (`MemoryBandwidth`). 0.62 s a forward pass in a run.
- **The second** has sixteen threads a group, each rotating sixteen values in registers twice around one
  exchange: every layer in **0.26 s**, the memory's speed for 21 GB read and 41 GB written, **0.14 s a pass** in
  a run (1.8 % at 97 frames).
- Packed holds 20.9 GB of weights instead of 41.1. The 257-frame video decode took 33 s packed against 53 s
  decoded (one run each, the direction `bugs/52` predicts).
- François chose the undecoded weights as the default (2026-10-08).

## Read in place (2026-10-08, night)

François, on the packed default: the load is fast but the memory still high — 34 GB on the GPU and 44.5 GB of
mapped files resident in the runner, the checkpoint's 15.9 GB among them beside their own packed copy
(20.9 GB). The decode reads the file in place as fast as the copy, so the copy went: the weights are the
file's codes (`ConvRot.Codes`, `ConvRot.Nibbles`), the backend holds their scales (`Ops.rotated`), and two
kernels decode them — the 4-bit one from the nibbles, which no longer become int8. The packed layout and its
kernels are removed. `-Ddrift.comfyQuant=inplace` (the default), `gpu`, `cpu`.

Measured, the same clip (97 frames, 8 steps), the same video and soundtrack bit for bit:

| weights held                | transformer | whole load | a step | GPU memory allocated, peak |
| --------------------------- | ----------- | ---------- | ------ | -------------------------- |
| decoded by the GPU at load  | 2.5 s       | 8.5 s      | 7.87 s | 50.7 GB                    |
| packed copy                 | 1.1 s       | 7.2 s      | 8.01 s | 33.2 GB                    |
| **read in place**           | **0.4 s**   | **6.4 s**  | 7.98 s | **10.3 GB**                |

- Both decodes are the CPU's bit for bit on the smallest and the largest layer of each kind; 0.94 ms for
  4096 × 16384 int8, 0.87 ms 4-bit; every layer in 0.25 s (`ConvRotBenchmark`). Nothing is allocated for the
  weights.
- Decoding at load from the file (`gpu`) now takes 0.9 s for the whole transformer.
- The transformer now costs its file's 15.9 GB of page cache and nothing else, against 36.8 GB packed and
  57 GB decoded.

**An int8 ConvRot text encoder (2026-10-09).** François launched LTX
with the int8 Gemma (13.2 GB against 24.5): `gather_convrot_i8 … named symbol not found`. Its token table is a
rotated int8 weight too, and only a product decodes codes in place; and its two text projections, regrouped
on the host at load as stored bytes, cannot be regrouped rotated (such an encoder had never run). Now a weight
is read in place only where a model asks for a linear's (`WeightSource.linear`: the LTX transformer, its text
connectors, Gemma's layers); asked any other way it comes decoded, on the GPU, as before the night. The text
projections of a quantized encoder are decoded on the host (`WeightSource.hostRows`), regrouped and kept in
BF16. Run: the load 7.0 s (text features 1.4 s), 8.00 s a step, 11.9 GB allocated on the GPU at the peak
(the token table decoded, 2 GB), the prompt followed; with the BF16 encoder the reference clip is unchanged
bit for bit.

**Its risk:** the weights are file pages, which the kernel may drop under pressure and read again
(`bugs/52`); to lock if it happens. Not run on a long clip, nor with a LoRA, nor from drift.

## What a true int8 product would add

hipBLAS has one (`hipblasGemmEx`, int8 × int8 → int32), it is right (checked on known sums) and as fast as the
BF16 product on gfx1151 — 31 TFLOPS at LTX's shapes, 27 against 18 when the inputs are 16384 wide
(`GemmBenchmark`). But it needs the activations rotated and rounded to 8 bits at every product, and the result
is no longer the float one. Against 0.14 s a pass it has little left to save. Not built.

## What is left

- **A cold load** reads the files at the disk's speed; nothing here changes that.
- The rest of the load: the video VAE 2.6 s, the tokenizer 0.8 s, recognizing the model 0.8 s (the checkpoint
  is opened three times before it is loaded).
- **The installed runner**: these numbers are the build in the repository; a launch from drift shows them once
  the runner is installed again, and takes the default (the property does not reach it).
- Loading each component only while it is needed (Gemma for the prompt): declined for now, see `bugs/52`.
