# Bug 52 — After the last LTX sampling step the video decode can crawl, silently, with the GPU idle

**Status:** open (raised by François 2026-10-08: a 10 s, 48 fps, 40 step LTX 2.5 video still not done after more
than an hour, the log stopped at 40/40, the GPU not fully used); cause measured on that run, nothing fixed
**Severity:** high for long videos (the decode can take as long as the sampling, and nothing says so)
**Files:** `runner/src/drift/runner/models/LtxVideoVae.scala`, `runner/src/drift/runner/diffusion/LtxPipeline.scala`,
`runner/src/drift/runner/models/WeightSource.scala`, `runner/src/drift/runner/native/RegisteredFile.scala`,
`runner/src/drift/runner/ops/HipOps.scala`

## What happens

Sampling ends on time (40 × 58.7 s ≈ 39 min, as estimated). What follows is `LtxVideoVae.decode`, which prints
nothing until its `decoded N frames in … s` line: the log and the bar stay at 40/40.

The decode itself is not computing, it is waiting. Machine memory is oversubscribed, the kernel reclaims pages
of the weight files, and each reclaim stops the GPU:

- the weight files are mapped and registered with the GPU (`RegisteredFile`: `hipHostRegister` on an `mmap`), not
  pinned (`Locked: 0 kB` in `smaps`). When the kernel takes one of their pages back, the driver evicts the
  process's GPU queues, then restores them by faulting the pages back in from disk;
- the decode allocates and frees GPU buffers per frame and per convolution (`conv`: `out` and `part`), and every
  `hipFree` waits for the queues to be restored. All thread dumps show the worker in
  `HipRuntime.free ← HipOps.release ← LtxVideoVae.conv`, asleep in the kernel in `kfd_wait_on_events`.

## Measured on the run (2026-10-08, 473 frames of 512 × 512, REDGraft LTX 2.5, 121 GB machine)

| what | value |
|---|---|
| GPU load, sampled each second | 1–3 % most seconds, short bursts to 30–95 % |
| queues evicted (`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`, summed over the process's 3 queues, so not wall time) | 1472 s by 12:47; over 12 one-second samples: +5.7 s and +5.4 s in two lumps, the GPU at 1–2 % between them |
| memory pressure (`/proc/pressure/memory`) | `full` 45 % |
| disk reads | 60 MB/s, 15–20 thousand file pages refaulted per second |
| runner CPU time | 5 min in 64 min |
| GPU memory of the runner (`drm-total-gtt`) | 59–60 GB |
| weight files in the page cache, mapped by the runner | 42 GB (`Pss_File` 44 GB) |
| free memory | 7 GB, 12 GB in zram |

Earlier decodes show the same dependence on memory: 313 frames took 336 s once and 1048 s another time, 153
frames 179 s and 266 s.

## Where the memory goes

- **the transformer twice**: REDGraft is a ComfyUI quantized checkpoint, decoded on the CPU at load and held on
  the GPU as BF16 (`WeightSource.toBf16`) — 29.9 GB, from the file's header: 14.3 GB of int8 and 0.6 GB of FP8,
  doubled — while its 15.9 GB file stays mapped and registered. Only 0.9 GB of it (BF16 and F32 tensors) is
  still read in place; the other 15 GB is never read again;
- **Gemma for the whole run**: the 24.5 GB text encoder file stays mapped and registered after the prompt is
  encoded, through the sampling and the decode;
- **the decode holds every frame at every stage**: for 473 frames, three sets of 473 × 128 × 128 × 128 floats in
  the last residual stage, about 12 GB — what tips a run that sampled fine over the edge.

60 GB on the GPU + 42 GB of mapped files + the desktop ≈ all of the 121 GB.

François read 43–44 GB for the runner in `amd-smi` while it sampled: the decode adds about 16 GB.

**Not accounted for:** 30 GB of transformer + 12 GB of decode leave about 17 GB of the 60 GB on the GPU
unexplained (LoRAs, the sampling's state, something not released?) — to be measured on the next run, by
printing `hipMemGetInfo` after the load, after the last step and at each decode stage.

## Fixes, by what they buy

1. **Say what is happening** — done 2026-10-08 for LTX 2.5: `Images.decoding` prints `sampling done,
   decoding … (VAE)` before every pipeline's decode, then `LtxVideoVae.decode` reports the share of its
   convolutions done, frame by frame, and `Images.decodeBar` prints it as a bar in hundredths (`34/100 -
   1.87%/s`), which drift shows as "decoding" (`ProgressKind.Decoding`). Each convolution is weighted by what it
   costs a pixel as measured (its product, plus gathering its inputs, worth a thousand outputs): on a 97-frame
   clip the bar is straight, a tenth every 1.2 to 1.4 s. Not seen in a browser; Wan's and MiniMax H3's decodes
   have the line but no bar yet.
2. **Stop holding the quantized file**: once a `ComfyQuant` / FP8 tensor is decoded to BF16, its file pages are
   dead weight. The file cannot simply be unregistered (0.9 GB of it is read in place): either copy those few
   tensors to the GPU too and unregister, or register only the ranges read in place. Frees nothing by itself,
   but those 15 GB can then be reclaimed without stopping the GPU.
   **Measured 2026-10-08 (`bugs/53`):** kept packed (`-Ddrift.comfyQuant=packed`), the weights are 20.9 GB
   instead of 41.1 for 0.6 s a forward pass, and a 257-frame decode took 33 s against 53 s (one run each).
   **2026-10-08, night:** the rotated weights are now read in place (`bugs/53`), which is this fix for them:
   no decoded copy beside the file. Written, to be compiled and measured.
3. **Not now (François, 2026-10-08):** he prefers every component kept in memory, so that the next generation
   does not reload Gemma, if reading the weights in place is enough. Kept as a note for machines short of
   memory: load each part only while it is needed.
   **Let Gemma go between the prompt and the decode**: unregister (or `madvise` away) the text encoder once the
   prompt is encoded, register it again for the next prompt (page-cache speed when memory allows).
4. **Decode in a window of frames** instead of all frames per stage: the 3 × 3 × 3 convolutions need one frame
   of context each side, so a stage can stream; peak memory then no longer grows with the length of the video.
5. **Reuse the buffers** in `conv` / `normSilu` (one `part` per stage, a pool of frames): thousands of
   `hipMalloc` / `hipFree` fewer — each one a kernel call, and the place the run blocks today. `HipOps.release`
   also searches `allocations` linearly.
6. **Warn**: `MemoryHeadroom` → `Session.memoryWarning` already exists for the assistant beside a video; the
   same warning fits a video whose own weights and decode exceed the machine.
