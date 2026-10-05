# Bug 38 — The runner spins a CPU core while the GPU works, and a PiD pass stalls the desktop

**Status:** fixed in code 2026-10-05, measured on a scratch build (Measured, below); whether the desktop still stalls during PiD is for François to say — it cannot be seen from a script
**Severity:** medium (no wrong result; one core at 100 % for the whole of a
generation, and during a whole-picture PiD pass the desktop lags so much that
keystrokes arrive several at a time)
**Files:** `runner/src/drift/runner/native/HipRuntime.scala` (`copy`, the
device set-up), `runner/src/drift/runner/ops/HipOps.scala` (`maxAbs`),
`runner/src/drift/runner/models/Pid.scala` (`fitValues`, `velocity`)

## Symptom

- While the runner generates, one thread (`image-worker`) uses 99 % of a core;
  the 31 others are idle. Its CPU time equals its wall time (1344 s of 1352 s
  sampled during a PiD pass).
- During PiD at 4096 × 6144 in one pass (≈ 6 min a picture), the desktop is
  barely responsive: typed characters show late, several at once. Qwen
  Image 2.1 sampling on the same machine does not do this to the same degree.

## What was seen

- Three stack samples two seconds apart, all in the same native call:
  `HipRuntime.copy` ← `HipOps.maxAbs` ← `Pid.fitValues` ← `Pid.velocity`. A
  device-to-host read waits for everything queued on the GPU, and HIP waits
  by spinning. No work is done on the CPU, so more threads would not make a
  generation faster.
- The stall is not measured. The likely cause: the desktop's compositor
  shares the GPU with the runner, and PiD's pixel-space layers at this size
  are long kernels the compositor's frames queue behind.

## To try

1. **Wait without spinning.** `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`
   before the context is created (or `hipDeviceScheduleYield`), or another
   primitive: an event polled with a short sleep (`hipEventQuery`), a stream
   callback. Measure the step time before and after on Qwen Image 2.1 and PiD:
   a blocking wait may cost a little on every sync.
2. **Fewer syncs in PiD.** `fitValues` reads a maximum back to the host on
   every call; keeping it on the device removes a sync per layer.
3. **Leave the desktop room during PiD.** Measure first (frame times with and
   without a run). Then: smaller kernels (tiles or rows of the pixel layers
   instead of the whole picture), or a yield between layers.

## SeedVR2: the GPU is not always full (François, 2026-10-04)

Sampled every 4 s during a SeedVR2 7B upscale of a 1024 × 1088 crop to
4096 × 4352 (162 s a tile), GPU load from `gpu_busy_percent` beside the
`image-worker` stack:

- **The VAE runs at 86–89 %, not 100 %.** In 17 of 24 samples the thread was
  in `HipRuntime.free` (← `HipOps.release` ← `SeedVr2Vae.causal`, `residual`,
  `VaeRun.attend`) or in `HipRuntime.allocate`: every layer allocates and
  frees its device buffers, and a free waits for the work that uses the
  buffer. A pool that hands buffers of the same size back out — or a release
  deferred to the end of a block — would remove a sync per layer.
- **Between tiles the GPU idles** (1 % and 33 % on two samples): the result is
  encoded to PNG and the next source decoded on the one worker thread. Small
  beside a tile, and the only CPU work that another thread could take.
- The two samples in `HipOps.launch` (`cacheWrite`, the transformer) were at
  100 %.

Not measured: how much time the pool would win. One tile, one run.

## Measured and changed, 2026-10-05

`~/dev/redraw-experiments/2026-10-04-auto` (`s9_runner_tests.py`, `s9b_pid_launches.py`), on a scratch copy of the
rebuilt runner.

- **The spinning core.** `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` after `hipSetDevice`: the same Qwen
  Image 2.1 tile (1280², 10 steps run) takes 115.3 s either way, with 3.7 s of CPU instead of 115.3 s. `yield` spins
  as much as the default (115.4 s of CPU). The runner now blocks; `DRIFT_HIP_SCHEDULE=spin` or `yield` brings the
  others back.
- **What stalls the desktop during PiD.** `DRIFT_HIP_TRACE=1` waits for every kernel as it is launched and prints,
  when the process ends, each kernel's launches, total, mean and longest time. A PiD tile of 1024 → 4096 (177 s):
  `attention_tiled_d64` was 56 launches of **2.1 s** each (119 s of the tile), `attention_tiled_d80` 8 of 1.9 s. A
  kernel is not interrupted, so every frame of the desktop waited up to two seconds behind one.
- **The fix.** The tiled attention kernel takes the workgroup it starts at (`first_group`), and `HipOps.attention`
  cuts its rows in launches of at most `TiledLaunchWork` (rows × keys × key heads): 3640 launches, the longest
  **44 ms** (57 ms for d80). The picture is the same to the bit; the tile takes 174.6 s against 169.7 s on the
  installed runner.
- **Left:** `convert` has one launch of about 300 ms in a PiD tile (the others are 7 ms); `cache_write` one of
  110 ms. Not cut.
- **SeedVR2** (a ×2 tile of 1024, 64 s traced): no kernel over 81 ms. Its time is `convert` (7805 launches, 16 s),
  `group_partials_f32` (13 s) and `im2col_3x3_bf16` (11 s) — the VAE's convolutions and group norms, where the
  buffer churn seen above also is. Nothing changed there.
- **Chat under the blocking wait** (`s10_chat_wait.py`): Qwen 3.8 27B, 300 tokens, twice each — 13.4 t/s blocking
  against 13.5 t/s spinning, 5.2 s of CPU against 24.6 s. The default holds for chat too.
