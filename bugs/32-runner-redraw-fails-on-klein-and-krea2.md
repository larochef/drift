# Bug 32 — Redraw on the drift runner fails for FLUX.2 [klein] and Krea 2

**Status:** fixed 2026-09-26 (not committed): every Klein 9B, Krea 2 and Qwen Image 2.1 configuration loads, generates, modifies an image and redraws on the runner, through drift
**Severity:** high (redraw on the runner works only with Qwen Image 2.1)
**Files:** `runner/src/drift/runner/models/WeightNames.scala` (new), `WeightSource.scala`, `Flux2.scala`, `QwenImage21.scala`, `Pid.scala`, `runner/src/drift/runner/diffusion/{ImagePipeline,Krea2Pipeline,Lora}.scala`, `shared/src/drift/shared/RuntimeRules.scala`, `backend/resources/reference/runtime-rules.json`, `backend/src/drift/backend/session/{JobServers,LaunchArguments}.scala`, `backend/src/drift/backend/sdserver/NativeJobs.scala`, `backend/src/drift/backend/postprocess/{TileRun,PostProcessJobs}.scala`

## Symptom

Redraw jobs (spec 27) on `drift-runner-images` fail on every Klein and Krea 2
configuration François tried. Qwen Image 2.1 works. Some failures are
refusals with a reason; one looked like an app crash. These are four separate
failures, all read from `~/.cache/drift/logs/postprocess-<job>.log`:

1. **Klein: `error: unknown flag --guidance`** (jobs …-1004, …-1007 to
   …-1009). The configuration launches with `--guidance 3.0`. The runner
   refuses any flag it doesn't know (`ImageOptions`), and `--guidance` isn't
   among the flags it accepts or ignores.
2. **Klein: `NoSuchElementException: the weights have no tensor
   encoder.mid.block_1.norm1.weight`** (jobs …-1011 to …-1013, once
   `--guidance` was gone). The VAE assigned is
   `Comfy-Org/ERNIE-Image vae/flux2-vae.safetensors`. That file uses
   **diffusers names** (`encoder.mid_block.attentions.0.…`, 251 tensors, with
   `bn.running_mean`). `FluxVae` reads only the BFL/LDM names
   (`encoder.mid.block_1`), and it fails at load (`ImageMain` →
   `VaeWeights`). This is the "nothing simple" failure François got after
   removing `--guidance`: the next thing the runner reads fails. This is not a regression from the `Flux2Vae` → `FluxVae`
   rename: neither version handles diffusers names. A Klein configuration with
   this VAE can't start on the runner at all, txt2img included.
3. **Krea 2 (FinePorn v4 bf16): `FormatException: … is no Krea 2 checkpoint
   (no txtfusion.projector.weight)`** (job …-1003), raised by `Krea2.open`.
   The same checkpoint, like the Klein one, **loads and generates with
   sd-cpp** (François). So the runner is stricter or narrower than sd-cpp's
   loader, and the runner is what has to change.
   This fine-tune's safetensors are laid out or named differently from the
   checkpoint the runner was built against. It could be a prefix
   (`model.diffusion_model.`), a Comfy export, or a split text fusion.
   **Unverified:** the header hasn't been compared yet.
4. **Krea 2 (Muse v3.5 Q8 GGUF): the job stops after "tile 1 of 16" with no
   error** (job …-1005). The model loads ("Krea 2 loaded in 4.6 s"), the first
   1216×1216 tile's img_gen is submitted, and the log ends there. There's no
   JVM crash dump and no kernel OOM or GPU fault in `journalctl` around it.
   This is probably the "app crash" François saw. **Unknown:** whether the
   runner process died (the failure would then be the tile's img2img path,
   1216² tiles, or the GGUF Q8 weights) or whether drift's side stopped
   polling.

## Suggested fix

0. **First, plain generation.** Before touching redraw, make each of these
   configurations generate an image (txt2img) on the runner: Klein
   PornMaster v4 Turbo with the ERNIE flux2 VAE, Krea 2 FinePorn v4 bf16,
   Krea 2 Muse v3.5 Q8. Failures 1 to 3 break loading, so they break every
   job, not only redraw. Redraw (img2img per tile) comes after.
1. **Flags the runner can't take:** reuse `RuntimeRule`
   (`shared/.../RuntimeRules.scala`, spec 16). It already drops flags a build
   refuses, flash attention on Vulkan for instance, and turns each drop into a
   `ResolutionNote`. Don't widen the runner's own flag parser, and don't make
   drift strip flags ad hoc. Rules are keyed by tool and backend today. The
   runner is an engine of its own (spec 43, `RuntimeEngine.DriftRunner`) on
   the same tool and backend as sd-cpp ROCm, so a rule needs to be able to
   name the engine. Then the runner's rule lists `--guidance` and any other
   sd-server flag a configuration may carry that the runner doesn't
   implement. Keep the runner refusing unknown flags: a new sd-cpp flag then
   shows up as a missing rule, instead of being silently ignored.
2. **Loading like sd-cpp.** sd-cpp loads both failing checkpoints, so read
   how it does: its name conversion (`model.cpp`: the diffusers ↔ BFL/LDM
   maps and the prefixes it strips) and how it detects a Krea 2 checkpoint.
   Match that at load, as one name map in `WeightSource`/`VaeWeights` rather
   than a second code path per model:
   - `FluxVae`: diffusers-named FLUX VAEs (`encoder.mid_block.…`). Check the
     other VAEs the runner loads for the same gap.
   - `Krea2.open`: compare FinePorn's header with a checkpoint that works,
     find what sd-cpp maps or tolerates, and do the same.
3. **The silent stop.** Reproduce the Muse redraw with the runner's log on.
   If the runner dies, fix the cause. Either way, make drift fail the job
   naming the tile and the runner's exit code, so it can't stop silently:
   `TiledJobs.runTiles` should notice a dead job server.

## Verification

- Each configuration first generates an image on the runner. Then a redraw
  of the same source completes on the runner with each of: Klein
  PornMaster v4 Turbo (ERNIE flux2 VAE, `--guidance 3.0`), Krea 2 FinePorn v4
  bf16, and Krea 2 Muse v3.5 Q8.
- A job whose server dies fails with a reason, and drift stays up.

## What was done (2026-09-26)

Root causes, as found:

1. **`--guidance`** is not a runner flag, and none of the runner's models has
   a guidance embedding (Klein's checkpoints have no `guidance_in`).
   `RuntimeRule` now names its engine (`engine: RuntimeEngine`, matched by
   `RuntimeRule.forRuntime` in the launcher and the form's preview). A new
   rule for the drift runner (sd-cpp tool, ROCm) drops `--guidance` and
   `--attn-scale` with a note. The runner still refuses unknown flags.
   François's three Klein configurations no longer need their
   `--guidance` removal (left in place in his config).
2. **ERNIE-Image's `flux2-vae`** is diffusers-named. `WeightNames` maps
   names once, at load, inside `WeightSource`, as sd-cpp's
   `name_conversion.cpp` does: ComfyUI's prefixes are dropped
   (`model.diffusion_model.`, `diffusion_model.`, `net.`,
   `first_stage_model.`, `vae.`), and a diffusers `AutoencoderKL` gets the
   LDM names. The per-model prefix code in `Flux2`, `QwenImage21`, `Pid`,
   `ImagePipeline` and `Lora` is gone.
3. **FinePorn v4 (and WinPic Asian)** store Krea 2 under
   `model.diffusion_model.`; `Krea2.open` read bare names only. Fixed by 2.
4. **The Muse "silent stop"**: the runner's Krea 2 took no init image, so
   it answered the first tile's img2img with a 400. drift failed the job
   with that reason, but the reason never reached the job's log, which
   just ended. Krea 2 now does img2img (Wan VAE encoder, packed 2 × 2,
   mixed at `steps − ⌊steps × strength⌋`). A failed job's log now ends with
   `# failed: <reason>`. A tile waiting on a job server that dies fails
   within a second with `tile N of M: the server exited with code X` (or `was killed`, with the signal)
   (`JobServer.exitCode` polled by `NativeJobs.await`), instead of after
   five failed polls.

Checked on the runner (1024², 4 steps, `ImageDemo`) and through drift
(isolated stage: session txt2img, then a 2048² redraw in four 1152 tiles on
`drift-runner-images`). Results are in the report of 2026-09-26.
