# 17 — krea2 launch segfaults instantly with an empty log

**Status:** root cause found — it is the **ROCm runtime**, not the launcher.
The Vulkan runtime runs the *same* configuration to `ready`. Two launcher
fixes fell out of the investigation (snapshot paths, cwd) and are done.
**Severity:** high for the ROCm runtime, which is unusable on this machine
until the TheRock pairing is sorted; drift itself can run krea2 today by
picking the Vulkan runtime.

## Root cause (2026-09-02)

The `master-841-6b3edaa-rocm` runtime segfaults in **HIP backend init**,
before any model is touched:

- Under drift's TheRock `LD_LIBRARY_PATH`
  (`therock/gfx1151-7.14.0a20260612/lib`), `rocminfo` enumerates **zero
  agents** — the ROCm stack in that TheRock build cannot see the GPU on this
  machine. HIP init then segfaults, which is why sd-server dies before its
  first line (the Vulkan build's first line is "Found 1 Vulkan devices"; the
  ROCm build never gets there).
- The GPU is fine: system ROCm 7.2.4's `rocminfo` sees `gfx1151`
  (AMD Radeon 8060S, Ryzen AI MAX+ 395). `HSA_OVERRIDE_GFX_VERSION=11.5.1`
  (from the login env) does not help.
- **The Vulkan runtime `master-841-6b3edaa-vulkan` runs krea2 to `ready`** on
  the same weights — verified through drift end to end, 29GB into VRAM,
  listening. So the models, the argv, the launcher and the machine are all
  good; only the ROCm/TheRock pairing is broken.

The likely culprit is the **TheRock version drift paired**. drift matched the
sd-cpp asset's declared ROCm `7.14.0` to the newest bucket build with that
triple (`7.14.0a20260612`). The user's known-good lemonade script pairs a
`rocm-stable` sd-cpp build with TheRock **7.13.0** instead. So drift's
version-triple match resolved a TheRock build that does not work on gfx1151
here, where 7.13.0 does.

### Suggested fix (runtime provisioning, spec 06)

- Let a runtime pair against a *chosen* / older TheRock version, not only the
  newest matching the triple — or fall back when the newest fails validation.
  Validation today runs `sd-server --help`, which does **not** touch HIP, so a
  runtime whose HIP init segfaults still validates as good. Consider a deeper
  validation that actually initialises the backend (a tiny generate, or a
  `rocminfo`-under-the-env agent check) so a broken ROCm pairing is caught at
  install, not first launch.
- Or adopt lemonade's already-working directories as a runtime (spec 06 already
  supports adopting an existing directory).

## Launcher fixes that came out of this (done 2026-09-02)

- **Snapshot paths over blob paths.** Two assigned models are `Local` sources
  pointing straight at HuggingFace `blobs/<sha256>` files, which have no
  extension; sd-cpp sniffs the weight format from the extension. `ModelCache`
  now rewrites a blob path to the snapshot symlink that references it
  (`ModelCache.preferSnapshotPath`), so sd-cpp sees `turbo.safetensors` /
  `….Q8_0.gguf`. Confirmed in the Vulkan run: "using safetensors format",
  "using gguf format". (Not the segfault's cause — it crashed on snapshot
  paths too — but correct, and sd-cpp would need it once the ROCm side works.)
- **Working directory = the executable's own directory.** The replaced shell
  scripts ran there, and sd-server may resolve resources relative to itself.
- **Spawn header in the log.** Session id, cwd, every env var drift sets, and
  the full command line are written to the log before the process starts, so
  a silent crash still leaves the exact invocation behind (see below).

## Symptom (original)

## Symptom

Launching the `krea2-turbo` run configuration (rocm runtime
`master-841-6b3edaa-rocm`) produces a session that fails within a second:

```
"error": "sd-server exited with code 139"
```

and `~/.cache/drift/logs/<sessionId>.log` held **0 bytes**. 139 = 128 + SIGSEGV.

Since 2026-09-02 the launcher writes a spawn header into the session log
before the process starts — the session id, every environment variable drift
sets (`LD_LIBRARY_PATH`), and the full command line — so a silent crash still
leaves the exact invocation behind, in the log file and in the failure error
the card shows. Reproducing by hand is now copy-paste from either.

## What is already ruled out (2026-09-02)

Reproduced outside drift — same segfault, so this is not the launcher's spawn:

```
LD_LIBRARY_PATH=<therock gfx1151-7.14.0a20260612>/lib:<install>/bin \
  <install>/bin/sd-server --diffusion-model … --vae … --llm … \
  --cache-mode spectrum --cfg-scale 1.0 --diffusion-fa --mmap --steps 4 \
  -H 1024 -W 1024 --listen-ip 127.0.0.1 --listen-port 7861
# → exit 139, dumps core
```

- **Not stdout buffering.** Under `stdbuf -oL -eL` the process still prints
  nothing before the segfault; the empty log is honest.
- **Not the extensionless blob paths.** Two of the three assigned models are
  `Local` sources pointing at HuggingFace `blobs/<sha256>` files with no
  extension (`turbo`, `qwen3-vl-4b-…-q8-0`), which looked like a
  format-sniffing trap — but substituting the snapshot symlink paths
  (`…/snapshots/…/turbo.safetensors`, `….Q8_0.gguf`) segfaults identically.
- **Not a gfx mismatch.** The dev machine *is* a Strix Halo
  (`lspci`: Radeon 8050S/8060S; `/sys/class/kfd/.../gfx_target_version` =
  `110501` = gfx1151), matching the TheRock build.
- `sd-server --help` runs fine under the same environment (that is how the
  runtime validated), so the binary and libraries resolve; the crash is in
  startup proper — plausibly HIP/ROCm device initialisation.

## Next steps (owner: François offered context)

- Get a backtrace: `coredumpctl` should have the dump from the manual repro
  (2026-09-02 ~22:2x, `sd-server`, exit 139).
- Compare against the lemonade-managed sd-cpp on the same machine (does *its*
  sd-server start?), and against the `master-841-6b3edaa-vulkan` runtime —
  if vulkan comes up, this is ROCm/TheRock init, not the model set.
- Try `sd-cli --help`-level GPU touch (e.g. a 64x64 txt2img with a tiny model)
  under the same TheRock lib to isolate HIP init from model loading.

## Verification once fixed

Launching `krea2-turbo` from the Inference page reaches `ready on :<port>`,
or at minimum fails with sd-cpp's own error text in the session card.
