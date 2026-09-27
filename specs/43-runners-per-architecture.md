# 43 — Runners per architecture

**Status:** done in code 2026-09-25; run end to end on a copy of François's configuration, not yet in his own drift
**Depends on:** 01 (architecture registry), 04 (run configurations), 06 and 17 (runtimes), 07 (launch), 18 (chat sessions), 19 and 41 (projects), 38 (migrations), 42 (the drift runner)

drift has more than one engine per tool: the drift runner speaks
llama-server's and sd-server's APIs, so it registers as a llama.cpp runtime
and an sd-cpp runtime. But it runs only on some GPUs and only some models.
drift knows which engine can run what, offers only those, and a run
configuration says which engine it runs on.

## Words

- **Tool**: the API and executable a runtime speaks (`RuntimeTool`: sd-cpp,
  llama.cpp). Unchanged.
- **Engine**: what does the work (`RuntimeEngine`: sd-cpp, llama.cpp, drift
  runner). A runtime's engine is derived: the drift runner's release tag makes
  it the drift runner, any other runtime is its tool's engine. An engine can
  have several installed builds (runtimes).
- **Model kind**: what the model is, as the engines name it: the GGUF
  `general.architecture` (`qwen35`, `qwen35moe`, `qwen4exp`, `qwen3`) or a
  diffusion family (`krea2`, `flux2-klein`, `flux2-dev`, `qwen-image-2.1`, `pid-flux2`,
  `pid-flux1`, `pid-qwen-image`).

## What it does

1. **Installing the drift runner** (François, 2026-09-25).
   - The runtimes page offers the drift runner beside sd-cpp and llama.cpp,
     when this drift carries it. One install creates both its runtimes, chat
     (`drift-runner`, tool llama.cpp) and images (`drift-runner-images`, tool
     sd-cpp), on one TheRock build. The build is chosen as for a ROCm
     release (the "ROCm (TheRock) build" field). It is paired against the
     ROCm version the runner's kernels were compiled with, and downloaded if
     missing.
   - Nothing is downloaded but TheRock: the files are the jar drift ships
     and a launcher running drift's own Java (25).
   - The runner's runtimes are drift's own, not adopted. "Change ROCm"
     re-pairs both at once, and deleting one deletes both.
   - At startup (`RunnerFiles.refresh`), drift only refreshes installed runner
     runtimes: the jar and launchers when this drift carries a newer runner,
     validated again then. It never creates them, so a deleted runner stays
     deleted.
2. **Only where it runs.** The runner's `--version` loads HIP and names the
   GPU it finds; it exits with an error on a GPU other than gfx1151. Its
   validation, at install and at each refresh, then fails with that reason,
   and an invalid runtime is never launched. The runtimes row shows why.
3. **What the runner runs.** The runner's `--model-kinds` lists the model
   kinds it runs. Validation keeps them on the runtime (`modelKinds`). The
   upstream engines keep none, which means they don't restrict.
4. **Architectures.**
   - Each architecture declares its `modelKind` (none when unknown) and its
     `runners`: the engines that can run it. The upstream engine of its tool
     is always one of them.
   - The built-ins declare theirs in the reference file. The drift runner is
     listed where it has been run: Qwen 3.6 35B-A3B, Qwen 3.8 27B, Krea 2,
     FLUX.2 [klein] 9B, Qwen Image 2.1 and PiD's FLUX.2, FLUX.1 and Qwen
     Image variants (1024 → 4096). User-made architectures (Flash Next) add
     it in the form.
   - The architecture form edits both. It offers the drift runner only for a
     kind the installed runner reports (when one is installed).
5. **Run configurations** carry a `runner`, one of their architecture's
   runners (the upstream engine for existing ones, by migration). The
   configuration forms have a Runner select. A runner that isn't installed or
   valid is shown so, and the launch refuses with that reason.
6. **Launching.** Every path resolves the runtime from the configuration's
   runner: sessions, projects, text projects, the assistant, and the tiled
   post-processing (PiD, redraw, edit). The engine's runtime is the tool's
   default runtime when that is the engine's, else the engine's newest valid
   one. A launch-time pick stays as an override, among the runtimes of the
   architecture's engines. ESRGAN and conversion need `sd-cli`, so they run
   on sd-cpp only.
7. **Seeing it.** The launch control lists the runtimes of the
   architecture's engines, the configuration's runner first. Each
   configuration card and the project pickers name the runner. The gallery's
   PiD tile size follows the configuration's runner (bug 30 still decides
   the size).

## Schema

- `Architecture`: `modelKind: Option[String]`, `runners: List[RuntimeEngine]`.
  Built-ins are reseeded. User-made ones get their tool's engine and no kind
  (migration 2).
- `RunConfiguration`: `runner: RuntimeEngine`, its architecture's tool's
  engine (migration 2, a coded step: it reads the architecture).
- `Runtime`: `modelKinds: Option[List[String]]`, none for existing ones
  (migration 2).
- No Scala defaults on these fields.

## How it is built

- **Install.** The runner is one more source in `RuntimeInstalls`: a
  request with the release tag `drift-runner`, no download, and an asset
  carrying the kernels' ROCm version. Pairing, the TheRock download, the jobs,
  validation and "Change ROCm" are the ROCm releases' own code. The files are
  written once TheRock is there (`RunnerFiles.write`), under
  `<runtimes>/llama-cpp/drift-runner-rocm/` and
  `<runtimes>/sd-cpp/drift-runner-rocm/`. `installRunner` answers both jobs,
  and the runtimes page has an "Install the drift runner" tab.
- **Flags.** A `RuntimeRule` names its engine (`specs/16`): the runner's
  images rule drops `--attn-scale` with a note; `--guidance` reaches the
  runner, which applies it to FLUX.2 [dev] and notes it ignored for the
  other models. The runner keeps refusing unknown flags.
- **Validation.** For the runner, validation runs `--version` (the runner
  loads its gfx1151 kernels on the GPU, `RunnerIdentity`) and then
  `--model-kinds`.
- **Resolution.** `resolveForLaunch(tool, engine, pin)`: a pin wins, else
  the tool's default when it is the engine's, else the engine's newest valid
  build. Sessions refuse a pinned runtime whose engine the architecture does
  not list, naming it. Tiled jobs read the configuration's runner first
  (`SessionManager.runnerOf`).
- **Migration 2** (`002-runners.json`):
  - a coded step `runner-from-architecture`: coded steps now read other
    records, and this one runs before the built-ins are reseeded;
  - `runners` added by tool;
  - built-ins reseeded;
  - the old seeded runner registrations made drift's own.
- **Migration 3** (`003-pid-runner-variants.json`): the built-in PiD FLUX.1
  and Qwen Image architectures reseeded, now listing the drift runner.
- **Migration 4** (`004-qwen38-27b.json`): the built-in `qwen3.8-27b`
  (`qwen35`, llama.cpp or the drift runner; slots `model`, optional
  `mmproj` and `mtp` `--spec-draft-model`) reseeded, so a stored copy gives
  way to the shipped one.
- **Migration 5** (`005-flux2-dev-runner.json`): the built-in `flux.2-dev`
  reseeded, now listing the drift runner (model kind `flux2-dev`).
- **Migration 6** (`006-qwen36-mtp-head.json`): the built-in
  `qwen3.6-35b-a3b` reseeded with an optional `mtp` slot
  (`--spec-draft-model`) in the main model's own family. Any 35B file with an
  MTP layer can lend its head to a build that has none, such as a ROCmFP4
  one.
- **Packaging fix found on the way.** The universal stage's jar packed the
  frontend built by the previous `universalStage`. The jar's resources were
  taken before `universalMappings` rebuilt it. Now `frontendBundle` is a task
  of its own that the jar carries, and `run` serves the fast build from its
  folder (`runClasspath`).

## Verified

- Backend tests, the migration 2 test included.
- On a copy of François's configuration:
  - 30 configurations got their runner and 27 built-ins were reseeded.
  - The runner installed on TheRock 10.0.0: both runtimes valid, the GPU named
    (Radeon 8060S), and their model kinds listed.
  - "Change ROCm" to 7.13.0 moved both runtimes, still valid.
  - The Qwen 3.6 configuration set to the drift runner launched on
    `drift-runner` with no runtime picked, and answered.
  - Gemma 4 pinned to the runner was refused.
- The forms, driven in a browser:
  - the runner tab;
  - the configuration's Runner select;
  - the architecture's checkbox, off for `gemma4` and on for `qwen35`.

