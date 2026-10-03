# 46 — Starter configurations

**Status:** done in code, not yet run live — seeding, migration and the
Models page checked on an isolated config directory; no starter has been
downloaded and launched from a fresh install yet
**Depends on:** 04, 05, 23, 43

A first-time user generates without learning the concepts first: drift ships
run configurations for a handful of models, and wherever a model is started, a
missing runtime or download is offered in place of the launch.

## What it does

- `backend/resources/reference/run-configurations.json` is seeded into
  `~/.config/drift/run-configurations/` **once**: unlike architectures and
  models, whose reference stays authoritative, a starter is an ordinary
  configuration from the moment it lands — edited, deleted, never rewritten
  or brought back. An id a user configuration already holds is skipped. Six
  starters, seeded on the upstream engine (sd-cpp or llama.cpp, which run
  everywhere) and moved to the drift runner wherever it is available — the
  runner is the default wherever it is (François, 2026-10-03):
  - `starter-flux2-klein-9b` — Flux.2 Klein 9B, Q4_K_M
  - `starter-krea2` — Krea 2 Turbo, Q4_K_M
  - `starter-qwen-image-2.1` — Qwen Image 2.1, Q4_K, with the vision projector
    so it edits too
  - `starter-minimax-h3` — MiniMax H3, with the audio VAE
  - `starter-pid-flux2` — PiD 1.5, FLUX.2, 1024 → 4096 in 4 steps
  - `starter-qwen3.6-35b-a3b` — Qwen 3.6 35B-A3B, UD-Q4_K_XL MTP build with
    its vision projector (MTP and vision together need llama.cpp b9240+)
- Wherever a model is started, a configuration missing weights offers their
  download instead: the run configuration card's footer, a project's image,
  video, chat and assistant pickers (the entry is suffixed *download the
  weights* / *downloading*, and picking it downloads without stopping the live
  model), the gallery detail's reuse and *try this task* actions, and the
  redraw, edit and PiD panels. Once the weights land the launch comes back.
- Where no valid runtime of a configuration's engine is installed, the same
  places say so by what the models make, not the technology — an image,
  video or text runtime, read off the architecture (llama.cpp's tool is text,
  an sd-cpp architecture tagged video and not image is video, the rest image):
  *No video runtime is installed.*, or, when one of the tool is but not one
  these run on, the models it is for (*Choose the text runtime for Qwen 3.6
  35B-A3B (chat)*) — with a
  choice of build and an **Install** button. The choices are the builds of
  the architecture's engines: drift's runner where the architecture runs on
  it and this drift carries it for the GPU (gfx1151), then ROCm (offered only
  when the ROCm kernel driver exposes an AMD GPU, paired with TheRock for its
  gfx target), Vulkan and CPU. The first is preselected and marked
  recommended: the runner where offered (`RuntimeManager.installOptions`
  marks it so), else ROCm for sd-cpp on such a GPU, else Vulkan. An upstream build installs the newest release like any
  latest-tracking runtime; the runner installs both its runtimes as Settings
  does. Picking another engine than a configuration's moves the
  configuration to it (`43`: a configuration names its engine) — on the
  backend, once the runtime has validated and before its job reads
  `Completed`, so a failed or cancelled install leaves every configuration on
  the engine it had, and leaving the page loses nothing. An engine already
  installed is marked so, and picking it only moves them. The first valid
  runtime of a tool becomes its default; the read-change-write of that
  selection is one step, so the runner's two runtimes validating together
  both become defaults.
- The button turns into a spinner the moment it is pressed — before the
  backend answers — and the notice reads *Installing the image runtime…*
  until the runtime validates, when it goes. A failed install shows its
  reason, and the button offers to try again.
- A project's picker says it once beside the picker, for the configurations
  it offers (those making the project's kind), each picker on its own row
  whatever the notice beside it says: every build any of them can take, and a pick on another
  engine switches each one whose architecture runs on it — those that do not
  (MiniMax H3 on the runner) keep asking for theirs. Each entry names what it
  lacks; picking one downloads its weights, never installs a runtime, and the
  pick stays selected until nothing is missing, when it launches (stopping
  the tool's live model). Picking again or the empty entry drops it. The
  gallery detail shows the notice for the generation's own configuration.
  Nothing installs by itself.

## Shape

- Migration 011 (`Migrations.coded` "starters-on-runner"): a seeded starter
  (its id in `settings/seeded-run-configurations.json`) still on its
  upstream engine moves to the drift runner when its architecture runs there
  and the runner's runtime for its tool is installed and valid. Once, as a
  migration is, so a starter moved back by hand stays where it was put. It
  came of a PiD starter left on sd-cpp on an install whose runtimes predate
  the starters: a 4096² upscale proposed nine 1536² tiles, where the runner
  decodes it in one.
- A new configuration's form starts on the drift runner when the chosen
  architecture runs there and the runner is installed, else on the
  architecture's first engine.
- `StorageService.seedOnce`: writes each reference item absent from
  `settings/seeded-<entityType>.json`, then records its id there — the records
  cannot say it themselves once the user may delete them.
- `services/LaunchPrerequisites`: configuration id → `Missing(runtime,
  weights)`; `choose` installs and switches, `download` fetches weights. Its
  `effects` mount the cache, download and runtime services, so only pages that
  do not already mount them (gallery, project workspace) mount it.
- `components/LaunchOrDownload`: the launch element, or the runtime notice and
  the weights' download in its place.
- `POST /api/runtime-installs/for-configurations`
  (`InstallForConfigurationsRequest`: the option, the configuration ids):
  `RuntimeInstallResolution` starts the install (none when its engine is
  already valid) and registers the move with `RuntimeInstalls.whenValid`,
  which runs it between validation and `Completed`.
  `RuntimeService.requested` holds the ids sent but not answered;
  `LaunchPrerequisites.followInstalls` re-reads the configurations when an
  install settles or is answered.
- `RuntimeManager.installOptions` answers `GET /api/runtime-installs/options`
  (`RuntimeInstallOption`: tool, engine, the `InstallLatestRequest` — none for
  the runner — the runtime id, a label, whether it is recommended);
  `runtime/GpuDetection` reads `gfx_target_version` from
  `/sys/class/kfd/kfd/topology/nodes/*/properties` (110501 → gfx1151).

## Notes

- The seeded file is checked against `architectures.json` and `models.json`
  when edited: every required slot filled, every model of its slot's family,
  the runner among the architecture's.
