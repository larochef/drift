# drift — feature specs

One file per feature. Each describes the feature **as it exists in the code**:
what it does, its shape (entities, endpoints, files, code locations) and the
non-obvious constraints anyone touching it must know. The road that led there
is not recorded; git history has it. Shared vocabulary lives in
[`00-overview.md`](00-overview.md). The user-facing side of the same features is
in [`../docs/`](../docs/README.md).

## Status board

| # | Feature | Status |
|---|---------|--------|
| [00](00-overview.md) | Product overview and domain model | reference |
| [01](01-architecture-registry.md) | Architecture registry | done |
| [02](02-model-registry.md) | Model registry | done |
| [03](03-model-discovery.md) | Model discovery (HuggingFace, Civitai, local) | done |
| [04](04-run-configurations.md) | Run configurations | done |
| [05](05-model-cache-and-downloads.md) | Model cache and downloads | done |
| [06](06-sdcpp-runtime.md) | sd-cpp / ROCm runtime management | done |
| [07](07-launch-and-supervision.md) | Launching and supervising `sd-server` | done |
| [08](08-inference-ui.md) | Inference UI (txt2img, img2img, edit, video) | done |
| [09](09-lora-management.md) | LoRA management | done |
| [10](10-generation-time-upscaling.md) | Generation-time upscaling (hires + VAE tiling) | done; ESRGAN to move out of the generation form |
| [11](11-status-websocket.md) | Status WebSocket | done |
| [12](12-gallery.md) | Gallery and history | done |
| [13](13-log-streaming.md) | Log streaming and progress feedback | done |
| [14](14-generation-form-completions.md) | Generation form: folded sections, batch, mask, guidance | done |
| [15](15-post-hoc-resize.md) | Post-hoc upscale: ESRGAN and PiD | done |
| [16](16-parameter-resolution.md) | Parameter resolution (architecture, model, runtime, configuration) | done |
| [17](17-assistant-runtime.md) | Assistant runtime (llama.cpp) | done |
| [18](18-assistant-models-and-sessions.md) | Assistant models, sessions and chat proxy | done |
| [19](19-projects-and-prompt-versions.md) | Projects and prompt versions | done |
| [20](20-prompt-assistant-conversation.md) | Prompt assistant conversation, compaction, restart | done in code, not yet run live |
| [21](21-assistant-page.md) | Assistant chat on the Models page | done |
| [22](22-free-play-and-scratch-generations.md) | Free play and scratch generations | done; its page is the Sandbox since 47 |
| [23](23-seeded-vendor-models.md) | Seeded vendor models | done |
| [24](24-model-details-in-browsers.md) | Model details in the browsers | done |
| [25](25-model-conversion.md) | Model conversion | partial — step 3 (imatrix + presets) parked |
| [26](26-tiled-pid.md) | Tiled PiD on the GPU | done |
| [27](27-redraw.md) | Redraw: tiled low-strength img2img | done |
| [28](28-configuration-loras.md) | Default LoRAs on run configurations | done in code, not yet run live |
| [29](29-split-oversized-files.md) | Code structure: facades and component split | done |
| [30](30-gallery-ergonomics-and-image-import.md) | Gallery ergonomics, image import, visible post-processing | planned |
| [31](31-project-kinds.md) | Project kinds: image or video | done in code, not yet run live |
| [32](32-prompt-library.md) | Prompt library: variants of the assistant, redraw and compaction prompts; the Ideogram caption writer | done in code, not yet run live |
| [33](33-lora-sources.md) | LoRA sources: HuggingFace and disk installs, the official LoRAs list | done in code, installs checked against a stub backend |
| [34](34-sharded-safetensors.md) | Sharded safetensors models: an index downloads, weighs and checks its shards | done in code, checked on a tiny repository |
| [35](35-assistant-loras.md) | LoRAs on chat models: adapters loaded at launch, defaults applied per reply | done in code, not yet run live |
| [36](36-huggingface-examples.md) | Examples in the HuggingFace browser: image tiles, an Examples tab, the card's gallery | done |
| [37](37-modelscope.md) | ModelScope: a model source, its browser (LoRAs by base model, covers), downloads | done |
| [38](38-entity-migrations.md) | Entity migrations | done in code, not yet run live |
| [39](39-seamless-edit.md) | Edit: change part of a picture by instruction, keep the rest | done in code, not yet run live |
| [40](40-pause-and-resume.md) | Pausing a tiled job, and resuming it after a restart | done in code, not yet run live |
| [41](41-text-projects.md) | Text projects: a kept conversation with a raw chat model | done in code, not yet run live |
| [42](42-drift-runner.md) | drift runner: own inference engine for Strix Halo (HIP kernels, Scala via FFM), chat first with MTP | partial — steps 1–9 done, 10–13 in part; step 14: MiniMax H3, Wan 2.2 A14B, LTX 2.5 (videos; H3 and LTX with their soundtracks) |
| [43](43-runners-per-architecture.md) | Runners per architecture: the drift runner installed from the runtimes page, supported runners and model kind, a runner per configuration | done in code, run on a copy of the configuration |
| [44](44-generation-progress.md) | Generation progress: batch and image bars, the runner reporting its own progress | planned |
| [45](45-redraw-steps-and-reference.md) | Redraw: full steps at any strength, a reference at the tile's scale, reference-taking models only | planned |
| [46](46-starter-configurations.md) | Starter run configurations; every launch becomes a download while weights are missing | done in code, not yet run live |
| [47](47-sandbox.md) | Sandbox page: free play out of the Models page, image / video / text switch | done in code, not run live |
| [48](48-soundtrack-polish.md) | Soundtrack polish: a post-processing step on a video's audio (loudness, de-harsh, denoise, low cut) through ffmpeg | planned |
| [49](49-lora-sampling-settings.md) | Sampling settings on LoRAs: a turbo LoRA carries its steps, CFG, flow shift and sigmas, and selecting it sets them | done in code, not yet run live |
| [50](50-inputs-from-the-gallery.md) | Inputs from the gallery: every input slot and the assistant pick from the gallery with its filters; an input goes to the assistant in one click | partial |

## What is left

- `25` step 3: importance-matrix collection and per-architecture rules presets.
- `39`: a live run of the Edit task (Flux.2 Klein, a selection and a whole image).
- `40`: a live pause, a drift restart and a resume on a long PiD job.
- `30`: batch thumbnails reachable without scrolling, long prompts folded in
  the detail, importing images that are not in the gallery, post-process
  progress visible from anywhere.
- Live walk-throughs of `20`, `28`, `31`, `41`, `32` (the Ideogram caption writer on a local model).
- `23`: first runs of Mage-Flow Turbo, Mage-Flow Edit Turbo and SenseNova U1.5;
  Qwen Image 2.1 once sd-cpp releases a binary that runs it (seeded 2026-09-20),
  generating and editing, and its defaults settled against the vendor's numbers.
- `33`, `34`: a live pass — SenseNova U1.5 from its official shards, and the
  official 8-step LoRA on it.
- `42`: steps 10–15 of the runner; step 14 has MiniMax H3 and LTX 2.5 (with their soundtracks; no image conditions) and Wan 2.2 A14B (no LoRAs, no 5B); video LoRAs (Wan 2.2's high/low-noise pairs first) and HunyuanVideo next.
- Ideas not specced: parameter sweeps (one prompt × configurations), timings
  and a loud CPU-fallback warning, Civitai example → form, assistant autopilot,
  a disk view, continuing a video (its last frame, read by ffmpeg, becomes the
  init image of a new generation with the same parameters, from the detail's
  actions; an image-to-video configuration takes it).

## Conventions

- One file per feature, `NN-short-slug.md`, numbered in the order the work
  happened. Numbers are stable: other files and code comments link to them.
- Sections: a **Status** line, **Depends on**, a short purpose paragraph, then
  **What it does** (observable behaviour), **Shape** (entities, endpoints,
  files, code locations by path — never line numbers), **Notes** (durable
  constraints and gotchas), and for unfinished features **Remaining**. An
  optional **Post-v1** lists what is wanted but not required.
- Present tense, no dates, no narrative. When the code changes, the spec
  changes with it; a spec that disagrees with the code is a bug in the spec.
- Status is `done`, `done in code, not yet run live`, `partial` (with what is
  missing) or `planned`. Keep the board above in step.
