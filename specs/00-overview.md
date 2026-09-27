# 00 — Product overview and domain model

**Status:** reference (not a work item)

## What drift is

drift runs [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
and [llama.cpp](https://github.com/ggml-org/llama.cpp) for you and puts a
project-oriented interface on top: describe a model once, fetch its weights, pick a
runtime that can load them, launch the process, watch it come up, then generate,
browse, post-process and iterate on prompts with an assistant. It targets **AMD /
ROCm** hardware; Vulkan and CPU builds are supported as fallbacks.

sd-cpp's interface is a long command line naming half a dozen weight files with
values that differ per model family. drift turns that command line into something
you *describe once* and *reuse*, and owns everything around it.

## The pipeline

```
Architecture        the shape: which weight slots, which flag each fills, defaults
      |
Models              the weights: one file per slot, with a source (HuggingFace,
      |             Civitai, local) and its own parameters
      |
Run configuration   the choice: architecture + a model per slot + overrides + LoRAs
      |
Runtime             the engine: an sd-cpp or llama.cpp build (+ ROCm libraries)
      |
Session             one long-lived sd-server / llama-server process
      |
Generation          HTTP jobs against that process; results in the gallery
```

**One process per configuration, loaded once.** Models are tens of gigabytes, so
drift never runs `sd-cli` per image: a session is a long-lived `sd-server` and every
generation is an async job against it. The slow, failure-prone step is startup, which
is why log streaming and progress ([`13`](13-log-streaming.md)) are a feature of
their own. Post-processing that needs a different model (upscaling, redraw, edit) runs its
own job server, or `sd-cli` for ESRGAN ([`15`](15-post-hoc-resize.md), [`26`](26-tiled-pid.md),
[`27`](27-redraw.md), [`39`](39-seamless-edit.md)).

sd-server's native API: `POST /sdcpp/v1/img_gen` and `/vid_gen` (async jobs),
`GET /sdcpp/v1/jobs/{id}`, `POST …/cancel`, `GET /sdcpp/v1/capabilities`. drift
replaces sd-server's built-in web UI entirely.

## Vocabulary

**Architecture** — the shape of one invocation for a model family: its checkpoint
slots, the flag each maps to, default parameters, tags saying what it is for, and
the runner (`tool`: sd-cpp or llama.cpp). Names no file. [`01`](01-architecture-registry.md).

**Checkpoint slot** — one entry of an architecture: a name, the flag it fills, the
family that can fill it, whether it is required.

**Family** — a plain string shared by a slot and the models that can fill it. Not an
entity, a join key: `wan-2.1-vae` is filled by the same file whichever architecture
asks for it, so a file is registered once.

**Model** — one concrete weight file in a family, with a source and its own
parameters. [`02`](02-model-registry.md).

**Run configuration** — an architecture, a model per slot, parameter overrides and
default LoRAs. The unit you launch. [`04`](04-run-configurations.md).

**Runtime** — a specific sd-cpp or llama.cpp build, paired with a ROCm distribution
when it is a ROCm build. [`06`](06-sdcpp-runtime.md), [`17`](17-assistant-runtime.md).

**Session** — a live process started from a configuration on a runtime, with a
port, a status and a log. [`07`](07-launch-and-supervision.md).

**Generation** — one job's recorded request and outputs, kept as a sidecar beside
the files. [`08`](08-inference-ui.md), [`12`](12-gallery.md).

**Project** — where work lives: a kind (image or video), prompt versions, a
conversation with the assistant, and the generations made in it.
[`19`](19-projects-and-prompt-versions.md).

## How the command line is assembled

`shared/src/drift/shared/CommandLine.scala` is the one merge site; the preview on a
configuration card and the launcher call the same function.

```
for each checkpoint slot:        emit slot.flag <cached path of the assigned model>
parameters, resolved in layers:  architecture ← model ← runtime defaults ← configuration,
                                 then runtime vetoes (each reported as a note)
                                 empty value = bare flag ("--diffusion-fa": "")
sd-cpp only:                     --lora-model-dir <loras>/<architectureId>, --hires-upscalers-dir
listen:                          --listen-ip/--listen-port (sd-cpp), --host/--port (llama.cpp)
```

See [`16`](16-parameter-resolution.md).

## Code layout

- `shared/` — entities and tapir endpoints, cross-compiled JVM/JS. One file per
  subject (`Api`, `Runtimes`, `Sessions`, `Generation`, `Projects`, …).
- `backend/` — tapir + Netty on port 4321, one package per subject (`runtime`,
  `session`, `sdserver`, `postprocess`, `projects`, `assistant`, `conversion`, …).
  Persistence is one JSON file per entity at `~/.config/drift/<type>/<id>.json`;
  no database. Reference data seeds from `backend/resources/reference/`.
- `frontend/` — Laminar SPA. Services own an `EventBus` of commands and a `Var` of
  state; status (sessions, downloads, generations, jobs) arrives over one WebSocket
  ([`11`](11-status-websocket.md)). Components are plain classes with a `lazy val
  element`.

## Navigation

Five sidebar entries, in the order the work happens: **Projects** (also `/`),
**Gallery**, **Models** (run configurations of both runners + architectures, one
tab each), **Model Cache**, **Settings**. Each page has exactly one URL.

## Where drift puts things

| path | holds | regenerable |
|---|---|---|
| `~/.config/drift/<type>/` | entities: architectures, models, run-configurations, runtimes, loras, upscalers, projects, conversations, auth-tokens, settings | no |
| `~/.cache/drift/models/<family>/<slug>-<modelId>/<slug>-<versionId>/<fileId>-<name>` | Civitai weights | yes |
| `~/.cache/drift/models/<family>/converted/` | converted GGUFs ([`25`](25-model-conversion.md)) | yes |
| `~/.cache/drift/loras/<architectureId>/…` | LoRAs, one root per architecture | yes |
| `~/.cache/drift/upscale/` | ESRGAN upscaler weights | yes |
| `~/.cache/drift/runtimes/{sd-cpp,llama-cpp}/<tag>-<backend>/`, `…/therock/<gfx>-<version>/` | unpacked releases and ROCm builds | yes |
| `~/.cache/drift/logs/` | session and job logs | yes |
| `~/.cache/huggingface/hub/` | HuggingFace weights, in HuggingFace's own layout | yes |
| `~/.local/share/drift/outputs/<date>/` | generations and sidecars; `scratch/` for free play | **no** |

Rules: HuggingFace files live in the shared HuggingFace cache, never under
`~/.cache/drift`; `~/.cache` must stay safe to delete; the user's work lives under
`~/.local/share` and their configuration under `~/.config`.

Every root is resolved once at startup by `backend/.../Locations.scala` and
logged: drift's own variable (`DRIFT_CONFIG_DIR`, `DRIFT_CACHE_ROOT`,
`DRIFT_RUNTIMES_ROOT`, `DRIFT_OUTPUTS_ROOT`), else the convention's variable
(`XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`; `HF_HUB_CACHE` or
`HF_HOME` for the HuggingFace hub), else the home default, where
`DRIFT_APP_NAME` (default `drift`) names the folder under each convention
root. Managers take paths; none carries a default of its own.

## Standing constraints

- **No backward compatibility, but migrations** (`specs/38`). Rename and remove
  fields freely and never write a codec that reads two shapes; where stored
  records need to follow, add a versioned migration beside the change. Keep
  breaks *loud*: `StorageService.list` skips what it cannot decode and logs a
  warning.
- **Reuse existing machinery**: the same shape means the same code with a
  parameter (llama.cpp runtimes are the sd-cpp runtime manager with a `tool`).
- **Drive sd-cpp, do not reimplement it**: upscalers, LoRA application, samplers and
  conversion are sd-cpp features drift exposes.

## Non-goals

Training or LoRA creation; multi-user or remote access; NVIDIA as a target (Vulkan
builds may work).
