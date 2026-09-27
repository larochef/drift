# Files and folders

drift keeps no database. Everything is a file you can read, back up or delete.

## Configuration — `~/.config/drift/`

One JSON file per entity, one folder per entity type:

| Folder | Holds |
|--------|-------|
| `architectures/` | Model families (Flux, SDXL, wan, …). Built-in ones are re-seeded from drift's reference data on every start; your own additions stay. |
| `models/` | Registered models: where a file comes from and which architecture slot it fills. |
| `run-configurations/` | Architecture + chosen models + parameters, the thing you launch. |
| `runtimes/` | Installed or adopted sd-cpp and llama.cpp builds. |
| `loras/`, `upscalers/` | LoRA and upscaler weights you installed, with their metadata. |
| `projects/` | Projects and their prompt versions. |
| `conversations/` | One assistant conversation per project. |
| `settings/` | Default runtime per tool, and other small settings. |
| `auth-tokens/` | Your HuggingFace and Civitai tokens. |
| `backups/` | A copy of everything above, taken before drift changes the format of stored entities. Never deleted on your behalf. |

Deleting a file here removes the entity; the weights on disk stay.

`schema.json` beside those folders records which version of the entity format
your configuration is in. When a new drift expects a newer one, it copies the
directory into `backups/<timestamp>/` and updates the records in place — on the
first start after the upgrade, before anything else happens. If that ever fails,
drift says so in the log and names the backup rather than carrying on with
half-changed files. Running an *older* drift afterwards logs a loud warning:
it will not understand fields the newer one added, and saving over a record
drops them.

## Cache — `~/.cache/drift/`

Large, re-downloadable things:

| Folder | Holds |
|--------|-------|
| `models/<family>/` | Model weights by architecture family; `converted/` under it for models drift converted. |
| `modelscope/<owner>/<name>/` | Files downloaded from ModelScope, by repository. |
| `loras/<architecture>/<sfw|nsfw>/<id>/` | LoRA files, one folder per LoRA. |
| `upscale/` | RealESRGAN weights. |
| `runtimes/` | sd-cpp and llama.cpp releases, and the ROCm (TheRock) builds they pair with. |
| `logs/` | One log per session and per post-processing or conversion job. |
| `previews/` | Screen-sized copies of your images, so the gallery does not decode a 150 MB file to show you a thumbnail. Deleting them costs nothing; they are made again on demand. |
| `assistant-uploads/` | Images you sent to the assistant from your disk. |

Models downloaded from HuggingFace also use the standard
`~/.cache/huggingface/hub/` layout, so other tools can share them.

## Outputs — `~/.local/share/drift/outputs/`

- `<date>/` — every image or video generated that day, each with a sidecar
  JSON recording the full request. The gallery is built from these sidecars.
- `scratch/` — free-play results. No sidecar, cleared on restart unless kept.

Derived results (upscales, redraws) sit beside their source with their own
sidecar pointing at it.

## Moving things

Every root is resolved once at startup and logged, from the first variable set:

| Root | drift's variable | Convention | Default |
|------|------------------|------------|---------|
| configuration | `DRIFT_CONFIG_DIR` | `$XDG_CONFIG_HOME/drift` | `~/.config/drift` |
| cache (models, loras, upscalers, logs, uploads) | `DRIFT_CACHE_ROOT` | `$XDG_CACHE_HOME/drift` | `~/.cache/drift` |
| runtimes | `DRIFT_RUNTIMES_ROOT` | under the cache root | `~/.cache/drift/runtimes` |
| outputs | `DRIFT_OUTPUTS_ROOT` | `$XDG_DATA_HOME/drift/outputs` | `~/.local/share/drift/outputs` |
| HuggingFace hub | | `HF_HUB_CACHE`, else `$HF_HOME/hub` | `~/.cache/huggingface/hub` |

`DRIFT_APP_NAME` (default `drift`) is the folder name under each convention
root: `DRIFT_APP_NAME=drift-test` gives a second instance its own
configuration, cache and outputs with one variable. So a systemd user unit, a
container run with your user id, or a test on scratch copies all relocate
drift with the variables they already set; the other `DRIFT_*` variables pin
one root at a time.
`GITHUB_TOKEN`, if set, raises the rate limit when drift lists runtime releases.
