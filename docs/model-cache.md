# Model Cache

The **Model Cache** page is where weights live on disk: what is downloaded,
what is still missing, and what takes space. Registering a model
([models.md](models.md)) never downloads anything, and launching never does
either — a configuration with missing weights refuses to start and names
them. Downloading is always your explicit action.

## Where files go

- HuggingFace downloads land in your shared HuggingFace cache,
  `~/.cache/huggingface/hub`, in its own layout — one copy serves drift and
  the `hf` tools alike.
- Everything else lands under `~/.cache/drift`: `models/<family>/…` for
  checkpoints, `loras/` for LoRAs, `upscale/` for upscaler weights. Set
  `DRIFT_CACHE_ROOT` to move drift's root.
- Every cache folder is safe to delete: your tuning (default strengths, NSFW
  flags, registered models) lives in `~/.config/drift`, not beside the files.

## Tabs

- **On disk** — everything the cache holds, referenced by a model or not,
  split into HuggingFace, ModelScope, Civitai, Local and LoRAs. Each row shows its size
  and which models use it. Delete warns when a ready configuration depends on
  the file. An unreferenced file can be **assigned** to a family, which
  registers it as a model (its HuggingFace or Civitai origin is kept when it
  can be traced); **unassign** removes the registered model and leaves the
  file. An orphan LoRA folder can be **adopted** by an architecture.
  A model split into shards (registered by its `model.safetensors.index.json`)
  counts as downloaded only once every shard is there, and each shard shows
  as used by it.
- **Configured** — every registered model, grouped by the architectures that
  can use it, dimmed with a download button when its file is absent.
- **Not downloaded (n)** — the missing ones in one list, with per-model
  buttons and **Download all**.
- **Upscalers** — the weights for generation-time hires upscaling.

## Downloads

- Progress shows on the row and in the **Downloads** panel at the bottom of
  the sidebar, which lists every kind of transfer: models, LoRAs, upscalers,
  runtimes and conversions.
- At most two transfers per host run at once; the rest queue.
- A download can be cancelled and resumed later: the partial `.part` file is
  kept, and an interrupted transfer is re-queued when drift restarts.
- When the source publishes a checksum, the file is verified as it is
  written; a mismatch fails the download rather than leaving a bad file.
- Gated HuggingFace repositories and restricted Civitai files need a token —
  see [settings.md](settings.md).

## Upscalers

Generation-time upscaling (highres fix) uses RealESRGAN-style weights. Install
them from the curated list of the classic RealESRGAN releases, from any URL,
or from the Civitai browser in Upscaler type. Installed upscalers are visible
to the next session you launch. Note that RealESRGAN **x2plus** does not load
in sd-cpp; use x4plus.

## Converting a model

Some checkpoints are published in formats sd-cpp only runs on the CPU (ComfyUI
int8 files, `int8_tensorwise` with or without convrot, marked "⚠ runs on CPU"
in the Civitai browser). fp8 files run on the GPU but a GGUF quant can be
smaller. Convert either to a GGUF quant:

1. On **On disk**, click **Convert…** on the file's row.
2. Pick a target type. `Q8_0` is near lossless; `Q4_K` is the usual choice
   when memory is tight. The modal shows the file's precision mix and an
   estimated output size.
3. Choose the family (fixed when a registered model already uses the file)
   and the output name.
4. Under Advanced you can set tensor type rules, a thread count, and keep the
   intermediate file for int8/fp8 sources.

int8 and scaled-fp8 files are first rewritten by drift to a plain float file,
then converted; the modal says so and counts the temporary disk it needs.
Conversions run one at a time, show a tensor progress bar and a log, and can be
cancelled. The result lands in `models/<family>/converted/` and is registered
as a model of that family, ready to assign in a run configuration.
