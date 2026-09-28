# 05 — Model cache and downloads

**Status:** done
**Depends on:** 02, 04

A `ModelSource` becomes a file on disk. drift indexes rather than owns: weights
are hundreds of GB, so the cache is a list of roots searched before anything is
fetched, and a HuggingFace download lands in HuggingFace's own layout so one
copy serves drift and `huggingface_hub` alike.

## What it does

- Every registered model reports whether its file is present, its size and its
  path; a run configuration shows how many of its weights are cached and how
  much is left to fetch, and refuses to launch (`WeightsNotCached`) while any
  is missing. Launching never downloads.
- Registering a model never downloads either. Downloads are explicit: per model
  on the architecture card, or in bulk from the run configuration card and the
  Model Cache's **Not downloaded** tab (**Download all**).
- Downloads resume (`Range` onto a `.part`), run at most two per host, report
  bytes/total/rate, and can be cancelled leaving the `.part` for later. A
  drift restart re-queues interrupted transfers.
- SHA256 is verified whenever the source publishes one; a mismatch deletes the
  `.part` and fails the job with both hashes. Sources without a hash are
  accepted. `Local` sources are never hashed, never copied: they resolve to
  their path.
- The Model Cache page has tabs **On disk** (everything the roots hold, whether
  or not a model references it, sub-tabs HuggingFace · Civitai · Local · LoRAs,
  a total, per-row delete warning when a ready configuration depends on the
  file), **Configured** (every model grouped by the architectures whose slots
  take its family, dimmed with a download button when absent, shared models
  counted once), **Not downloaded (n)** and **Upscalers** (`10`).
- On the on-disk view an orphan file can be **Assign**ed to a family, which
  registers a model whose source is the file's origin when the path resolves
  back to a HuggingFace or Civitai file, else `Local(path)`; **Unassign**
  deletes the registered model(s) after a confirm and leaves the file. Built-in
  models are not offered for unassignment. A LoRA folder is **Adopt**ed by an
  architecture (`09`); a weight file offers **Convert…** (`25`).
- The global downloads panel at the sidebar bottom lists every transfer kind
  (model, LoRA, upscaler, runtime, conversion) with progress; queued transfers
  fold into one summary line. Each row has a ✕ that cancels it, and the
  summary line one that cancels every queued transfer; a row being
  cancelled reads *cancelling…* over an animated bar until the job ends. A
  cancel lands wherever the transfer stands — before it starts, waiting for
  its host's slot, during the range probe, or in a read gone silent. A read
  with no data for 30 s is dropped (`StallWatch`: the JDK HttpClient has no
  body read timeout, and a silently dead connection held its transfer and its
  host slot forever): a chunk retries on a new connection, a single-stream
  transfer fails and resumes when started again. A cancelled download
  keeps its `.part` for a resume; a LoRA or upscaler stays registered, so a
  restart resumes it (deleting it is what drops it). Each kind cancels
  through its own endpoint: `/api/downloads/{model}/cancel`,
  `/api/lora-downloads/{lora}/{file}/cancel`,
  `/api/upscaler-downloads/{upscaler}/cancel`,
  `/api/runtime-installs/{runtime}/cancel` and the conversion's.

## Shape

- Roots: drift's writable cache root and the shared HuggingFace hub cache,
  both resolved at startup by `Locations` (00) from `DRIFT_CACHE_ROOT` /
  `XDG_CACHE_HOME` and `HF_HUB_CACHE` / `HF_HOME`.
  `backend/.../cache/ModelCache.scala` resolves a source to a path;
  `cache/CacheInventory.scala` walks the roots for the on-disk view.
- Layout of drift's own downloads:
  `models/<familyId>/<slug>-<modelId>/<versionSlug>-<versionId>/<fileId>-<filename>`,
  with `drift-metadata.json` (`ModelMetadata`) and up to two `preview-N.jpg`
  beside the weight. Grouping is by family, not architecture, because a family
  is exactly "interchangeable in this slot" and families are shared between
  architectures. The `<fileId>-` prefix keeps Civitai's same-named files of
  different precisions apart. Slugs: lowercase, non-alphanumerics to `-`,
  ~48 characters, bare id when empty.
- HuggingFace downloads: `.locks/models--{org}--{repo}/<sha256>.lock`, stream
  to `blobs/<sha256>.incomplete`, verify, rename to `blobs/<sha256>`, relative
  symlink under `snapshots/<commit>/<path>`, write `refs/main`. Only ever adds.
  `download/HuggingFaceDownloads.scala`.
- Shapes and endpoints: `shared/.../Api.scala` (`ModelCacheStatus`,
  `CachedFileEntry(path, bytes, root, group, label, kind, referencedBy,
  partial, source)`, `GET /api/cache/status`, `GET|DELETE /api/cache/files`,
  `GET /api/cache/civitai-files?modelId=`), `shared/.../Downloads.scala`
  (`DownloadJob` with states `Queued | Downloading | Completed | Failed |
  Cancelled`, `GET /api/downloads`, `POST /api/downloads/{modelId}[/cancel]`).
- Backend: `download/Downloader.scala` (streaming, hashing while writing,
  resume, per-host permits — shared by model, LoRA, upscaler and runtime
  transfers), `download/DownloadManager.scala` (jobs keyed by model id,
  sidecars, restart re-queue).
- Frontend: `pages/ModelCachePage.scala` (tabs as URLs `/model-cache/<view>`),
  `OnDiskCacheView`, `ConfiguredModelsView`, `MissingModelsView`,
  `components/DownloadsPanel.scala` fed by `services/GlobalDownloadsService`.
- Tokens for gated HuggingFace repos and restricted Civitai files come from
  Settings (`AuthTokens`), with `HF_TOKEN` / `CIVITAI_API_TOKEN` as fallbacks.

## Notes

- HuggingFace hashes: use the `x-linked-etag` response header, never `etag`
  (the Xet content hash, a different value), and request the model info with
  `?blobs=true` or the `lfs.sha256` is absent. For LFS files the etag is the
  SHA256, so the blob name is a value drift already verifies.
- A resumed download re-reads the existing `.part` to prime the digest; hashing
  only the new bytes would pass a corrupted partial.
- Some tools write real files straight into `snapshots/` with an empty
  `blobs/` and no symlinks; resolution and the inventory accept that layout.
  drift's own writes always use blob + symlink.
- A model has at most one download in flight, so the model id is the job id.
- jsoniter omits empty collections on the wire: `referencedBy` is absent for
  orphans.
- Extra read-only roots exist in `ModelCache` but nothing configures them yet.

## Post-v1

- Update checking against the source (Civitai versions, HuggingFace
  revisions), most useful for LoRA collections.
