# 09 — LoRA management

**Status:** done — a real wan 2.2 pair install and a turbo-LoRA speed check
are still worth one live pass
**Depends on:** 05, 08

Browse and fetch LoRAs (from Civitai here; HuggingFace, the disk and the
official LoRAs in [`33`](33-lora-sources.md)), keep their metadata and the user's tuning,
and apply them to a generation — including wan 2.2's high/low-noise pairs.
drift does not apply LoRAs itself: it files them where sd-server scans, and
sends paths and weights per request.

## What it does

- Each architecture card has a folded **LoRAs (n)** section
  (`components/FoldedSection`, shared with the post-processing panels' Advanced
  since 2026-09-21; the body is removed when folded, a card being lighter for
  it and the rows coming back from signals): the installed
  LoRAs with their label (click it to rename — an installed file's own name is
  often all an install had to go on), trigger words, an sfw/nsfw tag (click to
  move the folder),
  an editable default strength, per-file stage chips (general / low noise /
  high noise, click to cycle) and delete (entity and files). **+ Add LoRA**
  opens the browser (`33`), Civitai among its sources, in LoRA type and
  filtered by the architecture's `civitaiBaseModels`: its weight file rows are
  ticked — across versions if that is what is wanted — and one bar installs
  the lot as one LoRA, as one each, or into a LoRA already installed. The
  browser stays open, the bar saying what was sent. Back restores the search
  scroll. A model whose version is installed for this architecture carries a
  **✓ installed** chip on its grid tile, the version one beside its name and
  the file one on its row (`33`).
- An opened model lists only the versions published for the architecture's
  Civitai base models (François, 2026-09-18: installing one made for another
  base model is the commonest way to end up with a LoRA that does nothing).
  **Versions for other base models (N)** brings the rest back, still flagged
  ⚠ — Civitai's base model is whatever the author wrote. A version that
  declares none is never hidden, and the filter is off when the architecture
  declares no base model. It holds wherever the version list shows, a
  checkpoint's files included.
- Install classifies `nsfw` from Civitai's flags and each file's stage from
  `high`/`low` + `noise` markers in file or version names, writes the sidecar
  and previews, and queues one download per file.
- The generation form's picker (`components/LoraPicker.scala`, also used by
  the run configuration form, `28`) searches label, trigger words, tags and
  description, has an **NSFW** checkbox (on by default), seeds the strength
  from the LoRA's default, inserts trigger words into the prompt on click, and
  marks "⚠ needs restart" any file the live session's `capabilities.loras`
  does not list.
- On submit each selected LoRA sends all its files as `lora[]` entries at the
  chosen multiplier, `is_high_noise` on the high-noise file. The prompt is
  never rewritten.
- The Model Cache on-disk view lists the LoRA tree under its LoRAs sub-tab;
  deleting a row removes the whole folder; an orphan folder can be **Adopt**ed
  by an architecture, which rebuilds the entity from the sidecar (a folder
  without one adopts its files as they lie, `33`).

## Shape

- Layout: `~/.cache/drift/loras/<architectureId>/<sfw|nsfw>/<loraId>/` holding
  `drift-lora.json` (regenerable Civitai metadata), previews and the weight
  files, a Civitai one as `<fileId>-<filename>`. A wan 2.2 pair is two files in
  one folder. Every sd-cpp launch gets `--lora-model-dir <root>/<architectureId>`,
  so a session sees only its architecture's LoRAs, and a request's `lora[].path`
  is `<sfw|nsfw>/<loraId>/<fileName>`.
- `Lora` entity at `~/.config/drift/loras/<id>.json`
  (`shared/.../Loras.scala`): `architectureId`, `label`, `nsfw`,
  `defaultStrength`, `triggerWords`, `tags`, `description`,
  `files: List[LoraFile(fileName, source, stage, sizeBytes, sha256)]`
  (see [`33`](33-lora-sources.md)). `id` = `<architectureId>-<slug>-<source>`
  (the Civitai model id for a Civitai install) and doubles as the folder name;
  the architecture is part of the identity because one Civitai model publishes
  versions for several base models.
  `Lora.selections(strength)` builds the `lora[]` entries.
- Endpoints: `GET /api/loras`, `POST /api/loras/install`
  (`InstallLoraRequest(architectureId, source)`, a Civitai install being
  `CivitaiVersion(modelId, versionId, fileIds)`),
  `POST /api/loras/adopt` (`AdoptLoraRequest(path, architectureId)`),
  `PUT|DELETE /api/loras/{id}`, `GET /api/lora-downloads`
  (`LoraDownloadJob` keyed by lora + file name).
- Backend `backend/.../lora/LoraManager.scala` owns install, per-file jobs
  through the shared `Downloader`, the nsfw folder move, adoption and
  recursive delete. Frontend `pages/architectures/LoraSection.scala`,
  `components/LoraPicker.scala`, `services/LoraService`.

## Notes

- sd-cpp's server APIs intentionally ignore `<lora:name:weight>` prompt
  syntax; LoRAs go through the structured `lora` array, the tiled
  post-processing jobs of `26`/`27` included.
- `--lora-model-dir` is scanned at launch and subdirectories are included
  (unlike `--hires-upscalers-dir`). A file installed or moved after launch is
  invisible until restart — which is what the picker's warning reports.
- The user's tuning lives in the entity under `~/.config`, never in the cache,
  so `~/.cache` stays safe to delete.
- The picker keeps strength in a Var separate from the selected-ids list:
  rebuilding rows per keystroke steals the input's focus.
- The command-line preview prints `--lora-model-dir ~/.cache/drift/loras/…`
  with a literal `~`; the launcher passes the absolute path.
- `--lora-apply-mode` stays automatic (quantized weights get `at_runtime`).

## Post-v1

- Pairs published as two Civitai *versions* install as two LoRAs; merging is
  manual.
