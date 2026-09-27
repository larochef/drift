# 33 — LoRA sources and the official LoRAs

**Status:** done in code; disk, HuggingFace, catalog installs and adoption
checked against a stub backend, not yet run in a generation
**Depends on:** 05, 09, 28

LoRAs are not all on Civitai: vendors publish theirs on HuggingFace (SenseNova's
8-step distillation), and some sit on the disk already. A LoRA file carries the
same `ModelSource` a model does, so drift installs from Civitai, a HuggingFace
repository or the drift host; and a reference file lists the official LoRAs
drift offers, one click to install.

## What it does

- The LoRA install control (architecture card, run configuration's default
  LoRAs) is one button, **+ Add LoRA**, opening the browser with the source as
  a row in its head (`03`): **Civitai** (greyed out, the tooltip saying why,
  when the architecture declares no Civitai base model; not offered at all for
  a chat architecture, `35`), **HuggingFace**, **ModelScope**, and **This
  machine**, whose files are copied into the LoRA store (sd-cpp only sees what
  is under its `--lora-model-dir`), the original left alone.
- Every source installs the same way (François, 2026-09-18). An opened model
  or repository is one list of weight files, **none ticked to begin with**;
  ticks may cross a Civitai model's versions and a directory's folders; and
  one bar over the list says how many are ticked, how much they weigh, and
  where they go: **one LoRA of N files** (the default, and what makes a wan
  2.2 pair one LoRA), **N separate LoRAs**, or **files of 'X'** for each LoRA
  this model or repository already made — which is how a pair fetched one half
  at a time ends up whole. A single ticked file defaults to a LoRA of its own.
  The browser stays open afterwards, the bar saying what was sent, since
  finding a pair often means several installs from one search.
- **Pairing two halves** (François, 2026-09-18): a wan 2.2 pair is sometimes
  published as two repositories, one per stage, and installs as two LoRAs. A
  LoRA whose files are all of one stage — all high noise, or all low — carries
  a **⇄ Pair** button naming the other halves it could join: this
  architecture's LoRAs that are all of the opposite stage. Choosing one, and a
  name for the two together (suggested from what their names have in common),
  moves the other's files into this one's folder — renamed `high-`/`low-` when
  the two repositories named them the same, which they often do — and deletes
  the other entity. `POST /api/loras/pair`, `PairLoraRequest(loraId, otherId,
  label)`, `LoraManager.pair`.
- Two versions of one Civitai LoRA can be installed as two LoRAs and compared
  in the picker: a LoRA whose files all come from one version of a model that
  publishes several is named after that version, so their ids differ.
- The HuggingFace LoRA browser opens on LoRAs already: an architecture's
  `huggingFaceBaseModels` (seeded for every built-in except the PiD decoders)
  name the repositories its LoRAs are trained on, and the first one's adapters
  (`filter=base_model:adapter:<repo>`) are listed, most downloaded first,
  without typing. A "trained on" choice offers the other base models and Any
  repository; the text search narrows within the choice. HuggingFace ANDs
  repeated filters, so one base model is searched at a time; each is ordered
  most relevant first (the seeded checkpoint's own base, then its relatives).
  An architecture with no base models there starts on a search for its name
  (without a parenthesis: "Qwen3.6 35B-A3B"), the user's to clear; with base
  models no name is added, since many LoRAs trained on a model do not repeat
  its name. The same holds for ModelScope (`37`).
  A LoRA's base is whatever its author wrote, so a withdrawn repository keeps
  its name: Mage-Flow lists `microsoft/…` first and the community mirror
  second (no real Mage-Flow LoRA names either yet).
- A HuggingFace or disk install is labelled with the file's path without its
  extension; trigger words and tags start empty, nsfw off, stage guessed from
  the name like a Civitai file.
- The stage guess reads every name a file goes by (François, 2026-09-18):
  its own path — folders included, since a repository often says it there —
  and what the site calls the thing it came from: a Civitai version's name, a
  HuggingFace or ModelScope repository's. "high noise" and "low noise" win
  everywhere first; only then the short forms the repository sites use —
  `_HIGH`, `_LOW`, `hn`, `ln` — and only as words of their own, so `highres`
  and `shallow` guess nothing. `LoraManager.stageOf(names*)`. The HuggingFace listing's LFS hash is stored
  and verified after the download, which uses the saved HuggingFace token.
- Installing again what is installed keeps the entity and its tuning and
  fetches whatever file is missing, for every source. The browsers say so
  first: a result tile whose Civitai model, HuggingFace or ModelScope
  repository this architecture already holds a LoRA from carries a **✓
  installed** chip naming it, as does the exact row installed — a Civitai
  version, a HuggingFace or ModelScope file. It is reactive: a tile is marked
  as soon as its install starts, without a re-search. Only this architecture's
  LoRAs count; the same file installed from another architecture's browser is
  a separate entity with its own files.
- **Official LoRAs**: the card's LoRA section lists, under its installed ones,
  the entries of `reference/loras.json` for that architecture that are not
  installed, each with its description, origin and an **Install** button
  (the download size in its tooltip). Installing one saves the entry as it is
  (label, description, strength, stages) as an ordinary LoRA: tunable,
  deletable, and offered again once deleted.
- A file tag's tooltip on the card names the file and where it comes from; the
  picker's search matches the origin too (a Civitai model id, a repository).
- Adopting a folder: a `<digits>-<name>` file with a Civitai model id (sidecar
  or folder suffix) becomes a Civitai file, anything else a disk file under its
  own name — a hand-copied file is used where it lies.

## Shape

- `LoraFile(fileName, source: ModelSource, stage, sizeBytes, sha256)`
  (`shared/.../Loras.scala`): `fileName` is the name inside the LoRA's folder
  (`<fileId>-<filename>` for Civitai, the file's own name otherwise), what
  `storagePathOf`/`requestPathOf` append and what keys the transfer;
  `displayName` and `origin` are derived from the source. `Lora` has no
  `civitaiModelId`; its id is `<architectureId>-<slug>-<source>` with the
  Civitai model id, the HuggingFace repository owner, or `local`.
- `InstallLoraRequest(architectureId, source, grouping)`. A source carries a
  *list*, since what is ticked installs in one go: `CivitaiFiles(modelId,
  files: List[CivitaiFileRef(versionId, fileId)])`, `HuggingFaceFiles(repo,
  filenames)`, `ModelScopeFiles(repo, filenames)`, `LocalFiles(paths)`,
  `Catalog(loraId)`. `LoraGrouping` is `Together` (one LoRA — the one an
  identical install already made, so a re-install still fetches only what is
  missing), `Separate` (one each) or `Into(loraId)`. `LoraDownloadJob` is keyed
  by `(loraId, fileName)`.
- `Architecture.huggingFaceBaseModels` (not in the architecture form yet; kept
  on edit); `GET /api/hf-search` takes `baseModel`; `HuggingFaceBrowser` takes
  `baseModels`, passed by `LoraInstallButton`.
- `GET /api/loras/catalog` → `List[Lora]`, read from
  `backend/resources/reference/loras.json` by `StorageService.loraCatalog`
  once at start, like the runtime rules; nothing is seeded into
  `~/.config/drift/loras/`.
- `components/InstalledLoras.scala` matches an architecture's installed LoRAs
  against what a browser row shows (`fromCivitaiModel`, `fromCivitaiVersion`,
  `fromHuggingFace`, `fromModelScope`, each answering the LoRA's label) and
  renders the chip; the three browsers take it as an `installed` signal,
  `InstalledLoras.none` for a browser opened for a model.
- `LoraManager.place` saves what an install fetched where its grouping says
  and queues the files: `Into` appends to that entity (refusing another
  architecture's), `Separate` gives each file its own id — `freeId` takes the
  first of `<id>`, `<id>-2`, … not already holding those files, so the same
  install twice is still one LoRA — and `Together` derives one id from the
  group's label. A LoRA's name comes from what it holds: a Civitai model's
  name, its version's too when the model has several and the LoRA holds only
  that one; a repository's own name when several of its files travel together,
  the file's path without its extension when one does.
- `LoraManager` fetches by source: Civitai through the shared `Downloader`
  (URL rebuilt from version and file ids), HuggingFace through
  `HuggingFaceDownloads.lookup` (size, hash) and `downloadTo` (straight into
  the LoRA folder, not the shared HuggingFace cache), disk as a copy through a
  `.part` file. Frontend: `components/LoraInstallButton.scala`,
  `pages/architectures/LoraSection.scala`, `services/LoraService`
  (`LoadCatalog`).

## Notes

- The official LoRAs are a list to install from, not seeded built-ins: a LoRA
  is tuned (strength, nsfw, stages) and removed by the user, which a built-in
  rewritten on every start would forbid.
- The catalog holds SenseNova's own 8-step LoRA and, at François's request,
  lightx2v's 4-step LoRA for HunyuanVideo 1.5 (not Tencent's, repackaged by
  Comfy-Org, whose repository the sd-cpp doc names), and Comfy-Org's rank-128
  extraction of Boogu Image's Turbo distillation, which spares a second 20 GB
  download when the Base checkpoint is already on disk (the repository holds
  two builds; the hotfix one is offered). Of the three models added on
  2026-09-18 it is the only one with a LoRA worth offering: Boogu is the only
  vendor that ships its distillation as a whole checkpoint *and* has it
  extracted as a LoRA, while ERNIE Image ships Turbo as its own checkpoint
  (seeded) and HiDream O1's distilled Dev checkpoint is likewise a model, not
  a LoRA. The turbo LoRAs on Civitai and HuggingFace for those two are
  community work, not the vendors'.
- A catalog entry's files are what its vendor publishes, verified with a
  `HEAD` like the seeded models (`23`), with the LFS size and sha256 written
  in. Only architectures whose seeded diffusion model is not already distilled
  get a speed LoRA (LTX's and Krea 2's seeds are).
- A HuggingFace LoRA has no sidecar: an orphaned HuggingFace or disk folder
  adopts as disk files, which is exact as long as the files are there.
- The switch dropped every Civitai-only field; entities written before it are
  rewritten once (each file becomes `<fileId>-<fileName>` with a Civitai
  source), or skipped with a warning by `StorageService.list` and re-adopted.

## Post-v1

- Editing `huggingFaceBaseModels` in the architecture form.
- Candidates for the catalog: lightx2v's Lightning LoRAs (Qwen Image 2512,
  Wan 2.2), Krea 2's style LoRAs in `Comfy-Org/Krea-2`, and a 4- or 8-step LoRA
  for Qwen Image 2.1 when one is published — none existed on 2026-09-20, and
  2512's Lightning LoRAs do not fit its 7B DiT ([`23`](23-seeded-vendor-models.md)).
