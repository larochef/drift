# 37 — ModelScope

**Status:** done; search, paging, a repository's examples, files and card, a
model download (a file named with spaces) and a LoRA install checked against
ModelScope from a test backend and in chromium; not yet used for a real model
**Depends on:** 05, 33, 34, 36

ModelScope (modelscope.cn) hosts what HuggingFace does and more: the Chinese
labs publish there first, and its AIGC section is a Civitai-like community of
LoRAs with cover images. It copies HuggingFace's shapes in places — repository
ids, `resolve/<revision>/<path>` downloads — but its API is its own, so drift
talks to it through its own client, never HuggingFace's.

## What it does

- A model can come from ModelScope: the Add Model form has a ModelScope source
  (`owner/name` and a file path, or Browse), and every architecture's LoRA
  section a **+ ModelScope** button.
- The ModelScope browser opens on results at once — ModelScope lists every
  model without a query — as example tiles: a repository's first example
  (the next is tried when one does not load), a LoRA tag, the example count,
  downloads and stars. It sorts by downloads, stars, update or creation date,
  or ModelScope's own order; **LoRAs only** keeps what ModelScope files as
  LoRAs, **With examples only** hides the rest of a page; both start ticked
  when browsing for a LoRA. **Load More** pages 24 at a time.
- For a LoRA, the architecture's `modelScopeBaseModels` filter the search, all
  of them at once by default ("Any of its N base models"): ModelScope ORs the
  values of one criterion, unlike HuggingFace. One, or any base model, can be
  chosen instead.
- An opened repository has Examples (its versions' cover images with their
  prompts, or its image and video files when it has no covers; an example that
  does not load is left out), Files (weights and safetensors indexes are
  selectable) and Model card (its README, its `<Gallery />` made of the
  covers).
- A ModelScope file downloads into `~/.cache/drift/modelscope/<owner>/<name>/<path>`,
  verified against the sha256 ModelScope lists for every file; a
  `*.safetensors.index.json` brings its shards (`34`). A LoRA downloads into
  the LoRA store like any other (`33`).
- The Model Cache's on-disk view has a ModelScope tab, grouped by repository;
  deleting a row removes that file.

- A repository that fails to open says so above its tabs, with a **Retry**
  (`24`): ModelScope answers its own failure code on a tree request often
  enough that the modal must offer one, rather than showing an empty
  repository.

## Shape

- `ModelScope(repo, filename)` model source, `ModelSourceType.ModelScope`,
  `CachedFileKind.ModelScope`, `LoraInstallSource.ModelScopeFile`,
  `Architecture.modelScopeBaseModels` (seeded, not in the form; kept on edit).
  `ModelExample` and `ModelExampleGallery` (renamed from HuggingFace's) serve
  both browsers.
- `shared/.../ModelScope.scala`: `ModelScopeModelInfo`, `ModelScopeSearchPage`,
  `ModelScopeFileInfo`, `ModelScopeModelDetail`, `ModelScopeModelCard`,
  `ModelScopeSorts`; `GET /api/ms-search` (`q`, `page`, `sort`, repeated
  `baseModel`, `lorasOnly`), `GET /api/ms-model/{owner}/{name}`,
  `GET /api/ms-card/{owner}/{name}`, failures as 502 with ModelScope's reason.
- `backend/.../modelscope/ModelScopeApi.scala` (its own `HttpClient`):
  `PUT /api/v1/dolphin/models` (criteria `base_model`, `aigc_type`),
  `GET /api/v1/models/{repo}`, `GET /api/v1/models/{repo}/repo/files?Recursive=true`,
  `downloadRequest` (URL and headers). `ModelScopeDownloads` (targets, lookup, `download` with
  shards, `downloadTo` for LoRAs) over the shared `Downloader`.
  `ModelCache.modelScopeRoot`/`modelScopePath`; `CacheInventory.listModelScope`.
- Frontend: `services/ModelScopeService`, `components/ModelScopeBrowser`,
  `BrowserServices.modelScope`; the source selector, `ModelForm`,
  `LoraInstallButton`, `OnDiskCacheView`.

## Notes

- A model's `MuseInfo` (the AIGC section's data, where the cover images are)
  is an object in a search and a JSON string in a model's info; both are read.
  Error codes do not fit an `Int` (`10010205001`); answers carry their code in
  the body, often with an HTTP 200.
- A download is redirected to ModelScope's CDN with a signed query that can
  hold raw spaces, which the JDK will not follow: the client resolves the
  redirect with a one-byte ranged GET (a HEAD is answered 200 without it),
  percent-encodes the location and hands that URL to the downloader, anew at
  each start so a resume never reuses an expired signature.
- A mirrored repository's covers are at times Git LFS pointer files served as
  `image/png` (5 of `AI-ModelScope/pixel_art_style_lora_z_image_turbo`'s 7): a
  tile moves to the next example, the Examples tab leaves them out.
- Base model names carry revisions (`Tongyi-MAI/Z-Image-Turbo@master`) and
  ModelScope's own ids (`MusePublic/Qwen-image@v1`); each is a separate value.
  No LoRA declares Wan 2.2, Mage-Flow, SenseNova, HunyuanVideo 1.5 or the chat
  models there yet. Wan 2.2 LoRAs name Wan 2.1 bases (ModelScope's AIGC
  section offers no 2.2 one), mixed with real 2.1 LoRAs: Wan 2.2 keeps no
  ModelScope base models on purpose and starts on a search for its name
  instead — a text search over a model mismatch (François).
- A ModelScope token (Settings, `AuthProvider.ModelScope`, or
  `MODELSCOPE_API_TOKEN` as ModelScope's SDK reads it) goes with every request
  to modelscope.cn the way the SDK sends it: `Authorization: Bearer` for the
  newer endpoints and the `m_session_id`/`modelscope_session` cookies for the
  older ones. It never goes with the signed CDN address a download is
  redirected to, and the API client follows no redirect, so nothing carries
  it elsewhere. A wrong token does not stop public requests (checked); a
  refused download (401/403) names the token as the likely fix.

## Post-v1

- Sharing the ModelScope SDK's cache instead of drift's own tree.
- The generations people made with a LoRA on ModelScope (its image count is in
  the Muse data), like Civitai's posts.
