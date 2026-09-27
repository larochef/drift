# Bug 20 — A single Civitai model with `"downloadCount": null` fails the decode of the whole search page

**Status:** fixed (pending compile/verify — written while the mill lock was held)
**Severity:** medium (an entire base-model listing renders as "No models found")
**Files:** `shared/src/drift/shared/Civitai.scala`

## Context

The Civitai browser for the `flux.2-klein-9B` architecture showed no models at all,
while the same request against `civitai.com/api/v1/models` returned 25 items. Other
architectures (MiniMax H3, Krea 2, LTXV 2.3, Wan, Flux.2 D, Qwen) listed fine.

## Symptom

`GET /civitai-search?types=CHECKPOINT&baseModels=Flux.2 Klein 9B&nsfw=true` returns
`{}` (zero items) although Civitai returns a full page for the identical upstream
request. The backend log shows `Civitai search decode error`.

## Root cause

Two models in the Klein listing — `Miraclein NSFW` (2453960) and `KleiNova NSFW`
(2547526) — are served with a literal null count:

```json
"stats": { "downloadCount": null, "thumbsUpCount": 609, ... }
```

`CivitaiModelListStats.downloadCount` is `Int = 0`, and jsoniter applies a field
default only when the key is **absent** — a JSON `null` for an `Int` throws. One
such entry aborts the decode of the entire `CivitaiSearchResponse`, the catch in
`CivitaiClient.search` logs and returns an empty result, and the browser says
"No models found" for the whole page.

## Fix

`CivitaiModelListStats` gets a permissive companion codec (the same Raw-decode
pattern `CivitaiModelListInfo` already uses): decode through `Option[Int]` fields
and read `null` as 0. Encoding is unchanged. The codec is picked up wherever the
stats are nested (list entries and model detail).

## Verification

```
curl 'http://localhost:4321/civitai-search?types=CHECKPOINT&baseModels=Flux.2%20Klein%209B&nsfw=true&limit=25'
```

must return a full page including `Miraclein NSFW [Generation & Edit] [Flux2Klein]`,
and the UI browser for Flux.2 Klein 9B must list models.
