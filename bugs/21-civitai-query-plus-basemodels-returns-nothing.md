# 21 — Civitai: `query` + `baseModels` together return an empty list

**Status:** fixed (2026-09-03)

## Symptom

Searching for "turbo" in the LoRA browser of the Flux.2 DEV architecture found
nothing, though matching models exist (e.g. models 2266313 and 2267613, both
`type: LORA`, `baseModel: "Flux.2 D"` — exactly what the filter asks for).

## Cause

Civitai's text-search index does not support the `baseModels` parameter.
Verified live against `GET /api/v1/models`:

- `types=LORA&baseModels=Flux.2 D` (no query) → 82 items, both models present
- `query=turbo&types=LORA&baseModels=Flux.2 D` → **0 items**
- `query=turbo&types=LORA` (no baseModels) → pages of items

The combination silently annihilates everything rather than erroring — the
same genre of trap as `q` being ignored (bugs/18) and the absent `nsfw`
default (bugs/19).

## Fix

`CivitaiClient.search`: when a query is present, `baseModels` is not sent;
the base-model scoping is applied drift-side instead, keeping items with any
version whose `baseModel` is in the architecture's list. The no-query path
still filters server-side, unchanged.

## Second failure mode (same day)

The first fix was not enough: the browser pages with `limit=25`, and
Civitai's query search ranks by opaque relevance (ignoring `sort`) **and
returns a different result set at different page sizes** — four 25-item pages
of `query=turbo` contained neither turbo LoRA while a single 100-item call
contained both. So when filtering locally, drift now requests 100-item pages
from Civitai and follows the cursor (up to three extra pages) until the
requested count is met. Verified in the real UI from the Flux.2 DEV card:
both models render in browse and under a "turbo" search.

Related trap while testing: the browser's search box does not re-search when
its text is cleared — the previous (possibly empty) result list stays until
Search/Enter is pressed again.
