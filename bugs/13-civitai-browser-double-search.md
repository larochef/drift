# Bug 13 — Civitai browser: double search on open, and `baseModelsVar` never synced with incoming models

**Status:** partially fixed
**Severity:** low (wasted request on every open; the "re-search on change" intent is dead)

> **Symptom 1 (double search on open) is fixed.** The observer now subscribes to
> `baseModelsVar.signal.changes` instead of `.signal`, so it no longer replays the
> current value on subscribe alongside the mount callback.
>
> **Symptom 2 (the sync intent is dead) is still open.** `baseModelsVar` is written
> nowhere after construction, because `SourceTypeSelector` passes a plain `List`
> snapshot rather than the `Signal` it receives. Closing it means threading the
> `Signal` into `CivitaiBrowser`.
**Files:** `frontend/src/drift/frontend/components/CivitaiBrowser.scala`, `frontend/src/drift/frontend/pages/architectures/SourceTypeSelector.scala`

## Context

`CivitaiBrowser` is opened from `SourceTypeSelector` (Civitai source type → "Browse"). It searches `GET /api/civitai-search` and renders model cards. `SourceTypeSelector` receives `civitaiBaseModels` as a `Signal[List[String]]` (in `ArchitectureCard` it is `Val(a.civitaiBaseModels)`) and passes it to the browser as a plain `List`.

## Symptom

1. Every time the browser modal opens, **two** search requests fire for the same query (visible in the network tab / backend debug logs).
2. The comment "Re-search when baseModels change" is aspirational: the `baseModelsVar` is initialized once and never updated, so the observer never re-fires.

## Root cause

```scala
// CivitaiBrowser.scala (~lines 75, 81–88)
private val baseModelsVar = Var(civitaiBaseModels)

lazy val element: HtmlElement =
  div(
    ...
    onMountCallback { _ =>
      searchBus.writer.onNext(initialQuery)          // search #1
    },
    // Re-search when baseModels change
    baseModelsVar.signal --> Observer { _ =>          // fires on the INITIAL value too → search #2
      searchBus.writer.onNext(searchQuery.now().trim)
    },
    ...
```

A Laminar signal observer receives the current value immediately upon subscription, so the mount callback and the observer both trigger a search. And `baseModelsVar` is written nowhere else, so the "on change" behavior never happens.

## Suggested fix

- Keep exactly one trigger: drop either the `onMountCallback` search or the `baseModelsVar.signal --> Observer` subscription.
- If re-searching when base models actually change is wanted, make the incoming list reactive: accept `civitaiBaseModels: Signal[List[String]]` and either (a) `baseModelsVar`-sync it with `Signal(...).foreach(baseModelsVar.set)` guarded to skip the initial value, or (b) derive the search parameters directly from the signal in the `searchBus` pipeline.

## Verification

Open the Civitai browser once; the backend must log exactly one `Civitai search URI` line for it.
