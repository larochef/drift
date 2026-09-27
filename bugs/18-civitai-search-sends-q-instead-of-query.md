# Bug 18 — Civitai search sends `q`, but the API's parameter is `query` → searches return unrelated models

**Status:** fixed
**Severity:** medium (every Civitai text search was broken; base-model browsing mostly survived)
**Files:** `backend/src/drift/backend/routes/CivitaiRoutes.scala`

## Context

`GET /civitai-search` proxies to `https://civitai.com/api/v1/models`. The
`CivitaiBrowser` opens filtered by the architecture's `civitaiBaseModels` and lets the
user type a text query on top.

## Symptom

Searching for e.g. `krea2` shows one or two results (or none), while civitai.com shows
dozens of matching models for the same search.

## Root cause

Two compounding problems:

1. The text query was passed as `q`:

   ```scala
   query.foreach { q =>
     uri = uri.addParam("q", q)
   }
   ```

   Civitai's parameter is `query`; unknown parameters are **silently ignored**, so the
   API answered with the plain most-downloaded listing as if no search had been typed.
   (Verified live: `?q=krea2` returns Realistic Vision, DreamShaper, …; `?query=krea2`
   returns the actual Krea 2 models.)

2. A client-side substring re-filter — presumably added to compensate for (1) —
   then kept only entries whose name/description/tags literally contain the query:

   ```scala
   val filtered = filteredByType.filter { m =>
     m.name.toLowerCase.contains(lower) || ...
   }
   ```

   That cut the 25 unrelated most-downloaded models down to whichever ones happened to
   contain the string, i.e. usually ~0–1 results.

Two lesser gaps fixed in the same pass:

- No `Authorization: Bearer` token was sent on search/detail, although the download
  path sends one and a token is configured in Settings — anonymous API calls do not
  see account-gated models. The client now resolves `AuthTokens.civitai` per request.
- Paging used `offset`, which the Civitai API also ignores (it pages with an opaque
  `metadata.nextCursor`). The endpoint now takes a `cursor` and returns
  `CivitaiSearchResult(items, nextCursor)`; the browser gained a **Load More** button
  like the HuggingFace one. The bogus `meta=author` parameter was dropped too.

## Verification

With the server running:

```
curl 'http://localhost:4321/civitai-search?limit=3&q=krea2'
```

must return Krea-named items and a `nextCursor`; repeating with `&cursor=<that value>`
must return different items. In the UI: architecture → checkpoint → + Add Model →
Civitai → Browse shows the same models as civitai.com filtered by the base model, and
Load More appends the next page.
