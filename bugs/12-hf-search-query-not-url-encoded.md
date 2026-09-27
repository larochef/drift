# Bug 12 — HuggingFace search query is not URL-encoded → queries with spaces return "No models found"

**Status:** open
**Severity:** low (search only, but breaks on the most common queries)
**Files:** `backend/src/drift/backend/routes/HuggingFaceRoutes.scala`

## Context

`GET /api/hf-search?q=...` proxies to the HuggingFace Hub API. The frontend (`HuggingFaceBrowser`, `ModelSearchPage`) passes the raw typed query, which usually contains spaces.

## Symptom

Searching for "stable diffusion" (with a space) returns an empty result set ("No models found"), even though the model exists. Single-word queries work.

## Root cause

The query is interpolated raw into a `uri"..."` string:

```scala
// HuggingFaceRoutes.scala (~lines 43–46)
val response = hfRequest
  .get(
    uri"https://huggingface.co/api/models?search=$query&limit=$limit&skip=$offset&sort=$sortVal&direction=$dirVal&full=false&config=false"
  )
  .send(hfClient)
```

A raw space in the URI is invalid; sttp's `uri"..."` interpolator (or the underlying HttpURLConnection) fails, the exception is swallowed by `catch { case _: Exception => List.empty }` (lines ~53–55), and the user sees an empty list with no error.

## Suggested fix

Build the URI with parameter helpers so values are encoded:

```scala
uri"https://huggingface.co/api/models"
  .addParam("search", query)
  .addParam("limit", limit.toString)
  .addParam("skip", offset.toString)
  .addParam("sort", sortVal)
  .addParam("direction", dirVal.toString)
  .addParam("full", "false")
  .addParam("config", "false")
```

Also consider logging (debug) the caught exception so failures are diagnosable — the current `catch` swallows everything.

## Verification

`curl 'http://localhost:4321/api/hf-search?q=stable%20diffusion'` (or via the browser UI, typing a query with a space) must return results.
