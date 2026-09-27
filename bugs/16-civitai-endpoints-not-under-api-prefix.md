# Bug 16 — Civitai endpoints are not under the `/api` prefix, and unmatched `/api/*` paths silently return the SPA

**Status:** fixed — 2026-09-11, with `specs/24`: `base` is package-visible
and the Civitai endpoints build on it, and the SPA catch-all answers 404 for
unmatched `/api/*` paths
**Severity:** low (works today; makes every wrong `/api/...` path look like a success)
**Files:** `shared/src/drift/shared/Civitai.scala`, `backend/src/drift/backend/routes/FrontendRoutes.scala`

## Context

Every endpoint in `shared/src/drift/shared/Api.scala` is built from a shared prefix:

```scala
private val base = endpoint.in("api")

val listModels: PublicEndpoint[Unit, Unit, List[Model], Any] =
  base.get.in("models").out(jsonBody[List[Model]])
```

The two Civitai endpoints in `shared/src/drift/shared/Civitai.scala` are not:

```scala
val searchCivitaiModels: PublicEndpoint[...] =
  endpoint                       // <-- bare `endpoint`, no "api" segment
    .in("civitai-search")
    ...

val getCivitaiModelDetail: PublicEndpoint[Int, Unit, Option[CivitaiModelDetail], Any] =
  endpoint                       // <-- same
    .in("civitai-model" / path[Int])
```

## Symptom

The Civitai endpoints live at the server root, not under `/api`:

```
GET /civitai-model/2731187       -> application/json      (works)
GET /api/civitai-model/2731187   -> text/html             (SPA shell, HTTP 200)
```

Nothing is broken at runtime: `ApiClient` uses a base uri of `/`, and the endpoint
definitions carry their own full path, so the frontend calls the right URL.

The real damage is the second line. `FrontendRoutes` serves `index.html` as a
catch-all, so **any** unmatched path — including a mistyped or misremembered `/api/...`
one — answers `200 text/html` instead of `404`. A caller checking only the status code
sees success.

That is not hypothetical: during the ApiClient refactor a boot check reported
`/api/civitai-search HTTP 200` as healthy. It was the catch-all, and the endpoint was
never exercised.

## Root cause

Two independent things compounding:

1. `Civitai.scala` was written with `endpoint` rather than the `base` value that
   `Api.scala` defines — `base` is `private` to `Api.scala`, so it was not reachable
   from the other file and the prefix was simply dropped.
2. The SPA catch-all in `FrontendRoutes` has no exclusion for `/api`, so an unmatched
   API path is indistinguishable from a client-side route.

## Suggested fix

- Make the `base` prefix shared (move it somewhere both files can see, or give
  `Civitai.scala` its own `endpoint.in("api")`) and move the two endpoints to
  `/api/civitai-search` and `/api/civitai-model/{id}`. No compatibility concern — see
  `specs/00-overview.md`, "Standing constraints".
- Make the frontend catch-all refuse to serve `index.html` for paths beginning `/api`,
  returning a real 404 instead. This is the half that matters: it turns every future
  wrong API path into a visible failure rather than a silent HTML response.

## Verification

1. `curl -s -o /dev/null -w '%{content_type}' localhost:4321/api/civitai-model/2731187`
   returns `application/json`.
2. `curl -s -o /dev/null -w '%{http_code}' localhost:4321/api/does-not-exist`
   returns `404`, not `200`.
3. The Civitai browser still searches and lists files (it goes through
   `CivitaiService`, so the path change must land in `Civitai.scala` only).
