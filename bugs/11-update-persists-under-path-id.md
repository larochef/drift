# Bug 11 — Update endpoint persists the body under the *path* id, even if the body's id differs

**Status:** open
**Severity:** low (latent; frontend currently always sends body.id == path id)
**Files:** `backend/src/drift/backend/routes/RouteHelpers.scala`, `shared/src/drift/shared/Api.scala`

## Context

All entity updates go through the generic `endpointsFor` helper in `RouteHelpers.scala`. The update endpoint signature is `PUT /api/<entity>/<pathId>` with a JSON body, and the stored file name is derived from the id.

## Symptom / risk

If the request body's `id` field differs from the URL path id, the entity is saved to `<pathId>.json` while its JSON content says `"id": <bodyId>`. From then on:

- `GET /api/<entity>/<bodyId>` (matching the entity's own id) returns `None` — the file is `<pathId>.json`.
- `list` returns the entity with an id that doesn't match its file name; deleting by the body id works (path id), deleting by the list-reported id does not.

The frontend currently sends `body.id == path.id`, so this is latent, but nothing enforces it.

## Root cause

```scala
// RouteHelpers.scala (~lines 26–30)
updateEp.serverLogicSuccess[Identity] { (id, v) =>
  storage
    .get[T](entityType, id)
    .map(_ => storage.save[T](entityType, id, v))   // saves `v` under the PATH id, ignoring v's own id
}
```

## Suggested fix

Pick one consistent rule and enforce it:

- **Reject mismatches** (simplest): if `extractId(v) != id`, return a 400/409 error (tapir `serverLogic` with a `Failure` output) instead of saving.
- **Or rename**: save under `extractId(v)` and, if different from the path id, also `storage.delete(entityType, id)` to remove the old file.

## Verification

With a rule in place: `PUT /api/architectures/old-id` with a body whose `id` is `new-id` either fails with a clear error or results in exactly one file (`new-id.json`) whose content id matches — never a mismatched pair.
