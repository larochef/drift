# Bug 5 — `builtIn` flag is always clobbered to `false` on save

**Status:** fixed
**Severity:** medium (entity semantics change after one edit; deleted built-ins resurrect on restart)
**Files:** `frontend/src/drift/frontend/pages/architectures/ArchitecturesPage.scala`, `frontend/src/drift/frontend/pages/architectures/ModelForm.scala`, `backend/src/drift/backend/storage/StorageService.scala`

## Context

Seed data lives in `backend/resources/reference/architectures.json` and `families.json` and is loaded on startup by `StorageService.seedFromResource` (see `StorageService.scala` lines 62–117): items missing from disk are re-saved, and on-disk items **not** in the reference list get `builtIn = false`. Seeded entities have `builtIn: true`, which the UI uses to hide Edit/Delete (`ArchitectureCard.scala` line ~29: `if (!a.builtIn)` shows the buttons) and to gate the delete endpoint (`Routes.scala`: `canDelete = !_.builtIn`).

## Symptom

Edit any seeded architecture (e.g. `wan-2.2-14B`) and save — the saved JSON now has `"builtIn": false`. Consequences:

1. The card gains Edit/Delete buttons for what was a reference entry.
2. Deleting it succeeds, but the *next server start* re-seeds it from the reference file (it is in the ref list and missing on disk), so the deleted entity comes back to life.
3. Same for models created via `ModelForm.snapshot()` — the `builtIn` field is never carried over.

## Root cause

Both save paths construct the entity with a hard-coded flag:

```scala
// ArchitecturesPage.scala handleSave (~lines 63–79)
service.push(Command.Update(id,
  Architecture(id, label, checkpoints, params, civitaiBaseModels, builtIn = false)))
```

```scala
// ModelForm.scala snapshot() (~line 56)
Model(mid, familyId, mlabel, source, "safetensors", params)   // builtIn defaults to false
```

The current `builtIn` value of the entity being edited is read from the card/form but never propagated into the saved value.

## Suggested fix

Preserve the existing flag on update:

- `ArchitecturesPage.handleSave(id)`: look up the current entity (`service.architectures.now().find(_.id == id)`) and use its `builtIn` instead of `false` (fall back to `false` if not found).
- `ModelForm`: add a `builtIn: Boolean` constructor parameter (default `false`) and include it in `snapshot()`; pass the current model's flag when the form is used for editing.

## Verification

1. Edit the label of a seeded built-in architecture, save.
2. `~/.config/drift/architectures/<id>.json` must still contain `"builtIn": true`; the card must not show Edit/Delete.
3. (Optional) delete it, restart the backend, confirm it re-appears — this is the seeding behavior and is expected to stay; the point is that one edit must not silently demote the flag.
