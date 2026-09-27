# Bug 6 — Checkpoint `required` flag is always saved as `true`

**Status:** fixed
**Severity:** medium (data loss on edit: `required: false` checkpoints become `required: true`)
**Files:** `frontend/src/drift/frontend/components/CheckpointEditor.scala`, `shared/src/drift/shared/Api.scala`

## Context

`CheckpointRef` in `shared/src/drift/shared/Api.scala` is:

```scala
case class CheckpointRef(
    name: String,
    familyId: String,
    sdCppFlag: String,
    required: Boolean = true
)
```

Reference seeds use `required: true`, but the type allows `false`. `CheckpointEditor` is the UI for editing the checkpoint list inside `ArchitectureForm`.

## Symptom

Edit an architecture that has a checkpoint with `"required": false`, save — the checkpoint comes back as `"required": true`. The flag cannot be changed from the UI at all.

## Root cause

`CheckpointEditor`'s row model simply has no `required` field, and `snapshot()` always reconstructs with the default:

```scala
// CheckpointEditor.scala (lines ~7–12)
case class CheckpointRow(id: Int, name: String, sdCppFlag: String, familyId: String)
```

```scala
// CheckpointEditor.scala snapshot() (~lines 29–30)
def snapshot(): List[CheckpointRef] =
  rows.now().map(r => CheckpointRef(r.name, r.familyId, r.sdCppFlag))   // required = true, always
```

`init`/`reset` (lines ~16–27) copy only `name`, `sdCppFlag`, `familyId` from the `CheckpointRef`s.

## Suggested fix

- Add `required: Boolean` to `CheckpointRow` (default `true`).
- Populate it from `cp.required` in the `init` and `reset` mappings.
- Add a UI control (checkbox, or a `yes/no` select) per row bound to that field via the same `rows.update(_.map(...))` pattern the other columns use.
- Include it in `snapshot()`: `CheckpointRef(r.name, r.familyId, r.sdCppFlag, r.required)`.

## Verification

1. Set a checkpoint's `required` to `false` in an architecture's JSON under `~/.config/drift/architectures/`.
2. Open the architecture in the editor, change an unrelated field, save.
3. The JSON must still contain `"required": false` for that checkpoint; the UI should let you toggle it.
