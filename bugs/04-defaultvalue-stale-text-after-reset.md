# Bug 4 — `defaultValue` inputs + `reset()` → stale text concatenated into saved values

**Status:** fixed
**Severity:** high (silent data corruption on repeated create/edit)
**Files:** `frontend/src/drift/frontend/pages/inference/InferenceForm.scala`, `frontend/src/drift/frontend/pages/architectures/ArchitectureForm.scala`, `frontend/src/drift/frontend/components/ParamEditor.scala`, `frontend/src/drift/frontend/components/CheckpointEditor.scala`

## Context

Laminar detail that this depends on: `defaultValue := x` on an `<input>` is a **one-shot, uncontrolled** attribute — the DOM keeps its own value afterwards, and re-binding never re-applies it. `onInput.mapToValue --> someVar` only pushes DOM → Var, never Var → DOM. So clearing a `Var` (via `reset()`) does NOT clear the visible input.

## Symptom

- **Inference create form**: create a config with id `my-config`, save. Open the form again and type a new id without fully clearing the field → the saved id is `my-config<what-you-typed>` (or any other concatenation of leftover text).
- **Architecture edit**: edit architecture A's label, save, then edit architecture B — B's label input initially **shows A's label**. Saving without touching the field is fine (the Var holds B's value), but typing into the field concatenates onto A's leftover text.
- **Parameters / checkpoints (worst case)**: edit architecture A (e.g. 8 params), then B (8 params). The parameter rows visually show **A's keys/values** while the underlying state holds B's. Rows you don't touch save B's values (correct, invisible); rows you do touch save `A's leftover text + your edit`. Same for the checkpoint name/flag/family inputs.

## Root cause

All of these forms use `defaultValue` inputs whose state is mirrored in `Var`s, and their `reset()` methods only reset the `Var`s:

```scala
// InferenceForm.scala (~lines 49–54, 61–67)
input(cls := "input", placeholder := "e.g. my-config",
      defaultValue := idVar.now(),        // one-shot; DOM keeps old text after reset
      onInput.mapToValue --> idVar)
...
def reset(): Unit = { idVar.set(""); ... }   // Var cleared, DOM NOT cleared
```

```scala
// ArchitectureForm.scala (~lines 67–73, 80–87) — same pattern for id/label
```

The row editors make it worse via keyed reconciliation. `ParamEditor` and `CheckpointEditor` render rows with:

```scala
// ParamEditor.scala (~line 27) / CheckpointEditor.scala (~line 34)
children <-- rows.signal.split(_.id) { (id, initial, rowSignal) =>
  div(...
    input(defaultValue := initial.key, onInput.mapToValue --> (key => ...)),
    ...)
}
```

and `reset()` reassigns ids starting at 0:

```scala
// ParamEditor.scala reset() (~lines 13–18)
def reset(newPairs: List[(String, String)] = Nil): Unit = {
  nextId = 0
  rows.set(newPairs.map { case (k, v) => val id = nextId; nextId += 1; ParamRow(id, k, v) })
}
```

Because Laminar's `split(_.id)` **reuses the existing DOM element for an unchanged key**, the uncontrolled inputs keep their old text even though the `Var` now holds the new values. Any edit you make in such a row merges with the stale text.

## Suggested fix

Preferred: make these inputs **controlled** so the Var drives the DOM:

- Replace `defaultValue := x` with `value <-- someVar.signal` in `InferenceForm` (id, label), `ArchitectureForm` (id, label), `ParamEditor` (key, value) and `CheckpointEditor` (name, flag, familyId), keeping the existing `onInput --> var` bindings.
- If the row editors must stay uncontrolled (e.g. to avoid caret jumps), give `reset()` **fresh, never-reused ids** instead of restarting `nextId` at 0 (e.g. keep a monotonically increasing counter across resets, or prefix ids with a per-reset generation counter) so `split` recreates the DOM.

## Verification

1. Inference: create config `a1`, save; open form again, type `b1` into the id field without selecting-all first — saved id must be exactly `b1` (or the field must visibly be empty on reopen).
2. Architectures: edit `wan-2.2-14B` label, save; edit `wan-2.2-5B` — label and parameter fields must show *wan-2.2-5B's* values, not the previous architecture's.
3. Change one parameter value in the second edit, save, inspect the JSON — untouched params must be the second architecture's, the touched one must be exactly what was typed.
