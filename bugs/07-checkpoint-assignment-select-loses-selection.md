# Bug 7 — Checkpoint assignment selects never show the currently selected model

**Status:** fixed
**Severity:** medium (state is invisible; interacts badly with Bug 1)
**Files:** `frontend/src/drift/frontend/pages/inference/CheckpointAssignments.scala`

## Context

`CheckpointAssignments` renders one `<select>` per checkpoint of an architecture, letting the user pick which `Model` (by id) is assigned to that checkpoint. The assignment state lives in a `Var[List[(String, String)]]` (`checkpointName -> modelId`) owned by the caller (`InferenceForm` / `InferenceEditCard`).

## Symptom

Open the Inference create form, pick an architecture, select a model for a checkpoint — the select shows the choice, fine. But any time the list re-renders (e.g. the model list finishes loading, the architecture list refires), the select snaps back to the placeholder "Select model (required)" even though the assignment is still stored in the `Var`. In the edit flow (Bug 1), the displayed card therefore always looks unassigned.

## Root cause

The options are built without ever marking the current value as selected:

```scala
// CheckpointAssignments.scala (~lines 39–52)
select(
  cls := "select",
  onChange.mapToValue --> (v =>
    assignments.update { assigns =>
      val filtered = assigns.filter(_._1 != cp.name)
      if (v.nonEmpty) filtered :+ (cp.name -> v) else filtered
    }
  ),
  option(value := "", if (cp.required) "Select model (required)" else "None"),
  familyModels.map(m => option(value := m.id, m.label))   // <-- no `selected := true` anywhere
)
```

Compare with the architecture select in `InferenceForm.scala` (~lines 77–83), which does it correctly:

```scala
option(value := a.id, a.label, if (a.id == archId) selected := true else emptyNode)
```

## Suggested fix

Derive the current selection from the `assignments` Var and mark it:

```scala
val current = assignments.now().find(_._1 == cp.name).map(_._2)
...
option(value := "",
       if (cp.required) "Select model (required)" else "None",
       if (current.isEmpty) selected := true else emptyNode),
familyModels.map(m =>
  option(value := m.id, m.label, if (current.contains(m.id)) selected := true else emptyNode))
```

Note the whole `children <-- allModels.map { ... }` block is re-created when `allModels` fires, so `assignments.now()` at render time is the right source of truth; keep the single-entry-per-checkpoint invariant in the `onChange` handler (already present via `filter(_._1 != cp.name)`).

## Verification

1. Inference create form → select an architecture → pick a model for a checkpoint.
2. Trigger a re-render (e.g. add a model from the Architectures page and come back, or wait for the model list to load after opening).
3. The select must still display the chosen model.
