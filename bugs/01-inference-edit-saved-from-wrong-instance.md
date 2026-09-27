# Bug 1 — Inference edit: the save reads from the wrong card instance, all edits are lost

**Status:** fixed
**Severity:** critical (user-visible data loss on the Inference page)
**Files:** `frontend/src/drift/frontend/pages/inference/InferencePage.scala`, `frontend/src/drift/frontend/pages/inference/InferenceEditCard.scala`

## Context

drift is a Scala 3 monorepo: a Laminar (Scala.js) SPA in `frontend/`, a tapir + Netty backend in `backend/`, shared models in `shared/`. Data is persisted as JSON files at `~/.config/drift/<entity>/<id>.json` (see `backend/src/drift/backend/storage/StorageService.scala`). Frontend forms are plain classes holding `Var` state, and a `snapshot()` method is read when the user clicks a save/create button.

## Symptom

On the Inference page: click **Edit** on a run configuration, change the label, a checkpoint assignment, or an override parameter, click **Save** — the saved JSON is unchanged (the original values are re-saved). Nothing you typed is persisted.

## Root cause

There are **two separate `InferenceEditCard` instances** in play; the user edits one, and the other is what gets saved.

`InferencePage.scala`:

```scala
// line 16
private val editCard = Var(Option.empty[InferenceEditCard])
```

The card that is **displayed** is created fresh inside the `viewData` signal (line ~30), with the architecture's real checkpoints:

```scala
// InferencePage.scala, inside viewData.map
if (editing.contains(rm.id)) {
  val arch = archs.find(_.id == rm.architectureId)
  val cps = arch.map(_.checkpoints).getOrElse(Nil)
  div(
    InferenceEditCard(rm, cps, service.allModels),   // <-- DISPLAYED instance (fresh per render)
    div(
      cls := "buttons",
      button(cls := "button is-success", "💾 Save",
        onClick --> (_ => handleSave(rm.id))),
      ...
```

The card that is **saved** is a different instance created in `startEdit` (line ~64–67), with **`Nil` checkpoints**:

```scala
private def startEdit(rm: RunningModel): Unit = {
  editingId.set(Some(rm.id))
  editCard.set(Some(InferenceEditCard(rm, Nil, service.allModels)))  // <-- SAVED instance
}

private def handleSave(id: String): Unit = {
  editCard.now().foreach { card =>
    service.push(Command.Update(id, card.snapshot()))  // snapshots the OTHER instance
  }
}
```

`InferenceEditCard` (see `InferenceEditCard.scala` lines 13–24) keeps its own independent state (`labelVar`, `assignments`, `paramEditor`) initialized from `rm` at construction. So `card.snapshot()` on the stashed instance always returns the *original* values: the label, assignments and parameters the user edited in the displayed instance never make it to `Command.Update`.

Secondary symptom of the same root cause: the stashed instance was built with `Nil` checkpoints, so even its (hidden) checkpoint section renders the "Select an architecture to assign models to checkpoints." placeholder instead of rows.

## Suggested fix

Make the stashed instance the single source of truth:

1. In `startEdit`, resolve the architecture's checkpoints from `service.architectures.now()` and construct the card with the real checkpoint list (same `cps` computation the `viewData` branch does).
2. Render **that same instance** from `editCard` (e.g. `editCard.now()` inside the `editing.contains(rm.id)` branch) instead of constructing a new `InferenceEditCard` inside `viewData`.

Either direction works (one instance total), but keeping `editCard` as the owner of the form is the smaller change: the `viewData` signal is re-evaluated whenever `service.running`/`service.architectures` fire, which would keep recreating the displayed instance and losing in-flight keystrokes even after this fix.

## Verification

1. Create a run config with at least one checkpoint assignment and one override parameter (JSON lands in `~/.config/drift/running/`).
2. Edit it: change the label, swap the checkpoint assignment, change a parameter.
3. Save, then inspect the JSON file — all three changes must be present.
4. Reopen the edit form — fields must show the *current* saved values (see also Bug 7).
