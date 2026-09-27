# Bug 2 — Adding a Civitai base model on an architecture card is never persisted

**Status:** fixed
**Severity:** high (silent data loss; removal persists, addition doesn't)
**Files:** `frontend/src/drift/frontend/components/BaseModelManager.scala`, `frontend/src/drift/frontend/pages/architectures/ArchitectureCard.scala`

## Context

drift is a Scala 3 monorepo: a Laminar (Scala.js) SPA in `frontend/`, a tapir + Netty backend in `backend/`, shared models in `shared/`. Data is persisted as JSON files at `~/.config/drift/<entity>/<id>.json`. Architecture cards render a "Civitai base models" column that lets the user add/remove base-model names; saving goes through `ArchitectureService`'s `Command.Update`.

## Symptom

On the Architectures page, expand an architecture card and use the **Add** input in the "Civitai base models" column to add a base model. The tag appears in the UI, but after a page reload (or any re-render, e.g. typing in the architecture search box) it is gone. Removing a base model via its "x" tag *does* persist.

## Root cause

`BaseModelManager.renderColumn` (in `BaseModelManager.scala`, lines ~89–95) — the shared renderer used by `ArchitectureCard.baseModelsColumn` (lines ~234–266) — wires the "x" removal to `onRemove`, but its **Add** button only mutates the local `Var`:

```scala
// BaseModelManager.renderColumn
span(
  ...
  onClick --> (_ => onRemove(bm)),   // removal -> persisted via ArchitectureCard.removeBaseModel
  "x"
)
...
button(
  cls := "button is-small is-info",
  "Add",
  onClick --> (_ => manager.addBaseModel())   // <-- local state only, onSave is never called
)
```

In `ArchitectureCard`:

```scala
BaseModelManager.renderColumn(
  baseModelManager,
  onRemove = bm => removeBaseModel(bm),          // removeBaseModel -> onSave(arch) -> Command.Update (persisted)
  onSave = () => {
    val updated = baseModelManager.snapshot()
    val arch = a.copy(civitaiBaseModels = updated)
    onSave(arch)                                  // <- available, but the Add button doesn't call it
  }
)
```

Note the contrast: the "Add" button inside `ArchitectureForm` (the create/edit form, `ArchitectureForm.scala` lines ~125–144) is *not* affected, because the form persists everything through its `snapshot()` on save, which includes `baseModelManager.snapshot()`. Only the card's column is broken.

## Suggested fix

In `BaseModelManager.renderColumn`, trigger `onSave` after a successful add, e.g.:

```scala
onClick --> { _ =>
  val before = manager.models.now()
  manager.addBaseModel()
  if (manager.models.now() != before) onSave()
}
```

(`onSave` is an idempotent PUT of the full `civitaiBaseModels` list, so calling it unconditionally is also acceptable.)

## Verification

1. Open the Architectures page, pick a card.
2. Add a base model via the card's column.
3. Reload the page / check `~/.config/drift/architectures/<id>.json` — the new entry must be present in `civitaiBaseModels`.
