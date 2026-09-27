# Bug 15 — Architectures page never mounts `ModelService.effects`: model saves never reach the network

**Status:** fixed
**Severity:** critical (the reported "saving data doesn't work"; silent, total loss of every model create/delete)
**Files:** `frontend/src/drift/frontend/pages/architectures/ArchitecturesPage.scala`, `frontend/src/drift/frontend/services/ModelService.scala`

## Context

drift is a Scala 3 monorepo: a Laminar (Scala.js) SPA in `frontend/`, a tapir + Netty backend in `backend/`, shared models in `shared/`. Data is persisted as JSON files at `~/.config/drift/<entity>/<id>.json`.

Every service follows the same pattern: an `EventBus` of commands, a `Var` of state, and an `effects: Modifier[HtmlElement]` value holding the `cmdBus.events … --> Observer` subscriptions. `effects` is inert until a page applies it to an element it mounts. Airstream's `EventBus` does not buffer: an event pushed while nothing is subscribed is dropped, not replayed.

## Symptom

On the Architectures page, expand a checkpoint and use **Add Model**: the form closes as if the model was created, but nothing is persisted, no request is made, and the model list stays empty forever ("No models defined yet.", model count `0` on every checkpoint row). Deleting a model does nothing either. Restarting the app changes nothing.

## Root cause

`ArchitecturesPage.element` mounted only `ArchitectureService`'s effects:

```scala
lazy val element: HtmlElement = div(
  cls := "content",
  service.effects,        // ArchitectureService only
  ...
```

`modelService.effects` was mounted **nowhere in the codebase** (`grep -rn "effects" frontend/src` confirms). So `ModelService`'s four command subscriptions were never active, and all three call sites pushed into a bus nobody was listening to:

```scala
private def handleAddModel(m: Model): Unit =
  modelService.push(ModelService.Command.Create(m))    // dropped
private def handleDeleteModel(modelId: String): Unit =
  modelService.push(ModelService.Command.Delete(modelId))  // dropped
onMountCallback { _ =>
  service.push(Command.Load)
  modelService.push(ModelService.Command.Load)         // dropped
}
```

`ArchitectureCard.handleAdd` closes the add form on its own local state, which is why the UI looked like it had worked.

Confusingly, `ArchitectureService` *does* carry a fully wired model pipeline (`Command.CreateModel` / `UpdateModel` / `DeleteModel` / `LoadModels`, mounted via `service.effects`) — the page just pushes to the other service. That pipeline is currently unused.

## Fix

Mount it, next to the service that was already mounted:

```scala
  service.effects,
  // Without this, ModelService's command subscriptions are never active and
  // every `modelService.push(...)` below is silently dropped by its EventBus.
  modelService.effects,
```

(The alternative — routing the page's model commands through `service` and dropping the second `ModelService` — would also work, but the shared `ModelService` is what the Inference page needs too; see Bug 9.)

## Verification

Measured before and after against a running `./mill backend.run`, loading `/architectures` in headless Chromium and reading the server's request log plus the rendered DOM:

| | `GET /api/models` issued | model count on the `wan-2.1-vae` checkpoint row |
|---|---|---|
| before | no | `0` |
| after | yes | `1` |

Manually: Architectures → expand a checkpoint → **Add Model** → fill id and label → **Add**. A `models/<id>.json` file must appear under `~/.config/drift/`, and the model must still be listed after a reload.
