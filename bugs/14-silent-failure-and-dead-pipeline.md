# Bug 14 — Failed Create/Update: no user feedback, and the command pipeline dies until page remount

**Status:** fixed
**Severity:** medium (every failed save — e.g. the 500 from Bug 3 — leaves the form "dead")
**Files:** `frontend/src/drift/frontend/services/ModelService.scala`, `frontend/src/drift/frontend/services/InferenceService.scala`, `frontend/src/drift/frontend/services/ArchitectureService.scala`, `shared/src/drift/shared/Api.scala`

## Context

All service classes follow the same pattern: an `EventBus` of commands, mapped through `interpreter.toClientThrowErrors(...)` (tapir sttp client) into an `EventStream`, with a `--> Observer/Var` sink:

```scala
// ModelService.scala (representative; same shape in InferenceService / ArchitectureService)
private def createModel(m: Model): EventStream[Model] = toStream(createModelFn(m))
...
cmdBus.events
  .collect { case Command.Create(m) => m }
  .flatMapSwitch(createModel)          // toClientThrowErrors => the promise REJECTS on non-2xx
  --> Observer[Model] { m => ... }     // no error handler anywhere
```

## Symptom

Any non-2xx response (e.g. the 500 from Bug 3) rejects the promise; `EventStream.fromJsPromise` emits an error; with no error handling, Laminar ends the source stream and the Create pipeline for that page stops reacting to further commands until the page element remounts (navigate away and back). The user sees nothing: no error message, the button just stops working.

Related (different but same family): a **successful** Update that returns `None` (entity not found) is silently swallowed:

```scala
// InferenceService.scala (~lines 116–123) / ModelService.scala (~lines 75–82)
updateRunning(id, rm).collect { case Some(_) => (id, rm) }   // None => stream ends, no event, form stays open
```

## Suggested fix

- Wrap the request streams so failures become values instead of ending the stream, and surface them: e.g. `.recoverToTry.map(t => t.toOption)` and branch in the sink — on `None`/failure log to console AND notify the UI (a simple toast/inline error, or a `CommandBus`-driven `Var[Option[String]]` error banner shown on the pages).
- For Update returning `None`, emit a distinct event (e.g. `Event.UpdateFailed(id, reason)`) so the page can close the edit form and show a message, instead of hanging silently.
- Keep `toClientThrowErrors` (it's fine), just make sure every stream that can fail has a terminal error strategy.

## Verification

1. Stop the backend.
2. Frontend: try to create a model / run configuration → a visible error must appear.
3. Restart the backend; the same form must work without navigating away.
4. Delete an entity directly from `~/.config/drift/`, then Update it from the UI → a "not found" style feedback instead of a frozen form.
