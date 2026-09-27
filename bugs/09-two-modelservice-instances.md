# Bug 9 — Two independent ModelService instances: model list stale across pages

**Status:** fixed
**Severity:** medium (cross-page inconsistency; the Inference page can't see newly added models)
**Files:** `frontend/src/drift/frontend/Main.scala`, `frontend/src/drift/frontend/services/InferenceService.scala`, `frontend/src/drift/frontend/services/ModelService.scala`

## Context

`ModelService` fetches `GET /api/models` into a private `Var` (`_allModels`) and exposes it as `allModels`. Pages bind `service.effects` on mount and push `Command.Load` in an `onMountCallback`.

## Symptom

Add (or delete) a model on the Architectures page, then navigate to the Inference page: the checkpoint-assignment dropdowns still show the old model list (e.g. "No models defined yet") until you navigate away and back, because each page uses its own `ModelService` instance with its own `Var`.

## Root cause

```scala
// Main.scala (~lines 114–117)
val inferenceService = InferenceService()        // constructs its OWN ModelService internally
val architectureService = ArchitectureService()
val modelService = ModelService()                // separate instance, used by ArchitecturesPage
```

```scala
// InferenceService.scala (~line 29)
class InferenceService(val modelService: ModelService = new ModelService())
```

Two instances, two `Var`s, no shared state. Each re-fetches on its own mount, so they only agree by coincidence of timing.

## Suggested fix

Create a single shared `ModelService` in `Main` and inject it:

```scala
val modelService = ModelService()
val inferenceService = InferenceService(modelService)
// ArchitecturesPage already receives modelService
```

The `InferenceService(modelService = ...)` constructor parameter already exists for this purpose. Binding `service.effects` from both pages is fine because only one route is mounted at a time (Laminar unbinds modifiers when the element is removed).

## Verification

1. Architectures page → add a model in a family.
2. Navigate to Inference → new configuration → pick the architecture with that family.
3. The new model must be selectable immediately, without a page reload.
