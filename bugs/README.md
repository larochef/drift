# drift — bug tracker

Each file in this directory documents **one bug** in this codebase, written to be self-contained: a fixer should be able to work from a single file without any other context. Every file contains: status, severity, symptom, root cause (with file/line references and code excerpts), suggested fix, and verification steps.

## Project context (shared by all bugs)

- **drift** manages Stable Diffusion model configurations and runs inference via sd-cpp.
  See [`../specs/`](../specs) for what the app is meant to become; `00-overview.md`
  there carries the domain vocabulary these bug reports use.
- **Layout**: `shared/` (Scala 3 case classes + tapir endpoint definitions, cross-compiled JVM/JS), `backend/` (tapir + Netty JVM server), `frontend/` (Laminar/Scala.js SPA), `backend/resources/reference/` (seed data).
- **Persistence**: no database. Each entity is one JSON file at `~/.config/drift/<entityType>/<id>.json` (`architectures`, `models`, `running`), read/written by `backend/src/drift/backend/storage/StorageService.scala`. Entities are defined in `shared/src/drift/shared/Api.scala`; generic CRUD endpoints are generated in `backend/src/drift/backend/routes/RouteHelpers.scala` from the endpoint values in `Api.scala`.
- **Frontend pattern**: services (`frontend/src/drift/frontend/services/*`) own an `EventBus` of commands and a `Var` of state, exposing `effects` (bindings to apply on mount) and `events` (outcomes to observe). Pages/forms are classes holding `Var` state with a `snapshot()` method read on save. Laminar is used; note that `defaultValue :=` on inputs is a one-shot uncontrolled attribute (this is the root cause of Bug 4).
- **Build**: Mill (`build.mill`); frontend is compiled to JS and served by the backend on port 4321.

## Interdependencies (fix order hints)

- **Bug 1 ↔ Bug 7**: both are in the inference edit/create form. Bug 7 (select shows no selection) is invisible while Bug 1 (wrong instance saved) exists; fix 1 first, then 7.
- **Bug 3 ↔ Bug 10 ↔ Bug 14**: the slashed-id 500 (3) is a classic trigger of the dead pipeline (14); the backend hardening suggested in 10 (create parent dir / atomic write) also mitigates 3.
- **Bug 4** is independent of the others but touches the same forms as 1 and 7 — re-check those flows after changing input bindings.
- **Bug 5 and 6** are both "a field is dropped on save" in the architecture edit path; a single pass over `snapshot()`/`handleSave` can fix both.
- **Bug 8, 12, 13** are isolated; any order.
- **Bug 16** is fixed (2026-09-11, with `specs/24`): both halves landed — the
  Civitai endpoints are under `/api`, and unmatched `/api/*` paths answer 404
  instead of the SPA shell.
- **Bug 15** was the actual cause of "saving doesn't work" and gates any
  manual test of the model list, including the checks for Bug 9.
- **Bug 11** is backend-only and latent; do it with a quick API-level check.

## Dead code

All entries previously listed here have been resolved: `GenericRowEditor`,
`SearchBrowser` and `EntityService` were deleted, `CivitaiService` is now what
`CivitaiBrowser` talks to, and `InferenceForm.valid` was a non-reactive `val` that
is now a `def`.

Remaining known dead code lives in `ArchitectureService` — a complete, unused model
pipeline (four commands, four events, a second `_allModels`). It is tracked as a work
item in [`../specs/04-run-configurations.md`](../specs/04-run-configurations.md)
rather than here, because removing it belongs with that feature's refactor.

## Conventions

- One file per bug, numbered in priority order; file name = `NN-short-slug.md`.
- When a bug is fixed: set **Status** in the file to `fixed`.
- Line numbers in the files are from the tree as of the date of writing; use the quoted code snippets as anchors, not the exact line numbers.
