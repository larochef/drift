# 22 — Free play and scratch generations

**Status:** done
**Depends on:** 19, 08, 21

Projects are the work; everything else is configuration, and the one thing
configuration needs is to *try the model out*: launch it, poke at it, see
that it answers and how fast, then throw the results away. So the Models page
is free play: nothing it generates is persisted, and one invariant holds —
**a generation is persisted because it belongs to a project**. **Keep**
promotes a result that turns out to matter.

## What it does

- Launching a configuration from the Models page and generating shows each
  result with its duration and a Keep row; nothing lands in the gallery, before
  or after a restart. The panel says "Free play — nothing here is saved".
- **Keep into** "the gallery only" or an existing project: the files move into
  the day's outputs, a sidecar is written, and with a project the recipe
  becomes a version of it exactly as a submission would (19). A kept video is
  a plain gallery entry.
- Scratch outputs are swept when the session stops, at startup, and by
  "Clear results".
- The assistant in free play runs with **no system prompt** (the raw model);
  the workspace keeps the default one plus the project brief.
- Neither free-play panel offers a project, a version or a proposal hand-off.

## Shape

- `Generation.scratch: Boolean`; outputs under `outputs/scratch/`, no
  sidecar, externalized inputs beside them.
- `scratch` rides as a query parameter beside the body on both
  `submitImageGeneration` and `submitVideoGeneration`, not inside
  `SubmitContext` ("do not keep this" is not a project fact, and video takes
  no `SubmitContext`).
- `shared/.../Scratch.scala`: `POST /api/scratch/{generationId}/keep` with
  `KeepRequest(projectId)` answering the promoted `Generation`;
  `DELETE /api/scratch` answering the removed ids.
- Backend `sdserver/ScratchGenerations`; frontend `GenerationPanel` derives
  `scratch = project.isEmpty` rather than taking a flag, so the invariant is
  structural.
- `HistoryService.fold` and `ProjectService.fold` skip scratch generations:
  they fold completions off the status socket and would otherwise invent a
  day called "scratch".

## Notes

- Two facts make scratch nearly free: `GenerationHistory.days` counts only
  `\d{4}-\d{2}-\d{2}` directories, and `getOutputFile` only containment-checks,
  so `/api/outputs/scratch/<file>` serves through the existing route.
- The stop sweep is gated on `SdCpp` sessions: stopping the assistant beside
  a generation session must not throw away the images. The sweep spares a
  running job's files and record.
- The assistant system prompt is one shared `Var` set on mount by each panel
  (empty in free play, the default in a workspace).
- Free play as an opt-in button beside Launch would leave the gallery filling
  with smoke tests; making the configuration pages non-persisting removes the
  source.
- jsoniter omits `scratch` when false; readers treat "missing" as false.
- `DELETE /api/scratch` answers `List[String]`, not `Int`: a top-level
  `given JsonValueCodec[Int]` in `drift.shared` makes every `jsonBody[Int]`
  ambiguous.

## Post-v1

- Create a project from the Keep picker; keep a whole batch in one click;
  a rolling timing strip per configuration; free play remembering its last
  configuration.
