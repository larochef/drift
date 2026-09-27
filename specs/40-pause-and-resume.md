# 40 — Pausing a tiled job

**Status:** done in code 2026-09-22, not yet run live
**Depends on:** 15, 26, 27, 39

A tiled job is long: 169 PiD tiles is two hours, a redraw of an 8192² image not
much less. Two hours is too long to be unable to recompile drift, and too long
to hold the GPU when something else needs it. So a tiled job can be paused
between two tiles and picked up later — after a drift restart included.

## What it does

- A running PiD, redraw or edit job has a **Pause**. It stops after the tile in
  flight finishes — no work is thrown away — keeps every finished tile on disk,
  stops the job's sd-server so the VRAM is free, and the job becomes
  **Paused**. The tile in flight is the only wait: minutes at most.
- **The ask shows at once.** A paused-when-this-tile-ends job says so on its
  card (`PostProcessJob.pauseRequested`), and its Pause becomes **Force pause**:
  confirmed, that drops the tile in flight the way a cancel does — the img_gen
  job cancelled on its server, an `sd-cli` killed — and the job pauses there,
  that tile to be run again on resume. Without the notice the click looked
  lost, since nothing changes until the tile ends (François, 2026-09-22).
- A **Stop** after a pause still cancels: the later word wins, and it keeps
  nothing.
- A paused job has **Resume**: a server starts again and the job carries on at
  the first tile it does not already have, with the same seed, layout and
  settings, and ends with the blend as if it had never stopped.
- **A paused job survives a drift restart.** What it takes to finish is stored
  beside the other entities: the kind, the source, the request as it was
  resolved (its drawn seed included) and how many tiles it has. On start drift
  lists paused jobs again, and their tiles are where they were left.
- **Cancel** on a paused job throws its tiles away and forgets it, as cancelling
  a running one does.
- Resume refuses, loudly, when the job can no longer be what it was: its source
  is gone, its run configuration or runtime no longer resolves, or the tiles it
  would lay out now are not the ones it has — a changed tile size or
  architecture. Nothing is deleted; the reason names what changed.
- The job keeps one log across pauses: a resumed job appends to it, starting
  with the line that says it resumed and at which tile.

## Shape

- `shared`: `PostProcessState.Paused` (not active, so nothing treats it as
  running); `PausedWork` — the three requests as one sum (`Pid`, `Redraw`,
  `Edit`) — and `PausedJob(id, kind, sourceDate, sourceFileName, work, tiles,
  startedAt)`, stored under `post-process-jobs`; `pausePostProcessJob` and
  `resumePostProcessJob` endpoints beside `cancelPostProcessJob`.
- `PostProcessJobs`: a job's `JobCancellation` gains `paused`, set by `pause(id)`
  and read between tiles like `stopped`; `pause` writes the `PausedJob` record
  and leaves the tile files alone, where the end of a job removes them;
  `resumed(id)` records a job again under its own id; `cancel` of a paused job
  removes the record and the files; the list drift starts with is the stored
  records, as `Paused` jobs.
- `TiledJobs.startTiles(kind, src, rows, work, resuming)` carries the request to
  store and the id to keep. `runTiles` reads the pause flag where it reads the
  cancel flag, skips a tile whose output file it already has, and — for the
  jobs that run in order (39, and a redraw with context) — paints those files
  into the picture first, so a resumed tile is cut from the same picture the
  first run would have cut it from.
- `PidUpscale`, `Redraw` and `Edit` each take the request they were given and
  hand it on as `PausedWork`, with the seed they drew filled in; their `start`
  takes the id to resume, if any. `PostProcessManager.pause` / `resume` are what
  the routes call, `resume` rebuilding the job from its record.
- Frontend: the job card gets **Pause** while running and **Resume** ∕
  **Cancel** while paused (`PostProcessJobList`), through
  `PostProcessService.Command.Pause` / `Resume`; a paused job shows the tiles it
  has of the tiles it needs.

## Notes

- Pausing between tiles rather than killing the server is what keeps the work:
  the tile in flight would otherwise be lost, and on a PiD tile that is up to
  three minutes. François asked for both and settled on the tile boundary.
- The model is not held while paused: freeing the VRAM is half the point of
  pausing (the other half is recompiling drift).
- The tiles of a paused job stay where a running job's tiles are — beside the
  job log — and are removed when it ends or is cancelled, so a paused job that
  is never resumed costs disk until it is cancelled. The job list makes that
  visible rather than a cleanup rule making it disappear.
