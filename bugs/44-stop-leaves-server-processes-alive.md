# Bug 44 — Stop does not always end the server's process: the model stays in memory

**Status:** open — cause read in the code 2026-10-05 for the case François hit (a redraw on FLUX.2 dev on the
drift runner, stopped with the redraw task's Stop); not fixed, not reproduced; why the runner ignores a SIGTERM
sent by hand is still not known
**Severity:** high (the memory of the old model stays taken, and probably its job keeps the GPU; the next
model is loaded beside it — the oversubscription `MemoryHeadroom` warns about — and only a `kill -9` from a
terminal frees it)
**Files:** `backend/src/drift/backend/session/ServerProcesses.scala` (`terminate`),
`backend/src/drift/backend/session/SessionManager.scala` (stop, `GraceSeconds`),
`backend/src/drift/backend/session/JobServers.scala`,
`backend/src/drift/backend/postprocess/PostProcessJobs.scala` (a cancelled job's process); on the runner's
side, what it does on SIGTERM (`runner/src/drift/runner/server`, `native/HipRuntime.scala`)

## Symptom

The stop buttons do not always kill the process they stand for. The model's memory stays taken, and the job
that was running probably keeps running. A `kill` (SIGTERM) by hand did not end the process either; `kill -9`
did.

## What the code does today

`ServerProcesses.terminate`: SIGTERM, wait `SessionManager.GraceSeconds`, then `destroyForcibly` (SIGKILL) and a
5 s wait; it logs "ignored SIGTERM for …s; killing" when it gets there. `SessionManager` stops its sessions the
same way. A cancelled post-process job kills its process forcibly at once. The runner's `sd-server` is a shell
script that `exec`s the JVM, so the pid drift holds is the JVM's.

So a process that ignores SIGTERM should be killed after the grace period — and one was still there.

## Found: a stopped redraw waits for its tile, and the runner cannot stop a tile

The backend's log of the case:

```
23:26:43.194 JobServers  Job server for 'flux-2-dev-redraw': pid 56971 on port 7861
23:30:16.968 NativeJobs  sd-server answered 409 to cancelling job 80c70ca0-…
23:30:16.969 PostProcessJobs  Redraw g1791235603184-1002 cancelled
```

1. **The runner only cancels a job that has not started.** `ImageServer.job`: `POST …/jobs/{id}/cancel` takes a
   queued job out of the queue; for one that is generating it answers 409 ("the job is generating") and the
   generation runs to its end. Nothing in the pipelines looks at a stop between two steps.
2. **drift records the redraw as cancelled at once** (`PostProcessJobs.cancel`), so the page says stopped.
3. **The job's server is not stopped by the cancel.** `cancel` stops `cancellation.server`, but only a SeedVR2
   video registers one (`jobs.runsOn` in `SeedVr2Upscale`). A tiled job's server is stopped by the tile loop
   itself (`TileRun`, `stopServer()`), and the loop reads the cancel flag *between two tiles*: it is waiting for
   the tile in flight, which the runner refused to drop. On FLUX.2 dev that is minutes of a model held in
   memory and a GPU kept busy by a tile nobody wants — and a next job started meanwhile loads beside it.
4. Only when that tile ends does `ServerProcesses.terminate` run (SIGTERM, the grace period, SIGKILL).

sd-cpp answers the same 409 past the point its build can cancel (`NativeJobs.cancel`'s comment: "the job
simply finishes"), so the wait in 3 is not the runner's alone.

Still unexplained: the SIGTERM sent by hand did not end the runner either. See the first lead below — and if
that is what happens, step 4's own SIGTERM is ignored too and the kill comes only after the grace period.

## To do

- **A cancel stops a tiled job's server at once**: `TileRun` registers its server with `jobs.runsOn`, as the
  SeedVR2 video does, when the server is the job's own (not a session's it borrowed). The tile's wait then ends
  on a dead server and must be read as the cancel it is, not as a failure.
- **The runner cancels a running job**: a flag the samplers read between two steps (and the tiled VAE between
  two tiles), the job ending `cancelled`, 200 to the cancel. A pause's forced stop and a session's Stop get it
  too, and the server stays loaded for the next job where that is wanted.
- **The runner ends on SIGTERM** whatever its GPU thread is doing (below).

## Leads

- **The process ignores SIGTERM.** Confirmed by hand. A JVM on SIGTERM runs its shutdown hooks and waits for
  them: a hook, or an ox scope closing, that waits on a thread blocked in a native GPU call (the blocking wait
  of bug 38, a kernel that never returns) never finishes. sd-cpp and llama.cpp have their own handlers and may
  wait for the step in progress.
- **The SIGKILL that should follow is not sent, or not to that process.** Which stop paths go through
  `terminate` and which only ask the server to stop over HTTP or call `destroy()` alone; a stop that returns to
  the browser before the grace period ends and is then forgotten; an entry dropped from the manager's map
  while its process lives (a server the manager no longer knows cannot be stopped from the page).
- **A process drift does not hold.** A server started for a tiled job (`JobServers`) or kept loaded for the
  Sandbox against the session shown on the page: the button stops one and the other holds the memory.
- **Children.** A server that starts a process of its own (ffmpeg, a helper) is not reached by a kill of its
  parent: `Process.descendants`.

## To record when it happens again

`pgrep -af '[s]d-server|[l]lama-server|[d]rift.runner'` before and after the click, which page and button,
whether a job was running, and the backend's log around the stop (the "ignored SIGTERM" line, or its absence,
tells the first two leads apart). For a runner that ignores SIGTERM: `jstack <pid>` before the `kill -9` shows
what its shutdown is waiting on.

## Wanted

- Stop means the process is gone and its memory free when the button says stopped — or the page says it is
  not, by name, with a way to force it.
- Every way of stopping (a session, a job's server, the Sandbox's kept model, a restart, drift's own exit)
  ends in the same SIGTERM → grace → SIGKILL, descendants included, and checks the pid is gone.
- The runner ends promptly on SIGTERM whatever its GPU thread is doing.
