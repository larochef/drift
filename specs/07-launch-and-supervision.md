# 07 — Launching and supervising `sd-server`

**Status:** done
**Depends on:** 04, 05, 06

A session turns a run configuration into a live `sd-server` (or `llama-server`,
see 18) process on a known port, and drift owns that process for its whole
life: spawn, readiness, crash detection, stop, and cleanup when drift exits.

## What it does

- Launch on a run configuration card spawns the process with the argv the
  card's command-line preview shows, on the runtime picked in the select beside
  the button (the default runtime preselected). The choice lives on the
  session, not the configuration.
- A launch that cannot proceed — weights not cached, a required slot empty,
  the concurrency cap reached — answers with a session already `Failed` and
  the reason in `error`, the same way refused downloads and generations report.
- One session per run configuration; launching a running one focuses it. At
  most one generation session and one assistant session at a time by default.
- `Starting` becomes `Ready` when the server answers `/sdcpp/v1/capabilities`
  (`/health` for llama.cpp), never from log parsing; a process that exits early
  becomes `Failed` with its exit code and the tail of the log.
- Stop sends `SIGTERM`, then `SIGKILL` after a grace period. Killing drift
  stops every child first.
- The card shows status, progress bar, activity line and parameter notes,
  all riding the `sessions` topic of the status socket (11).

## Shape

- `shared/.../Sessions.scala`: `Session(id, runConfigurationId, tool,
  runtimeId, port, pid, status, startedAt, error, progress, activity,
  parameterNotes)`, `SessionStatus = Starting | Ready | Failed | Stopped`,
  `LaunchSessionRequest(runConfigurationId, runtimeId)`.
- Endpoints: `GET /api/sessions`, `POST /api/sessions`,
  `DELETE /api/sessions/{id}`.
- `backend/.../session/`: `SessionManager` (facade), `LaunchArguments`
  (argv through `CommandLine.resolve`, see 16), `ServerProcesses` (spawn,
  monitor fork, reaping), `SessionOutput` + `SessionLog` (the log reader
  and ring buffer, see 13), `JobServers` (a server started for one
  post-process job, not a session, see 26), `SessionSettings`.
- Settings file `~/.config/drift/settings/sessions.json`:
  `maximumConcurrentSessions` 1, `maximumConcurrentAssistantSessions` 1,
  `portRangeStart` 7860, `portRangeEnd` 7899, `readinessTimeoutMinutes` 15,
  `memoryHeadroomWarningPercent` 10. Read on every launch; no UI.
- Memory warning: while a session is ready, its monitor has `MemoryHeadroom` read
  `/proc/meminfo` (free pages plus the file cache nothing maps — mapped
  weight files are not memory left). Under the threshold, every ready
  session carries `memoryWarning` naming the loaded configurations; the
  card and the generating panel show it (`MemoryWarningView`). Kept
  current every second, so it comes and goes with the memory. A warning only: nothing is refused or stopped.
- Machine status: `MachineMonitor` reads the GPU's load and held memory (amdgpu's
  `gpu_busy_percent`, `mem_info_vram_used` + `mem_info_gtt_used`; absent with another
  driver) and `MemoryHeadroom` every 2 s on a watch of its own, page open or not, with how
  long the GPU has stayed under 10 % in steps of 5 s, and keeps the last 5 minutes
  (`MachineSample`). It crosses the status socket as the `machine` topic (`MachineStatus`,
  history included). Shown in a box at the bottom of the sidebar on every page
  (`MachinePanel`): two graphs of those 5 minutes (`MachineGraph`), the GPU's load and the
  memory in use, the curve green in the lower third, yellow in the middle, red in the upper,
  dashed lines at the thirds, a line each minute, the value and its age under the pointer;
  then the memory on the GPU. And in one line under every log bar
  (`MachineLine`, inside `LogProgressView`: a loading session, a generation, a
  post-processing job). From 15 s idle the line says "GPU idle for N s"; idle or under the
  memory threshold it takes the warning colour.
- Logs: `~/.cache/drift/logs/<sessionId>.log`.
- Sessions are runtime state held in memory, never persisted; a drift restart
  forgets them.

## Notes

- The runtime supplies a binary **and** an environment: `LD_LIBRARY_PATH`
  must hold `<therock>/lib` and the install's `build/bin` for ROCm builds,
  the install root for Vulkan, or the process dies at once with
  `error while loading shared libraries: libstable-diffusion.so`.
- Argv assembly is shared code, so the preview and the launcher are the same
  function; the preview resolves cached weights to the local paths the launcher
  passes, and falls back to source references for weights not on disk.
- Ports promised to still-starting sessions are excluded from allocation: the
  server has not bound them yet, so a bind test alone hands a port out twice.
- The process's stdout and stderr are drained by a reader fork that mirrors
  them to the log file (13); leaving the pipe unread stalls sd-server.
- A running session keeps the argv it started with when its configuration is
  edited; silently restarting would throw away a warm multi-gigabyte load.
- `app.css` pins the neutral tag colour on `.tag` only, not the coloured
  variants; the plain pin out-cascades Bulma v1 and flattens every status chip.
