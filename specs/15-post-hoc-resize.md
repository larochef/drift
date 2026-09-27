# 15 — Post-hoc upscale: ESRGAN and PiD

**Status:** done
**Depends on:** 12, 10 (the upscaler store)

Take an image already in the gallery and produce a bigger copy **next to it**,
keeping both, with no live session required. Two operations: **ESRGAN** through
`sd-cli --mode upscale`, and **PiD** (NVIDIA's Pixel Diffusion Decoder), a
small diffusion model that re-renders the image at ×4 conditioned on the
source and a prompt — sharper than ESRGAN on generated images. PiD's tiling
is 26; the redraw pass that usually follows is 27.

## What it does

- The gallery detail view has a post-processing section (`PostProcessSection`)
  holding three tasks — redraw (27), PiD upscale (26) and the ESRGAN upscaler
  — one on screen at a time behind a picker, with the live sessions and the
  jobs shared underneath. Every panel is built once and hidden rather than
  removed, so what is typed survives a status tick, a change of output and a
  change of task; each opens with a line saying what it does and closes with
  the job's summary and its button, and nothing of a task sits after its own
  button. A panel shows the **model** and nothing else: every other parameter
  is behind an "Advanced" fold, so a panel can be read at a glance
  (François, 2026-09-21). The fold is `components/FoldedSection` — the
  architecture page's LoRA fold (09), which the panels asked for by name rather
  than a button of their own, so it moved into a component both use — with its
  body hidden rather than removed, since a form loses what has been typed into
  it otherwise. Small controls, one labelled group of fields a row,
  every field named beside its own control, and the space between the rows on
  the rows themselves — a dozen small numbers joined into one strip were what
  made the panel unreadable before it was grouped.
- `UpscalePanel`: the upscaler from the store and how many passes it runs.
- Each job runs on a daemon thread, is listed under the source with a
  progress bar (PiD: tile n of m), and appears on the `postProcessJobs` topic
  of the status socket. A failure attaches the log tail and leaves no
  partial output in the gallery.
- **One job per image at a time** (`PostProcessJobs.busyWith`): a second one on
  an output that already has a running or paused job is refused by name — stop
  or cancel that one first. So the tiles drawn over a picture are one job's,
  and nothing has to choose between two.
- **A running job's tiles are drawn over its picture**, each in the state it is
  in: done, running, still to come (`PostProcessJob.tiles` in the source's own
  pixels, in the order they run, with `progress.completed` saying where the
  model is). **A tile that is done shows what the model made of it** —
  `GET /api/post-process-jobs/{id}/tiles/{index}?side=512`
  (`PostProcessJobs.tilePreview`, the file scaled for the screen, 404 until
  that tile exists) — so the result appears piece by piece over the original
  as the job goes. What lies on disk is the *finished* tile: a job that
  finishes its tiles (an edit's composite, 39) writes that, which is also what
  a resume repaints. A job's grid replaces the
  panel's while it is on that image, and a paused job keeps showing what it
  did. `TileState` and the `is-done` / `is-running` / `is-waiting` classes.
- **What is left, in words**: after each tile a job records what a tile has
  cost *this run* (`secondsPerTile`, the mean over the tiles done since it
  started or resumed), and the card says "about 1 h 20 left". Nothing is shown
  until the first tile has finished — an estimate before any measurement is a
  guess with a number on it.
- The result is a new gallery entry beside the original, with a "Made from"
  link, a Result ↔ Original toggle at matched zoom, and "Made from this" on
  the parent. Cards carry a badge per operation. A derived entry offers no
  reuse; its original does.
- While a generation session is live the section says so and offers to stop
  it to free memory; starting a job never stops anything.
- A running job shows two bars: how far the job is (tiles done), and what the
  model is doing inside the current one — "loading weights 15/298", then
  "sampling 3/20 (15%) · 1.42s/it" — with the log's last line when no bar is in
  flight. It is the inference panel's view, reading the same sd-cpp output
  through the same parser.
- A running job has a **Stop**: the `sd-cli` it waits on is killed, a tile
  waiting on a server has its img_gen job cancelled there, and a job between
  two tiles stops at the next one. What it had written is removed and the row
  reads "cancelled — nothing was kept".
- A finished card — done, failed or cancelled — closes with a **×**. Only the
  notice goes: the job stays in the list the backend holds and the result stays
  in the gallery. It is what keeps a second redraw of the same output from
  being read as the first one's result (François, 2026-09-20). A running card
  has none — it carries the only Stop. The closed set is the browser's, not the
  server's: a reload shows the cards again.
- Video sources are refused.

## Shape

- `shared/.../PostProcess.scala`: `UpscaleRequest(upscalerId, repeats,
  tileSize, runtimeId)`; `PidUpscaleRequest(runConfigurationId, width,
  height, prompt, negativePrompt, steps = 4, cfgScale = 1.0, seed = -1,
  runtimeId)` with `PidUpscaleRequest.target` (both sides or neither: ×4 of
  the source's real pixel size, ratio exact, longest side ≤ 16384; explicit
  sides are multiples of 4 from 256 to 16384, and a target no larger than the
  source is refused — it would not be an upscale); `PostProcessJob(id, kind,
  source…, state, progress: PostProcessProgress(completed, total),
  logProgress: Option[SessionProgress], activity, outputTail, result)`;
  `PostProcessState` is Running → Completed | Failed | Cancelled.
- A tiled job can be paused and resumed, a drift restart in between
  (`specs/40-pause-and-resume.md`).
- Endpoints: `POST /api/outputs/{date}/{file}/upscale`, `…/pid`,
  `…/redraw` (27), `POST /api/post-process-jobs/{id}/cancel` (false when no
  running job has that id), `GET /api/post-process-jobs`.
- `backend/.../postprocess/`: `PostProcessManager` (facade) over
  `PostProcessJobs` (records, threads, logs at
  `~/.cache/drift/logs/postprocess-<id>.log`), `EsrganUpscale`, `PidUpscale`,
  `Redraw`, `TiledJobs`, `TileBlending`, `PostProcessImages`; the tile and
  window geometry is `shared/.../Tiling.scala` (27).
- Progress inside a tile: `PostProcessJobs` keeps a `SessionLog` (13) per
  running job — the same ring buffer a session keeps, so the bar redraws
  collapse and the parsing is not written twice — and mirrors its `progress`
  and `activity` onto the job, which the status socket carries at its own 500 ms
  cadence however fast sd-cpp redraws. Lines reach it three ways: `spawn`
  drains the `sd-cli` pipe through `ProcessOutput.capture` (the log file is
  still written verbatim, and an unread pipe would block the child anyway), a
  job server hands its lines to the `onLine` `JobServers.start` now takes, and
  tiles sent to a **ready session** are followed instead — `TiledJobs` registers
  a source with `PostProcessJobs.follows` and `list` reads that session's own
  progress, since the output there belongs to the session, not to the job.
- Cancelling: `PostProcessJobs` keeps a `JobCancellation` per running job —
  a flag, the live `sd-cli` process, and the `(port, native job id)` a tile is
  waiting on, the last two set while they run and cleared after. `cancel`
  raises the flag, kills the process and calls `NativeJobs.cancel`, then
  records the state at once so the answer is immediate. The job's own thread
  unwinds through `fail`, which sees the flag and records a cancel rather than
  a failure, having already deleted the partial output. `NativeJobs.cancel`
  is the one place that knows `POST /sdcpp/v1/jobs/{id}/cancel`;
  `GenerationManager.cancel` calls it too.
- The derived entry is a full `Generation` sidecar in the parent's day, kind
  `upscale`/`pid`/`redraw`, no session, no request, `derivation:
  Derivation(parentId, parentDate, parentFileName, operation, upscalerId,
  repeats, width, height, configurationId, prompt, steps, seed, strength,
  instructions)`. Entries with operation `resize` from before that operation
  was removed still display.
- `Architecture.pixelDiffusionDecoder` flags a PiD architecture; only its
  configurations appear in the PiD picker and the endpoint refuses others.
  Built-ins: `pid-flux2`, `pid-flux2-2048`, `pid-flux1`, `pid-qwen-image`
  (26 lists the pairings). Their defaults: `-W/-H`, 4 steps, cfg 1,
  `--rng cpu`, no `--diffusion-fa` (26 says why: above 2048 it returns a black
  frame, and at tile scale it costs fidelity to the source).
- ESRGAN spawns `sd-cli --mode upscale --upscale-model … --init-img …
  --output … --upscale-repeats n [--upscale-tile-size t]` on the default (or
  the request's) runtime with the launch environment; sd-cli sits beside
  sd-server. PiD resolves the configuration as a launch would (blockers
  refused by name) and runs it as a job server, one img_gen job per tile with
  the tile's crop as its reference (26).
- Setting PiD up: `Comfy-Org/PixelDiT` `pid_1.5_<variant>_1024_to_4096_4step_bf16`
  (the diffusion slot), `gemma_2_2b_it_elm_bf16` (text encoder), and the VAE
  family of the variant. Weights are under NVIDIA's non-commercial licence.

## Notes

- `sd-cli`'s exit code is worthless: with an unloadable model it logs
  `[ERROR]`, saves the input unchanged and exits 0. Success is judged on no
  `[ERROR` line in the log **and** the output being larger than the input;
  a PiD tile that is entirely black also fails the job.
- `RealESRGAN_x2plus` does not load in sd-cpp (its 12-channel `conv_first`);
  x4plus works.
- Use the `_4step_` PiD files: drift samples 4 LCM steps, and the undistilled
  decoders give a green-cast grid at that count. Use PiD 1.5 whatever the
  target: the 1.0 decoder colour-casts on small tiles.
- The decoders are strictly ×4: the reference must be exactly a quarter of
  the target, so PiD scales the source (fill semantics, crops another ratio)
  and pads it to multiples of 16 by repeating the last row and column, then
  crops the blend back.
- Flash attention spoils PiD (speckle and colour shift on ROCm, black on
  CPU), and without it one ROCm pass is capped at 1792² (2048² aborts in
  `ggml_cuda_op_scale`; Vulkan asserts at 2048² either way). Hence 26.
- Fused hires (10) cannot give the original back; post-hoc is where the
  original ↔ result compare lives.

## Post-v1

- A slider compare; thumbnails for 4096² results (25 MB PNGs in the grid).
