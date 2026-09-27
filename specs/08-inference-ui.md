# 08 — Inference UI

**Status:** done
**Depends on:** 07

Given a ready session, generate images and videos through the native sdcpp
job API, proxied by drift, and record every generation on disk. The form is
built from what the loaded model reports it supports, not from hardcoded lists.

## What it does

- While a session is live the configuration list swaps to the generation
  panel (`pages/generate/GenerationPanel`), during loading too, with an honest
  loading state; Stop returns to the list. Projects host the same panel in
  their workspace (19).
- The form's controls come from `GET /api/sessions/{id}/capabilities`
  (samplers, schedulers, modes, per-mode features and limits) and its defaults
  from the run configuration: `sd-server` seeds `defaults_by_mode` from its
  launch argv, so capabilities are the effective parameters and no flag
  parsing happens in the browser.
- Image and Video tabs appear only when `supported_modes` has both; init,
  reference, end-image and mask pickers appear per `features_by_mode`. "Edit"
  is not a mode: it is `img_gen` with `ref_images`. txt2img, img2img, edit,
  txt2vid and img2vid all work; video results play inline as webm.
- Submit shows queue position, then status and elapsed time, then the result;
  the only step progress is what the log reports (13). A queued or generating
  job can be cancelled — the panel is the same on the inference page and in a
  project workspace, so the stop is wherever the generation is.
- **The session's queue** sits above the result and lists every active job the
  result does not already show, each with its own ⏹ — a waiting job could not be
  dropped otherwise, though that is the cheapest cancel sd-cpp offers. The list
  is the session's, not the project's: a job queued from elsewhere holds this
  one up just the same, and is nowhere else on screen. One active job that *is*
  the shown one lists nothing, its stop being right below.
- The result follows the **running** generation, falling back to the newest
  when none is: showing the newest submission meant cancelling a queued job
  replaced a live generation on screen with that cancellation, which read as
  the running job having been cancelled too (François, 2026-09-18).
- **⏹ Cancel is always offered** while a job is active. Where the server says
  it can cancel (`cancel_queued` / `cancel_generating` in `features_by_mode`),
  it cancels. Where it says it cannot, the button asks first — killing sd-cpp
  and launching the same configuration again is what stopping costs, and it
  beats killing drift, which was the only other way out. On yes the generation
  is marked cancelled and `SessionManager.restart` takes the server down and
  brings it back on a thread of its own, waiting for the old process so the
  port is free. The queue dies with the server, so every active generation of
  that session is marked cancelled there and then — "dropped when sd-cpp was
  restarted" — rather than surfacing as failed polls a second later; the
  warning says so before anything happens.
- The restart is offered for the job **under way** only: killing the server to
  drop a job that has not started would take the running generation and the
  rest of the queue with it. A queued job the server refuses to drop stays
  queued instead — upstream drops them on request (`cancel_queued: true`,
  `cancel_generating: false` for img_gen), so the interrupt is the real case
  and that one is theoretical.
- `pages/generate/GenerationCancel` holds the rule — the `features_by_mode`
  reading, the warning, the two commands and the button itself — for the result
  and the queue alike; `pages/generate/GenerationQueue` is the list.
- A stop is never instant (a running job ends at sd-cpp's next step boundary, a
  forced one once the server is down), so the button spins and stops taking
  clicks until it lands: `GenerationService.cancelling` holds the ids asked for,
  filled by `push` and emptied by `settle` as the listings come back — in the
  service, not the button, since the panel rebuilds its rows on every status
  tick. A cancel the backend never received clears it, rather than spinning on.
- Random seed: with Random ticked the panel draws a 31-bit seed per
  submission and shows it; the backend draws one for any request that still
  arrives negative, so the sidecar always records the seed actually used.
- The request carries structured `lora` (09), `hires` and `vae_tiling` (10)
  objects; the rest of the form is 14.

## Shape

- `shared/.../Generation.scala`: `ImageGenerationParameters`,
  `VideoGenerationParameters`, `SessionCapabilities`, `Generation(id,
  sessionId, runConfigurationId, kind, status, queuePosition, submittedAt,
  startedAt, completedAt, imageParameters, videoParameters, outputs, error,
  derivation, projectId, promptVersionId, scratch)`, `GenerationOutput(date,
  fileName, url, mimeType, format, index, fps, frameCount)`, `GenerationStatus
  = Queued | Generating | Completed | Failed | Cancelled`. Codecs are
  snake_case: the same bytes travel browser → drift → sd-server.
- Endpoints: `GET /api/sessions/{id}/capabilities`,
  `POST /api/sessions/{id}/generations/image|video` (query `project`,
  `version`, `origin` = `SubmitContext`, and `scratch`),
  `GET /api/sessions/{id}/generations`,
  `POST /api/generations/{id}/cancel[?force=true]` (`force` marks it cancelled
  and restarts the session; `GenerationMonitor.fail` leaves a generation that
  already ended alone, so the polls failing after the kill do not turn it into
  a failure),
  `GET /api/outputs/{date}/{file}`.
- `backend/.../sdserver/`: `GenerationManager` (facade) over
  `GenerationSubmissions`, `GenerationMonitor` (one daemon poll thread per
  native job, so results persist with no browser watching),
  `GenerationRegistry`, `GenerationFiles`, `NativeJobs` (the sdcpp client),
  `ScratchGenerations` (22).
- Outputs: `~/.local/share/drift/outputs/<date>/<id>-<n>.<ext>` with an
  `<id>.json` sidecar (the `Generation`). Input images are externalized to
  `<id>-init.png`, `<id>-ref-<n>.png` etc. and the recorded parameters carry
  their URLs, never base64.
- A refused submission answers with a `Generation` already `Failed`.

## Notes

- **Reference images keep their shape** (`sdserver/ReferenceImages`, 2026-09-22).
  sd-server's `auto_resize_ref_image` decides two resizes: with the default it
  stretches every reference to the output's width × height while decoding the
  request (leejet/stable-diffusion.cpp#2004 — still the default after the fix)
  and then scales it to about a megapixel before the VAE; `false` (honoured
  from master-892, `SdCppBuilds.keepsReferenceSize`) does neither. So on such a
  build drift sends `false` and does the second resize itself, as
  `resize_before_vae` does in `src/pipeline/image.cpp` — a megapixel (the
  output's area for Qwen-Image 2.1), never above the output's area, the shape
  kept, sides on 32 for the Qwen-Image family and 16 for the rest; on an older
  build each reference is letterboxed to the output's shape with grey, so the
  stretch is an even scale. The forwarded body carries the prepared images;
  the recorded parameters keep the ones given. Tested in
  `ReferenceImagesTests`.

- Native API only, never the OpenAI or A1111 compatibility layers: it is
  async by job, which a UI generating 30-second images needs. Ground truth is
  `examples/server/api.md` in the sd-cpp repo at the installed commit; the
  top-level `defaults`/`features` capability fields are deprecated mirrors, so
  only the `*_by_mode` fields are read.
- drift keeps no queue of its own: sd-server queues and reports
  `queue_position`; a second scheduler would fight the first.
- `sd-server` turns a `-1` seed into its default 42, hence drift drawing the
  random seed itself.
- A finished job must never be abandoned on HTTP timeouts: a tiled ESRGAN
  pass starves sd-server's HTTP loop for minutes although the job is
  asynchronous, so the monitor judges timeouts by silence duration (15
  minutes) and only connection-refused fails fast.
- jsoniter traps: `enforce_snake_case` splits `b64Json` into `b_64_json`
  (fixed with `@named`); a field equal to its default is omitted on write, and
  sd-server refills it. tapir cannot auto-derive `Schema[Map[String, Derived]]`
  — every companion carries an explicit `Schema.derived`.
- A Laminar `Observer` val declared after the `effects` Seq that captures it
  is null at bind time.
