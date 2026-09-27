# 12 — Gallery and history

**Status:** done
**Depends on:** 08, 11

Everything drift ever made, browsable after the fact, with "reuse these
parameters". The sidecars 08 writes beside each output are the durable record;
the gallery rebuilds from them, so nothing depends on a session or on drift's
in-memory state.

## What it does

- Sidebar entry **Gallery** (`/gallery`): a grid grouped by day, newest first.
  One tile per output; a batch's tiles sit side by side with an accent along
  their top and a "▦ 2/4" badge.
- Filters: project, NSFW (off by default, hides generations whose project is
  flagged NSFW and drops those projects from the project filter), run
  configuration, kind (images / videos), and a search over prompt text.
- Paging: the newest days load until about two dozen generations are on
  screen; older days show a "Show n generations" button plus "Show all
  remaining days". Filters apply to loaded days only.
- A generation completing while the gallery is open appears by itself, folded
  in from the status socket's `generations` topic.
- The detail view: the media (a `<video>` player for video formats), every
  recorded parameter, the input images, the lineage of derived entries with an
  original ↔ result toggle (15), and the post-processing section (15, 26, 27).
- Reuse: **Full** into a live session of the same run configuration reproduces
  the recorded request field for field, seed included; **Task** anywhere else
  carries prompt, negative prompt, input images, seed, size, frames and the
  sampling fields the form shows over the target's defaults. When nothing is
  live a picker launches any other configuration with the task.
- Delete removes outputs, externalized inputs and the sidecar after a
  confirmation. A **Select** toggle puts the grid in selection mode: cards
  tick instead of opening, each day heading offers "select all", and one
  confirmation deletes the ticked set.

- **The detail view is a height budget, not a stack** (François, 2026-09-19:
  "there shouldn't be any need to scroll in that page"). The header carries
  what is being *looked at* — the title, `n of m` for a batch, Result/Original,
  ⤢ Full size — and the footer what is *done* with it. Between them the image
  takes whatever height is left (`flex: 1 1 0`, no fixed `vh` cap), and the
  batch filmstrip runs **down the left of it**, where it costs no height and
  cannot be scrolled out of sight: not noticing that a generation was a batch
  is how a result gets missed. Its thumbnails are squares cropped from the top
  — a 2048×1024 frame letterboxed into a strip is a sliver between two empty
  bands, and the subject is usually at the top — and their side shrinks with
  the size of the batch (`--strip-count`, against the height the card leaves),
  so eight fit as readably as two. Picture and strip are both centred on the
  card's middle line, so an image shorter than the height available does not
  hang from the top over a band of nothing (François, 2026-09-20).
- **The right column is tabbed**, one section at a time: *Parameters*,
  *Redraw & upscale* (badged while jobs run), *Inputs* — only when the
  generation *has* input images, an empty section being a question with no
  answer — and *History* (the chain, badged with its size). Each wants the
  full height and only one is wanted at once. Every body is built once and
  hidden rather than removed: the redraw panel holds what has been typed into
  it, so `DetailTabs` owns the bodies instead of leaving them to a `child <--`
  that would rebuild them — which is why it is not `BrowserTabs` (24). The
  open tab lives on `GenerationDetailHost` and survives opening the next
  image; a tab that image does not have falls back to the first.
- **Inside *Redraw & upscale*, one task at a time**: redraw, PiD upscale and
  the upscaler behind a picker, built once and hidden the same way (15). Only
  one is ever run, redraw has the fields of a small form and wants the height,
  and the task is also what says whether the selection tool belongs on the
  picture. A task shows its model and folds the rest behind Advanced (15). The open task lives on `GenerationDetailHost` too — a run of
  redraws over a batch would otherwise pick the task again on every image
  (François, 2026-09-20).
- **The selection tool belongs to its task.** A box can be drawn on the
  picture only while *Redraw & upscale* is the tab on screen **and redraw is
  the task** — the upscalers have no use for a box — only on the result rather
  than the original, and not while the panel is folded away: drawing one that
  nothing visible can act on is the tool turning up where it does not belong
  (François, 2026-09-19). The selection itself survives the switch.
- **◨ Hide details** folds the right column away and gives the whole card to
  the picture, for the landscape images where width is the limit and height is
  not. Like the tab, it lives on `GenerationDetailHost`: it follows the shape
  of what is being looked at, not which image is open.
- **Prompts are not table rows.** Prompt, negative prompt and a redraw's
  instructions are prose, and the table's label column left them a sliver of
  the width for the field the page is most often read for. They sit full
  width above the table, cut to four lines, and a click — on the text or its
  ⤢ — opens the whole of it over the page (François, 2026-09-19).
- **The pages show a scaled copy, never the file.** drift makes big images on
  purpose — a PiD upscale of a 2048 picture is 8192², 150 MB on disk and about
  256 MB decoded — and the detail view gives it a 700 px box. Fetching and
  decoding the original took 694 ms of frozen main thread on an idle machine,
  and seconds of it on a busy one (measured 2026-09-19). So
  `GET /api/outputs/{date}/{file}/preview[?side=]` answers a copy whose longest
  side is at most 2048, made once and cached under `~/.cache/drift/previews/`:
  43 ms for the same picture, sixteen times quicker. An image already that
  small, and any video, is served unchanged.
- **⤢ Full size** opens the file itself in a new tab — there is no zoom in the
  detail view, so that is where full resolution is looked at
  (François, 2026-09-19). It carries `rel="external"`, which is the only thing
  frontroute's `LinkHandler` consults: a same-origin `<a>` without it is
  swallowed, `preventDefault`ed and pushed into the router, which finds no
  route for `/api/outputs/…` and renders the shell around nothing. `target`
  is never read.
- `GET /api/outputs/{date}/{file}/size` answers the *file's* dimensions, read
  from its header. The selection for a redraw is made on the preview but has
  to be recorded in the pixels of the file it will be repainted from (27), so
  the viewer asks for them and uses the preview's size only until the answer
  arrives.
- **The open image is in the URL**: `/gallery/<generation id>[/<output index>]`,
  and `/projects/<id>/<generation id>[/<output index>]` for a workspace. A
  refresh, a bookmark or the back button reopens what the address names —
  which matters while repainting, where a pass takes minutes and the page is
  reloaded to see the result (François, 2026-09-19).
  `GenerationDetailHost.boundToUrl` wires it both ways, each side checking the
  other before it writes, so they do not chase each other. The gallery holds
  only its newest days, so a named generation that is in none of them has its
  day looked up and loaded on arrival (`HistoryService.Command.LoadGeneration`)
  — before that it appeared only once the visitor happened to page back far
  enough to reach it (François, 2026-09-20);
  `AppShell` routes `pathPrefix("gallery")` and
  `pathPrefix("projects" / segment)` with `extractUnmatchedPath.signal`, the
  page tab convention of `specs/01`.

## Shape

- Endpoints (`shared/.../History.scala`): `GET /api/history` (days with a
  sidecar count), `GET /api/history/{date}` (one day's generations),
  `GET /api/history/generation/{id}` (the day holding one generation — one
  existence check per date directory, since the sidecar is named after it),
  `DELETE /api/history/{date}/{id}` (every file named `<id>-*` or `<id>.*`,
  then the day directory once empty). Ids are `g<millis>-<n>`, so one id is
  never a prefix of another's files.
- Backend `sdserver/GenerationHistory` walks the outputs root; undecodable
  sidecars are logged and skipped. Failed generations write no sidecar, so
  history is implicitly "things that produced an output".
- Frontend `services/HistoryService` (day index + loaded days, outlives the
  page), `pages/gallery/`: `GalleryPage`, `GenerationCard`,
  `GenerationDetailHost` (one detail per opened generation, via `split`),
  `GenerationDetail` composed of `GenerationMediaViewer`,
  `GenerationParameters`, `RecordedParameters`, `GenerationLineage`,
  `PostProcessSection` with `EsrganUpscaleRow`, `PidUpscaleRow`, `RedrawRow`,
  `LiveSessionNotice`, `PostProcessJobList`.
- Reuse rides `GenerationService.requestReuse` / `takePendingReuse`: the
  gallery sets it and navigates; the freshly built panel consumes it after
  seeding from capabilities. On `Reuse.Full` the recorded request is the
  **base** the submit copies the form over, so fields the form never shows
  round-trip. On `Reuse.Task` there is no base; a sampler the target does not
  list falls back to "(model default)" and recorded LoRAs are named and
  skipped — a LoRA suits a model, not a task.

## Notes

- Thumbnails are the full-size files with `loading="lazy"` (videos
  `preload="metadata"`); hundreds of megabytes scroll fine, so there is no
  backend thumbnailing.
- Deletes go out as the per-generation call even in selection mode; a batch
  endpoint would be a second path to the same thing.
- The detail host must build one detail per opened generation and feed it
  only fast, `.distinct` signals: rebuilding on every status tick lags the
  page, flickers the original/result toggle and loses typed upscale fields
  (bugs/24).
- Laminar/Bulma traps: `emptyNode.asInstanceOf[HtmlElement]` throws (it is a
  `CommentNode`); `.card:not(:last-child)` margin breaks a stretched grid row;
  Bulma's table paints its own background under the theme's text.
- `DRIFT_CONFIG_DIR` and `DRIFT_OUTPUTS_ROOT` let a test run work on scratch
  copies so delete and launch never touch the real files.
