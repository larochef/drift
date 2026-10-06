# 30 — Gallery ergonomics, image import, visible post-processing

**Status:** planned — three items built, three remain
**Depends on:** 12, 15, 26, 27, 29

The gallery detail view is where post-processing happens, and a few things
around it get in the way once images are large, batches are common and
prompts are long. Plus one gap: post-processing only accepts gallery entries.

## Built

- **NSFW filter in the gallery.** "Show NSFW projects" in the gallery
  toolbar, off by default, kept for the page's life. Off, a generation whose
  project is flagged NSFW is hidden — derived entries carry their source's
  project, so their upscales and redraws too — and NSFW projects leave the
  project filter, which falls back to all projects if it was on one of them.
- **One tile per batch image.** The grid shows one `GenerationCard` per
  output, keyed by generation and output index, opening the detail on that
  image. A batch's tiles sit side by side with an accent along their top and
  a "▦ 2/4" badge; selecting one ticks the whole generation, since a delete
  removes it all.

- **Importing images.** "⤓ Import images or videos" in the gallery toolbar, or files
  dropped anywhere on the gallery, `POST /api/history/imports` (file name +
  data URL). Each image becomes a gallery entry of its own: kind `import`,
  completed, one output `<id>-0.<png|jpeg>` and its `<id>.json` sidecar under
  today, the usual `g<millis>-<n>` id; no session, no run configuration
  (shown as "Imported"), no request, no project, its original name in
  `importedFileName`. Post-processing takes it like any recorded output and
  derivations chain from it. Only PNG and JPEG, judged from the bytes — what
  sd-cli and the JDK decoders read; anything else is refused, not converted.
  One picked image opens in the detail view; several stay in the grid.
  Videos go the same way from the same button and drop, as a file body:
  `POST /api/history/imports/videos?fileName=…`, written to disk as it arrives
  and moved into the gallery unchanged, `<id>-0.<webm|mkv|mp4|mov>` judged from
  the container's first bytes (`GenerationImports.videoFormatOf`).

## Remaining

- **Switching between a batch's images.** The thumbnails that switch between
  a batch's outputs sit below the image and need a scroll to reach. They
  should be reachable without scrolling (beside the image, overlaid, keyboard
  arrows — to decide).
- **Long prompts.** A prompt worked with the assistant is long and pushes
  everything else down the detail view. It should take little room by default
  (folded to a few lines, expandable, or moved out of the parameter table).
- **Post-processing that is easier to follow.** A running job shows its
  progress only in the detail view of its source. It should be findable from
  anywhere — beside the global downloads panel, or on the gallery card of the
  source — with the tile count PiD and redraw already report.
