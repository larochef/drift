# 30 — Gallery ergonomics, image import, visible post-processing

**Status:** planned — two items built, four remain; nothing is designed yet
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

## Remaining

- **Switching between a batch's images.** The thumbnails that switch between
  a batch's outputs sit below the image and need a scroll to reach. They
  should be reachable without scrolling (beside the image, overlaid, keyboard
  arrows — to decide).
- **Long prompts.** A prompt worked with the assistant is long and pushes
  everything else down the detail view. It should take little room by default
  (folded to a few lines, expandable, or moved out of the parameter table).
- **Post-processing images that are not in the gallery.** Upscale, PiD and
  redraw only accept gallery entries (derived entries of a recorded
  generation). Open questions: an imported image as a gallery entry of its
  own (kind `import`, no request, its own sidecar) so derivations chain from
  it as they do now; where the import lives (gallery, drag and drop, the
  detail view); whether it joins a project.
- **Post-processing that is easier to follow.** A running job shows its
  progress only in the detail view of its source. It should be findable from
  anywhere — beside the global downloads panel, or on the gallery card of the
  source — with the tile count PiD and redraw already report.
