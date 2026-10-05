# Bug 41 — Importing a large picture into the gallery is refused

**Status:** open (found 2026-10-05, during the spec 52 end-to-end run; not investigated)
**Severity:** medium (a picture of a few megabytes cannot be imported; the message names a buffer, not the cause)
**Files:** `shared/src/drift/shared/History.scala` (`importHistoryImage`, `ImageImport`), the server options the
backend's endpoints are served with

## Symptom

`POST /api/history/imports` with a 2048 × 3072 PNG of 8.2 MB (11 MB as the base64 data URL the request carries)
answers 400:

    Invalid value for: body (too long string exceeded 'maxCharBufSize', offset: 0x00a7abd2)

The same picture as a 2.8 MB JPEG is imported. Seen through the API; the gallery's **Import images** button sends
the same request, so a large PNG dropped there should fail the same way — not checked in a browser.

## To try

The JSON reader's `maxCharBufSize` for this endpoint, as `NativeJobs` raises it for tiles
(`ReaderConfig.withMaxCharBufSize(256 * 1024 * 1024)`), or the picture sent as a multipart body instead of a string
inside JSON.

## A second report, closed: an imported picture of 706 × 517 could not be redrawn (François, 2026-10-05)

**Closed the same day**: he tried again on the current build and it redraws; the earlier failure was not kept.
What follows is what was checked.

"Its dimensions were not multiple of 32 and I was unable to redraw it." Not reproduced on the build of 2026-10-05,
through an isolated copy of drift, with a 706 × 517 PNG imported the same way:

- Qwen Image 2.1 (size multiple 32) and FLUX.2 dev (16) on the runner: the whole picture redrawn, 706 × 517 out
  (padded to 736 × 544 for the model and cut back);
- a selection of 300 × 250, and a redraw with 128 px of context around the tile: done;
- from the panel in a browser: the job starts and runs its one tile.

No job log on his machine names a 706 × 517 source, so his redraw was refused before a job existed, or never left
the browser. To pin down: the message he saw, the model, and whether it was the whole picture or a selection.
One thing did fail in the test, for another reason: the same redraw forced onto sd-cpp ended with "sd-server exited
with code 0 while loading" — its log says "no GPU devices; using the default backend", which is the isolated copy's
environment, not the picture's size. Not looked into further.
