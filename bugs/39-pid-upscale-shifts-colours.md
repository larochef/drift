# Bug 39 — A PiD upscale shifts the picture's colours

**Status:** fixed in code 2026-10-05 (`PidUpscale` passes `colourMatched` as `correctTile`, as redraw and SeedVR2
do; compiled, backend tests green); not yet seen on a drift upscale
**Severity:** medium (every PiD upscale; plainest on an illustration: paler,
reds less warm, white paper turned to a grey-green gradient)
**Files:** `backend/src/drift/backend/postprocess/PostProcessImages.scala`
(`colourMatched`), `backend/src/drift/backend/postprocess/TiledJobs.scala`
(`correctTile`), the PiD job beside `Redraw.scala` and `SeedVr2Upscale.scala`

## Symptom

A picture upscaled ×4 by PiD on the runner does not keep its source's colours.
Measured on six subjects (`~/dev/redraw-experiments/2026-10-04-auto/subjects`,
`<name>.png` against `<name>-4k.png` brought back to the source's size), as
the mean difference after a 12 px blur, in levels of 255: illustration 7.3
(up to 34), landscape 6.9, beach 4.9, portrait 3.8.

Those upscales were made by the experiment's harness, the whole picture in
one pass at 4096 × 6144 and 6144 × 4096 — not through drift, which tiles. To
confirm on a drift upscale before fixing.

## To try

Redraw and SeedVR2 tiles are colour-matched to their source
(`correctTile = Some(PostProcessImages.colourMatched)`: the result's detail,
the source's low frequencies); a PiD upscale is not. Pass the same correction
to the PiD job, the source tile scaled to the result's size.
