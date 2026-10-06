# Bug 42 — A SeedVR2 upscale done ×2 then ×2 shows its tiling on some tiles

**Status:** fixed in code 2026-10-06 for the line inside one pass (Found, below), measured on the picture that
showed it: the step at y = 1360 is gone. For François to look at on his own pictures. Whether drift's own tile
grid also shows on a ×2×2 is still not reproduced (reported by François 2026-10-05)
**Severity:** medium (seen on some tiles only, sometimes; the upscale is otherwise the one he prefers on
pictures — "way better than PiD")
**Files:** `backend/src/drift/backend/postprocess/SeedVr2Upscale.scala`,
`shared/src/drift/shared/PostProcess.scala` (`SeedVr2UpscaleRequest.tilesFor`, `MaxTile`),
`shared/src/drift/shared/Tiling.scala` (`Overlap`), `backend/src/drift/backend/postprocess/TileBlending.scala`

## Found (2026-10-06): the runner's VAE stitch steps where a tile starts

The line measured below is inside one `upscale` job, at y ≈ 1360 of a 2048 × 3072 output. That is exactly where
the third of four VAE tiles starts: `VaeTiles.split(3072, 1024, 128, 16)` gives starts 0, 672, **1360**, 2048
(overlaps grown to 352, 336, 336), for the encode and for the decode alike. The attention windows are ruled out:
for that grid they are 25 token rows high (400 px), with boundaries at 400, 800, 1200, 1600 and, shifted, 200,
600, 1000, 1400 — bug 40's "17 tokens" is the windows' width.

`VaeTiles.stitch` was diffusers' `_stitch_tiles`: a tile is blended with the one above, then with the one to its
left — both *as made*. The left neighbour's top rows were never blended with what is above them, so over the
whole overlap between two columns the picture jumps, at the row where the tiles start, from the row of tiles
above to the left tile's raw top edge (weight 1 at the overlap's left, fading to its right). The width of 2048
is three columns overlapping by **512 px each** — half the picture's width carries the step, in the middle,
where the chest is. A VAE tile's edge rows are also its worst (less context), hence "smoother above, more
textured below".

Fixed: `stitch` is now a weighted mean — every tile ramps over each overlap on both axes at once, so nothing
steps (`TinyModelTests`, "VaeTiles stitches tiles that disagree without a step", fails on the old code).
MiniMax H3's VAE keeps the old one as `stitchAsDiffusers`: it is checked against its reference's frames to the
pixel, and its video has the same flaw to look at later.

Measured (`~/dev/redraw-experiments/2026-10-06-seam`, `run.py`; the same source, model and seed as
`beach-2k-raw.png`, through an isolated drift): the mean change from one row to the next over the middle half of
the width is 3.82 levels at y = 1360 in the old picture against about 2.0 on the rows around it, and 2.11 in
the new one — a row like its neighbours. The two other rows where a tile starts had the same step in the old
picture and lost it: y = 672, 7.6 against about 5.4, now 5.5; y = 2048, 2.1 against about 1.0, now 1.2. `seam-old-new.jpg` has the two crops, old above new. The "shorter, fainter line a hundred pixels higher" of the 1k re-upscale and
the last 8 rows and columns stepping harder (bug 40) are not explained by this.

## Symptom

A picture upscaled by SeedVR2 ×2 and then ×2 again: on some of the tiles the tiling is visible in the
result. Not on every tile, not on every picture.

To record when it is seen again: the picture's sizes at each pass, where the mark is (on the line between
two tiles, or a whole tile that looks different from its neighbours), and whether it is already there after
the first ×2.

## What the job does today

A picture is a tiled job like PiD's: the padded source times the scale, cut into tiles of `MaxTile` target
px at most that overlap by `Tiling.Overlap` = 256 target px, blended by the feather `TileBlending` gives
every tiled job; each tile is colour-matched to its own crop of the source. One tile for a target within
`MaxTile`, so the first ×2 of a 1k picture is usually a single tile and the second ×2 is the pass that is
tiled — on a source that is itself a SeedVR2 output.

The overlap is counted in target px: at ×2 the two neighbours share only 128 px of the source.

## Leads

- **Each tile is restored on its own.** SeedVR2 is a diffusion restorer: two tiles invent different fine
  detail over the same source pixels, and a feather across 256 px mixes two textures rather than joining
  them. A redraw has the same problem and answers it with more context per tile.
- **François's suggestion:** more tiles, or bigger ones, as the redraw does — a wider overlap, or each tile
  restored with a margin of the picture around it that is cut away afterwards, so the kept part of two
  neighbours was drawn from the same surroundings.
- **Per-tile colour match:** each tile gets its own correction against its crop; two neighbours corrected
  differently differ by a step in tone that the feather turns into a visible band.
- **Not the same as bug 40's line:** that one is a horizontal line inside a single pass (one `upscale` job,
  no drift tiling), put down to the VAE's own tiles or the transformer's attention windows. A mark that
  follows drift's tile grid is this bug; one that does not is that one. Both may be on the same picture.

## Measured 2026-10-06 (`~/dev/redraw-experiments/2026-10-05-edit`, `FINDINGS.md` §8 and §15)

On a generated subject (a chest of a SeedVR2 4k with a horizontal line through the skin):

- the line is **already in the 2k picture**: upscaling the 2k window again leaves it where it was. It is drawn by
  the first ×2 pass — bug 40's line inside one pass — and carried by the second;
- no redraw removes it (bug 46), and a full redraw at 0.4 erases the freckles and pores around it;
- the same window cut from the **1k original** and upscaled ×2 then ×2 has no line there, and its selection
  pasted into the picture within 64 px shows no join.

**A repair to build:** "upscale this selection again" — on a picture that is an upscale, a box over the line; the
window around it is cut from the picture it was upscaled from, upscaled by the same model, and the selection
pasted back (`TileBlending.paste` with its `reach`). The cause stays bug 40's to find.

## To try

- Reproduce on a generated subject (never the gallery): ×2 then ×2 through drift with *keep tiles*, and
  look at the raw tiles beside the blended picture — is the difference in the tiles or made by the blend.
- The second pass with the overlap doubled, and with tiles restored through a margin (as a redraw's
  context margin) and cropped.
- The second pass without the colour match, to rule it in or out.
- The second pass as one tile when memory allows, as the reference of what no tiling looks like.
