# Bug 30 — PiD upscale: which tile size, on which runner

**Status:** answered 2026-09-25: the runner's one pass wins; the rule stays (Result)
**Severity:** medium (image quality of every PiD upscale; nothing fails)
**Files:** `shared/src/drift/shared/PostProcess.scala` (`MaxTile`, `RunnerMaxTile`, `maxTileFor`, `tilesFor`), `backend/src/drift/backend/postprocess/PidUpscale.scala`, `specs/26-tiled-pid.md`, `specs/42-drift-runner.md` (PiD one pass)

## Symptom

The same upscale is decoded in very different pieces depending on the runner, and nobody has checked which gives the better image:

- **sd-cpp** tiles the output at 1536 px at most (`MaxTile`): 1024² → 4096² is 3 × 3 tiles, each decoded from a 384 px crop of the reference. Above that, ROCm aborts in the attention scale's kernel launch (spec 26).
- **drift runner** takes 4096 px (`RunnerMaxTile`): 1024² → 4096² is a single pass from the whole 1024 px reference.

## The question

The built-in PiD architectures are all 1024 → 4096, which suggests the model was trained to see a whole 1024 px reference. If so:
- a 384 px crop is off its training size, and each tile also loses the rest of the picture as context: sd-cpp's tiles may look worse (detail, consistency between tiles), not only show seams;
- the runner's one pass would then be the right way, and the runner should tile only above 4096, in tiles of about 1024 px of reference.

It could also go the other way: smaller tiles may give more detail per pixel. Only a comparison can tell.

## Root cause

`maxTileFor` picks the tile from the runtime's release tag, a capacity limit (what fits without crashing), never from quality. The rule also keys on `Runtime.DriftRunnerTag`, which spec 43 replaces with the runner kind.

## Suggested fix

1. Make a purpose-built test subject (fine texture, text, faces at several scales, straight lines crossing tile borders) at 1024².
2. On the drift runner, which can do both, decode it to 4096² in one pass and in sd-cpp's 3 × 3 layout (force `maxTile` = 1536). On sd-cpp, the 3 × 3 layout.
3. Compare detail, tile-to-tile consistency and seams at 100%.
4. Set the tile per runner from that result:
   - if one pass wins, the runner keeps 4096, and above 4096 tiles on 1024 px reference crops;
   - if sd-cpp's tiles are clearly worse, say so in the PiD upscale UI, or prefer the runner for PiD when it is installed.
5. Move the choice onto the runner kind of spec 43, off the release tag.

## Verification

- The comparison images kept beside the result in `specs/26-tiled-pid.md`, with the conclusion.
- `tilesFor` tests updated to the chosen per-runner rule.

## Result (2026-09-25)

Measured through drift's own job, on an isolated drift (a copy of the
configuration, scratch outputs). Subject: 1024², Qwen Image 2.1, seed 42 (a
freckled face, a crowd at several distances, a "FRESH BREAD DAILY" sign, knit,
brick, fence bars crossing the tile borders). PiD `pid-flux2`, 4 steps, cfg 1,
seed 7, the runtime picked per job.

| | drift runner, 1 × 1 | sd-cpp master-919, 3 × 3 of 1536 |
|---|---|---|
| RMSE, result scaled back to 1024, against the source | 0.0588 | 0.0633 |
| luminance / deviation (source 0.272 / 0.273) | 0.278 / 0.275 | 0.263 / 0.290 |
| per-tile luminance against Lanczos ×4, in ‰, spread | +15 … −2 (17) | −1 … −29 (28) |
| time, model load included | 5 min 30 s | 7 min 01 s |

- The tiles drift apart in tone: the bottom-middle tile is 3% darker, and the
  knit changes colour across an overlap. The one pass holds one tone.
- The tiles lose context: a face half hidden behind hair, whole in the one
  pass, becomes a dark blob in its tile.
- Detail at 100% is comparable (hair, freckles, sign edges). The tiles are
  crunchier and darker, like the flash-attention drift of spec 26.
- The one pass invents nothing. The sign and chalkboard read as in the source.

So the runner keeps 4096, which above 4096 already tiles on reference crops of
about 1024 px. The choice is on the engine (`maxTileFor(RuntimeEngine)`, spec
43), and `PidTargetTests` covers both rules. Left open, as product choices: say
in the PiD panel that sd-cpp's tiles are weaker, or prefer the runner for PiD
when it is installed.

Not isolated: the runner forced to 1536 tiles, which needs a tile-size knob
the job does not have. So the gap mixes engine and tiling. The tone drift
between tiles is a tiling effect either way. One subject and one seed.

Found on the way: bug 31 (a job server on sd-cpp right after one on the
runner can fail to start).
