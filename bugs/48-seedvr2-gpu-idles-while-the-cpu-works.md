# Bug 48 — During a SeedVR2 upscale the GPU waits while the CPU works on the same thread

**Status:** open (raised by François 2026-10-06, from an earlier session's remark that was never written down:
"there is a time during the generation when the GPU is not 100 % used, because the CPU does some stuff in the
same thread … we could spawn another thread for the CPU stuff"); places read in the code, nothing measured
**Severity:** low to medium (speed only; how much is not known)
**Files:** `runner/src/drift/runner/diffusion/SeedVr2Pipeline.scala`, `runner/src/drift/runner/models/VaeTiles.scala`,
`runner/src/drift/runner/server/ImageServer.scala`; `backend/src/drift/backend/postprocess/TileRun.scala`

## Where the GPU has nothing to do (read, not timed)

**In the runner, one `upscale` job** — everything is on the `image-worker` thread, one thing after another:

- before the model: the picture decoded from base64 PNG and resized on the host;
- `VaeTiles.mapped`, encode then decode: each tile cut on the host, uploaded, run, read back (`toFloats`), and
  only then the next one cut; the stitch of all tiles on the host at the end (a 2048 × 3072 × 3 float picture);
- after the model: the result clamped, encoded as PNG and base64 — seconds for a 4352² tile — before the job
  is `completed` and the worker takes the next one.

**In drift, between two tiles of a tiled job** (`TileRun`'s loop is a fold, strictly one tile at a time): the
answer decoded from PNG, checked (`imageSize`, `isBlack`: the file read again), colour-matched against its crop
(`colourMatched`: two blurs of the whole tile), written as PNG, painted into the picture and into the live
picture — and only then the next tile's crop cut, encoded and submitted. The runner sits idle, model loaded,
for all of it.

## Measured 2026-10-06 (`~/dev/redraw-experiments/2026-10-06-seam`: `gpu.csv`, `run.log`)

`gpu_busy_percent` read five times a second during a ×2 (1024 × 1536, one tile, 73 s) and a ×2 of its result
(two tiles of about 120 s), SeedVR2 7B through drift:

| where | time | GPU load |
|---|---|---|
| runner, the 2k job | 73 s | 83 % on average; under 50 % for 6.4 s — 3 s at its start, 3.5 s in its middle |
| runner, a 4k tile | 118–125 s | 87–92 %; under 50 % for 1 to 7 s — again at the start and in the middle |
| drift, between the two 4k tiles | 6 s | idle |
| drift, after the last tile | 8.5 s | idle (nothing to overlap it with) |

So drift's work between two tiles costs about 5 % of a tiled upscale, and the runner's own gaps 1 to 9 % of a
job. The middle gap is between the encode and the decode — the stitch and the copies — not timed apart.

## To do

1. Measured, above: a few percent each. Worth doing for long jobs (64 tiles ≈ 6 minutes of idle GPU), not
   urgent.
2. **drift:** submit tile n + 1 as soon as tile n's answer is in, and finish tile n (checks, colour match,
   paint, files) in an ox fork — the tiles do not depend on each other for an upscale (`picture` is empty); a
   redraw or an edit, which cut the next crop from the painted picture, cannot.
3. **Runner:** the PNG encode and the stitch off the worker thread (the job stays `generating` until its result
   is written, the worker starts the next queued job's GPU work meanwhile); the next VAE tile cut while one runs.
