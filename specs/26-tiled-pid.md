# 26 — Tiled PiD on the GPU

**Status:** done
**Depends on:** 15

A PiD decode that looks like a good upscale and runs on the GPU. Flash
attention spoils PiD and, without it, one pass on ROCm is capped at 1792²
(15). sd-cli tiles only VAEs, video frames and ESRGAN, so drift tiles the
decode itself: the reference is cut into overlapping tiles, each decoded at
×4 as one img_gen job on a single sd-server that stays up for the whole job,
and the results are feather-blended.

## What it does

- Any source size, ratio kept: with no size given the target is ×4 of the
  source's real pixel size (read from the file), longest side capped at 16384;
  an explicit target crops another ratio. 1920×1080 passes.
- **A target no larger than the source is refused**, asked for or capped into
  being: PiD decodes a quarter of the target, so such a job shrinks the source
  to that quarter and hands the same size back — which is what a 8192² source
  did before the cap moved (François, 2026-09-22). The panel says the target in
  pixels beside "×4 of the source", and the refusal in its place, with the
  button disabled.
- Adaptive layout per axis: the fewest tiles of at most 1536 px overlapping
  by at least 256 px, then each tile as short as that count allows (a multiple
  of 64), starts spread evenly on multiples of 4 so every tile is an exact
  crop of the quarter-size reference. 512² → 2048² is 2×2 tiles of 1152;
  1024² → 4096² is 3×3 of 1536; a target that fits one tile is one tile.
- The job shows "tile n of m" over a real progress bar, then "blending".
  The same seed is used for every tile; a random one is drawn per job.
- Each tile is judged on its own (a refused or failed img_gen job, no output,
  wrong size, entirely black); the first failure fails the job naming the tile.
- The models load once per job, or not at all when a session of the PiD
  configuration is ready on the job's runtime.
- A runtime older than sd-cpp **master-892** is refused, naming it: before
  that build sd-server scales every reference image to the request's size
  (Notes).
- Four built-in PiD architectures, all 1024→4096, the same Gemma 2 2B text
  encoder and its `tokenizer.json` (`--tokenizer`, required: sd-cpp embeds no
  Gemma 2 tokenizer and refuses to build PiD without one), the VAE and
  `--vae-format` sd-cpp's `docs/pid.md` pairs with it:

  | Architecture | Decoder | VAE family | `--vae-format` |
  |---|---|---|---|
  | `pid-flux2` | `pid_1.5_flux2_…_4step_bf16` | `flux2-klein-9b-vae` | `flux2` |
  | `pid-flux1` | `pid_1.5_flux1_…_4step_bf16` | `z-image-vae` (Comfy's Z-Image `ae.safetensors` is the Flux VAE byte for byte) | `flux` |
  | `pid-qwen-image` | `pid_1.5_qwenimage_…_4step_bf16` | `qwen-image-vae` | `wan` |
  | `pid-flux2-2048` | the 1.0 `512_to_2048` file | `flux2-klein-9b-vae` | `flux2` |

## Shape

- `shared/.../Tiling`: `axis` and `layout` (PiD-agnostic: tile size, overlap,
  multiple, alignment as parameters), the `Overlap` both sides count on, and
  the partial-redraw window (27) — integer geometry the browser shares, to lay
  out the same tiles and price a job before it runs.
  `backend/.../postprocess/TileBlending`: `blend` (rows painted into a strip,
  every overlap a linear ramp per RGB channel, one row of tiles in memory at a
  time) and `paste` (27).
- A cancelled job (15) stops between two tiles: `runTiles` checks the flag
  before each one, and the tile's img_gen job is cancelled on the server it
  was submitted to — `runTiles` hands `NativeJobs.image` a callback with the
  native job's id for exactly that. The job server of the job's own is stopped as always.
- `TiledJobs.runTiles` serves PiD and redraw (27): one sd-server for the job —
  a ready session of the configuration on the job's runtime, else a job server
  (27) — and one img_gen job per tile, built by the job's `request`. PiD's:
  the tile crop as the only `ref_images` entry with `auto_resize_ref_image:
  false`, the tile's size, the request's prompt, steps, cfg and seed, the
  configuration's LoRAs as `lora`, and the server's defaults for the rest
  (sampler, scheduler, VAE tiling). `TiledJobs.TileOverlap` = 256,
  `PidUpscale.MaxTile` = 1536, `SdCppBuilds.FirstKeepingReferenceSize` =
  892, checked against the runtime's release tag (`master-<build>-<sha>`).
- `PidUpscale` scales the source to a quarter of the target (fill), pads it to
  multiples of 16 by edge repetition, holds the reference in memory, and crops
  the blend back to the target. Crops and results are written beside the job
  log and deleted when the job ends.
- `PostProcessJob.progress: PostProcessProgress(completed, total)`.
- A PiD request carries a tile size and a grid offset in target px
  (`tileSize`, `gridOffsetX`, `gridOffsetY`), as a redraw's does (27): the
  panel holds the tiled tasks' shared piece (`TileAreaFields`, `Cut.Whole`),
  its grid drawn on the source and moved there. Browser and backend lay the
  tiles out from one `UpscaleTiling` (`PidUpscaleRequest.tilingFor`); the tile
  is kept between 1024 px and the runtime's largest.
- The PiD panel (`pages/gallery/PidUpscalePanel`): the configuration, then
  Advanced (15) holding the target size, steps, seed and the prompt it sends —
  the source's, inherited through the derivation chain; "→ ×4" and the button
  on the last line, and empty size fields send no size. Configured LoRAs go
  in the request's `lora` array (28).

## Notes

- Until sd-cpp master-892, sd-server scaled every reference image to the
  request's width × height while decoding it, whatever
  `auto_resize_ref_image` said, so PiD saw a blurred full-size reference
  instead of the quarter-size crop and returned a smear (bugs/25). PiD ran one
  `sd-cli` per tile meanwhile, reloading the models every time. Reported as
  leejet/stable-diffusion.cpp#2004, fixed by #2011 (merged 2026-09-21): the
  opt-out now holds from the decode on, while the default still scales — so
  PiD sends `false` on every tile and refuses older builds rather than
  smearing.
- The PiD built-ins carry no `--diffusion-fa` and no `--backend` assignment:
  the whole run is on the GPU. Run configurations carry no parameters, so
  every PiD job takes this from the architecture.
- **drift's runner decodes 4096² in one pass**, all three variants (`specs/42`,
  step 13, 2026-09-25): `PidUpscaleRequest.maxTileFor` gives 4352 (4096 + the overlap)
  on the drift runner engine and 1536 on sd-cpp, in the backend's job (the launch runtime's) and
  in the panel's preview (the configuration's runner), and
  `SdCppBuilds.keepsReferenceSize` counts the runner in.
- **One pass beats sd-cpp's 3 × 3, measured 2026-09-25** (bug 30): the same
  1024² subject (Qwen Image 2.1: a face, a crowd, a sign, knit, a fence
  across the tile borders), `pid-flux2`, 4 steps, seed 7, through drift's job.
  Against the source, the runner's one pass is closer (RMSE 0.0588 against
  0.0633, luminance 0.278 / deviation 0.275 against 0.263 / 0.290, source
  0.272 / 0.273). It also keeps one tone across the frame. The 3 × 3 tiles
  drift apart, the bottom-middle tile 3% darker than a Lanczos upscale, and
  the knit changes tone across an overlap. A face half hidden behind hair
  survives the one pass and turns into a dark blob in its tile. Fine detail
  is comparable, the tiles' crunchier. 5 min 30 s against 7 min 01 s, loads
  included.
- **Why 4352 on the runner, measured 2026-09-25:** one 1088 → 4352 pass
  (the same subject, scaled) decodes as well as 1024 → 4096: same detail,
  luminance 0.274 / deviation 0.274 against the source's 0.272 / 0.273,
  nothing invented, 96 s a step against 80. With the overlap added to 4096,
  n tiles cover n × 4096. So 8192² is 2 × 2 tiles instead of 3 × 3, and
  16384² 4 × 4 instead of 5 × 5. What follows is sd-cpp's.
- **Why the tiles cannot become one 4096 pass, measured 2026-09-20** (sd-cli
  outside drift, `pid_1.5_flux2_1024_to_4096`, ROCm gfx1151, sd-cpp
  master-881-17860c0). Without flash attention a single 1024→4096 pass never
  starts: sd-cpp asks for **400 GB** of attention memory ("model manager cannot
  make enough memory available on ROCm0: need 400225.12 MB device … available
  62954.86 MB"), 768→3072 fails the same way and 512→2048 crashes in `sample`.
  Flash attention makes the memory fit — 1024→4096 runs to completion in 457s,
  exit 0, no error — but the result is **uniformly black above 2560**: clean to
  640→2560, black from 672→2688 up. Cut to `--steps 1` the failure shows itself:
  whole **8×8 pixel blocks return zeros** (98.4% of 8×8 blocks are wholly zeroed
  or wholly clean, boundaries exactly on multiples of 8, so it is patch tokens
  being lost), clustered in the brightest part of the frame. 0.05% of the frame
  at 2560, 5.98% at 3072, 8.30% at 4096; from the second step those zeros feed
  back and the whole frame collapses. Not the overflow
  `docs/troubleshooting.md` describes, and its cure does not touch it:
  `--attn-scale` reaches the model (the single-step image moves, RMSE 0.027) but
  leaves the corrupt share where it was, 5.98% → 6.04%, and 0.00390625,
  0.000244140625 and `--attn-scale 1` all give the same black frame at 3072.
  Not the sampler either — `lcm` and `euler` agree. Reported upstream as
  leejet/stable-diffusion.cpp#2003 (2026-09-20); nothing to tune here.
- Flash attention *does* work at tile scale and is **3.5× faster** there
  (384→1536: 57.4s without, 16.4s with), but it is still off, because it moves
  the picture away from its source — which is the whole job of an upscaler.
  Same seed, same prompt: source luminance 0.2588 and deviation 0.2661 become
  0.2363 / 0.2870 without flash attention and 0.2251 / 0.3044 with it, and the
  RMSE against a Lanczos upscale of the same source is 0.0481 without and
  0.0765 with. Darker, crunchier, less faithful — the "fogged / wrong light"
  complaint from the other side. No scale value shifts those numbers by more
  than the fourth decimal, so there is nothing to tune here either. And a tile
  is already 1536² at most (`PidUpscale.MaxTile`), which is the ceiling a single
  pass without flash attention reaches here — 2048 crashes in `sample` — so
  there is no bigger-tile win to take either.
- Padding to multiples of 16 by edge repetition was chosen over a
  multiple-of-64 rule because it is barely more code and lets any well-known
  resolution through.
- The overlap blend averages two renderings of fine detail and is slightly
  softer there; narrower overlaps keep more of it.
- Tiles run sequentially; the overlap is a constant, not a setting.
