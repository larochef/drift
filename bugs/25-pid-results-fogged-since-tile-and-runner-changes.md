# 25 — PiD results look fogged, as if steps were missing

**Status:** fixed 2026-09-14 — PiD tiles ran on sd-cli again; François
confirmed on a real PiD run. Fixed upstream 2026-09-21
(leejet/stable-diffusion.cpp#2004, #2011, sd-cpp master-892): PiD is back on
one sd-server per job, sending `auto_resize_ref_image: false`, and refuses
older builds.
**Reported by:** François (2026-09-14): PiD upscales of a 1024² image "didn't
fail, but the result was strange, like fogged, as if steps were missing. There
were only 9 jobs for a 1024x1024 image, I think before we had more tiles."

## What happens

Two PiD jobs on 2026-09-14 (00:06 with `pid-flux-1`, 00:16 with
`pid-flux-2-1024-to-4093`, same 1024² source, 4096² target) completed all
tiles without an error, and both results look foggy.

## Cause: sd-server upscales every reference image to the output size

PiD is conditioned on the **low-resolution** crop (a quarter of the tile's
side). sd-cli encodes the reference at its own size; sd-server scales it up to
the request's width × height while decoding the request, so the model gets a
blurry full-size reference and returns a smear.

Seen in every job log: sd-cli tiles log `encode_first_stage completed, taking
0.42s`; server tiles log `media_io.cpp:592 - resize input image from 384x384 to
1536x1536` then `encode_first_stage completed, taking 6.30s`.

Upstream (master, `examples/common/common.cpp`): `from_json_str` decodes
`ref_images` through `decode_base64_image(…, width, height, …)`, the request's
output size as the expected size. Nothing turns it off:

- `"auto_resize_ref_image": false` in the request only appends
  `resize_before_vae=0` to the ref-image args later — tested, still resized;
- `--ref-image-args resize_before_vae=off` at launch — tested, still resized;
- sdapi `extra_images` and the OpenAI edits route pass the output size too once
  one is given, and PiD always gives one.

## The test (2026-09-14)

One tile from `out/images/original.png` (512²): the top-left 288² crop as the
reference, 1152² output, seed 42, empty prompt, 4 steps, cfg 1,
`pid_1.5_flux2_1024_to_4096_4step_bf16`, runtime `master-859-7f410a3-rocm`.

| Run | Laplacian SD (sharpness) | Look |
|---|---|---|
| sd-cli | 17.1 | sharp fur, whiskers, eyes |
| sd-server (drift's request) | 11.4 | blocky smear, eyes barely there |
| bicubic ×4 of the crop | 2.8 | soft |

The tile count (16 × 1280 → 9 × 1536) is not the cause: the 09-13 14:37 sd-cli
run already had 9 × 1536 tiles and was fine.

Redraw (spec 27) sends its whole-image reference the same way, so it is also
resized to the tile size — less harmful there (an edit model reads any size),
but the encode is slower than it needs to be.

## Fix

François's choice: PiD tiles back on sd-cli, one process per tile, as before
spec 27. `runTiles` takes a `TileRunner` — `Cli` for PiD (the configuration's
argv from `SessionManager.resolveArguments`, now resolved with the runtime's
rule like a server's, minus `withoutRequestFlags`), `Server` for redraw — and
shares the judging, blending and cleanup. The models load for every tile
again.

Reported upstream 2026-09-20 as leejet/stable-diffusion.cpp#2004, with the
`sd-server` / `sd-cli` command pair and a reference image, reproduced on
`master-881-17860c0`: the `ref_images` decode scales to the output size while
sd-cli keeps the reference's own size. Once fixed there, PiD can go back on the
job server.
