# Bug 40 — A SeedVR2 upscale damages small faces: black strands, faces partly erased

**Status:** fixed in code 2026-10-05 for the damage (a ×4 runs as two ×2 passes, `SeedVr2Pipeline.passes`; a 512
crop asked ×4 in one job on the rebuilt runner: 66 s, two passes); the faint line inside one pass is still open
**Severity:** medium (seen zoomed in, on faces that are small in the source;
the large faces of the same pictures are fine)
**Files:** `runner/src/drift/runner/models/SeedVr2Pipeline.scala` and the
SeedVR2 VAE beside it; `backend/src/drift/backend/postprocess/SeedVr2Upscale.scala`

## Symptom

On pictures upscaled ×4 by SeedVR2 7B
(`~/dev/redraw-experiments/2026-10-04-auto/subjects`, 1536 × 1024 sources →
6144 × 4096):

- `stall-4k.png`: strange black strands across the face of the girl in the
  white polo, seen when zoomed in.
- `street-4k.png`: the faces of the people walking are partly erased.

The 1k sources do not show it (François, 2026-10-04): the upscale adds it.

## Not drift's path

These upscales were made by the experiment's script (`s1c_seedvr2.py`): two
tiles of 1088 source px a picture through the runner's `upscale` job at ×4,
colour-matched and blended by ImageMagick — the same cut as drift's, not
drift's code. The raw tiles as the runner returned them are in `tiles/`
(`<name>-<n>-raw.png`): look there first, to rule the script's colour match
and blend out.

## Measured 2026-10-05: ×4 in one pass is the cause, ×2 twice is clean

`~/dev/redraw-experiments/2026-10-04-auto/sheets/50-scale-*.png`, 512 crops of the two sources, SeedVR2 7B:

- ×4 in one pass, crop alone: the same damage as in the two-tile picture — strands across the girl's face, an aged
  harsh face, a mesh over the walkers and their faces, a hatch on skin and foliage. So it is not the tile size, the
  colour match or the blend.
- ×2 then ×2: a clean young face, the walker's face drawn, no mesh, no hatch, sharp lettering.
- Same time: 58 s for ×4 against 8 s + 52 s.

François: "we should never do 4x but always 2x". To do: ×4 runs as two ×2 passes (in the runner's `upscale` job,
so pictures and videos both get it), and the panel's default follows.

## To try

- **Is ×4 too much?** ×2 twice against ×4 in one pass, on 512 crops of the two
  sources (the girl in the white polo, the walkers): `s5_seedvr2_scale.py`,
  sheets `50-scale-*.png` in the experiment's directory.

- The same crops on the 3B and the 7B "sharp" checkpoints, and at ×2.
- A face tile upscaled alone, so that the face is larger in the model's input.
- Against the Python reference on CPU, on one small crop.

Spec 52 meets this from the other side: these are the defects a single-tile
redraw should repair, and the assistant, shown the picture at 1024 and 1536,
reported nothing broken on `stall-4k.png`.

## A faint horizontal line inside one pass (2026-10-05)

`beach-2k-raw.png` (1024 × 1536 → 2048 × 3072, one `upscale` job): a hard horizontal line at about y = 1360, the
skin smoother above it and more textured below; it carries into the 4k (`view/seam-hunt.jpg`, `chest-x4-vs-x2x2.jpg`
in the experiment's directory). It is in the runner's raw output, so not the script's blend. 1365 is where the third
of four VAE tiles starts for a height of 3072 (tiles of 1024, overlaps grown to about 341), and 85 tokens down is
also a multiple of a 17-token attention window: VAE tiling (encode or decode) or the transformer's windows, not
told apart. `VaeTiles.stitch` read through against diffusers' `_stitch_tiles`: the arithmetic agrees — and that
arithmetic is the cause (2026-10-06, `bugs/42`, Found: a tile is blended with its left neighbour's raw top edge;
the windows are 25 rows high and do not fall there). Fixed in code, not yet seen on a picture. To do: the same
picture with one VAE tile (`tile` ≥ the picture) to tell the two apart. Also: the last 8 rows and columns of a raw
output step harder than the rest of the picture (a border effect of the padding to multiples of 16?).

## Not verified after the two-pass change

A ×4 job now prints two progress bars, one a pass. A tiled picture is one job a tile, so its tile count is right; a
video is one job for the whole video, and its progress may run to the end and start again for the second pass. Not
run: to look at on the first ×4 video.
