# 27 — Redraw: a low-strength, tiled img2img pass on a gallery image

**Status:** done
**Depends on:** 15, 26

An upscale keeps the source's detail but cannot invent what the source never
had; skin comes out smooth or waxy. A **redraw** runs an ordinary image model
over the image at low denoising strength, tile by tile, so the model repaints
texture while composition, colours and size stay — the "Ultimate SD Upscale"
idea applied to an image that already exists. It is its own operation on any
gallery image, same size in and out; after an upscale you redraw its result.

## What it does

- The redraw panel, one of the three post-processing tasks (15): a run
  configuration among the `SdCpp` architectures tagged `image`, and nothing
  else in the form: "show the grid" and "↺ reset the grid" are parameters of
  the picture rather than of the job, so they sit in the panel's footer under
  the line that says what the job would cost — the rest of what is read off the
  picture. Behind Advanced, in labelled groups: *prompt* (restoration template, and the
  instructions as a textarea) and *negative*, a textarea too, both being prose;
  *pass* (strength 0.4, steps — empty keeps the configuration's —, seed —
  empty draws one per job —, soften); *reference* (model default / always /
  never, and the reference size 768, hidden when none is sent); *area* (tile
  size 1280, window, margin); *options* ("keep the tiles").
  What the job would cost is the last line before "✨ Redraw".
- **The picture draws the tiles the job would run** while the redraw panel is
  open — "show the grid" in *options*, on by default: `RedrawGeometry.layoutFor`
  through `Tiling.layout`, the call the backend lays its tiles out with, over
  the whole image or over the window once a box narrows the pass. The overlaps
  read as the brighter bands, which is where `TileBlending` feathers
  neighbours together. One `RedrawGeometry` per keystroke feeds the cost line,
  the box's sticky lengths and the grid, so the three cannot disagree.
- **The grid can be moved**, which is how a face is kept out of a seam: a
  press on one of the grid's own lines (not the picture's edges, and the
  selection's handles win over it) drags the whole grid, snapped to the
  architecture's `sizeMultiple` and taken modulo the stride — shifting by a
  whole tile advance is the grid it started from. It rides on
  `RedrawGeometry.offsetX/offsetY` to the picture and on
  `RedrawRequest.gridOffsetX/gridOffsetY` to the backend, which lays its tiles
  out with the same `Tiling.axis` call, so what was seen is what runs. The
  panel's "↺ reset the grid" stays on screen, disabled while the grid is where
  it started: nothing else says the grid can be moved, a drag on a line being
  invisible until the pointer is on one. The offset lives on `GenerationDetail`
  — it survives stepping through a batch, whose images share a size, and starts
  again at nothing on the next image opened.
- **A shifted grid is a lattice, an unshifted one the even spread**
  (`Tiling.axis`): offset 0 keeps the fewest tiles of `tileFor`, their starts
  spread evenly and the last ending at the axis. A shift lays those same tiles
  every stride from `-phase`, cuts the two at the ends back to the axis and
  rounds them out to the multiple — so it usually costs one tile more, and the
  end tiles are shorter. An end tile under two overlaps is too little picture
  to paint, so it is grown to a full tile against that end and a neighbour it
  then covers whole is dropped; that bound (`2 × Overlap`) is also what keeps
  the grown tile touching what comes before it, so a shift can never open a
  gap. `TileBlending.blend` already took a tile's own width and a row's own
  height, so nothing there had to change.
- **The framing sentence differs by what the model was given**
  (`Redraw.framing`). With a reference: where the tile sits in it, then the
  reference named as context only — identity included — and "repaint only what
  this crop already shows". Without one: "This image is a close crop of a
  photograph. Repaint only what it already shows, at the same framing and
  scale: do not widen the view, do not add anything that is not already
  visible, and keep every element exactly where it is." Naming a larger
  photograph and quoting percentages of a frame the model cannot see described
  a scene it could only imagine, and ERNIE duly imagined one — repainting the
  whole picture inside a single tile (François, 2026-09-19).
- The identity sentence moved out of the built-in restoration template into
  that framing, for the same reason: it spoke of "the reference image" in
  every job, including the ones where none is sent (32; built-ins re-seed, so
  an existing install picks the shorter text up on its next start).
- The source's prompt is **not** sent. Every tile gets the built-in
  restoration prompt (a crop of an upscaled photograph; remove the upscaling
  artifacts; natural skin, hair and fabric detail; keep composition, pose,
  anatomy and colours; every person as in the reference: eye colour, skin
  tone, hair, facial features, ethnicity), the instructions appended, and a
  position sentence ("This image is the bottom left part of the reference
  image, from 0% to 50% of its width and from 43% to 100% of its height").
- Every tile gets the whole source, scaled to at most the reference size, as
  `ref_images` beside the tile crop as `init_image`. This is what keeps a tile
  from finding the prompt's subject where it is not.
- Instructions are descriptive, not commands ("natural skin texture, visible
  pores"); an img2img model follows descriptions of what should be seen.
- The job runs on a ready session of the chosen configuration if one exists,
  else on its own sd-server started for the job. The tiles are judged like
  PiD's; progress counts tiles.
- A box dragged over the image in the detail view redraws **that part
  alone**: the button becomes "✨ Redraw selection", a line above the fields
  reads the selection, the window it is repainted through and what the job
  costs in tiles, and a click on the picture clears the box. Everything else —
  the restoration prompt, the instructions, the whole image as reference, the
  tiling — is what a full redraw does.
- The result is a derived entry (`operation = "redraw"`) recording
  configuration, steps, seed, strength, instructions and the region.

- **Context** (`RedrawRequest.contextMargin`, 0–512 px, 0 by default): each
  tile is handed to the model inside a window of up to that many px of the
  picture as redrawn so far on every side, with a mask — white over the tile,
  black around it — so the model repaints the tile and keeps its
  surroundings; the tile is cut back out of the returned window, and the
  tiles run in order, each painted in before the next window is cut. The
  window is the tile's size plus the context on both sides, cut short at the
  picture's edges and trimmed to the model's multiple; tile and context
  together stay within 2048 px. For models that cannot take a reference
  (Krea 2): seeing only a close-up tile, they take skin for another body
  part (Notes).

## Shape

- `TiledJobs.runTiles(…, context = Some(TileContext(margin, sizeMultiple)))`:
  `windowOf` places the window, `maskOf` draws the mask, and each tile's
  request is built from a `TileInput(tile, window, image, mask)` — a redraw
  paints `window`'s size with `initImage` and `maskImage`; the returned window
  is kept as `tile-NN-window.png` with the tiles, the tile cut from it is the
  output. The in-order picture is the one an edit uses (39), finished with
  `TiledJobs.keepReturned`. Tested in `TileContextTests`.

- `RedrawRequest(runConfigurationId, strength = 0.4, instructions,
  negativePrompt, steps: Option[Int], seed = -1, runtimeId, contextSide = 768,
  tileSize = 1280, region: Option[ImageRegion] = None,
  minimumWindowSide = 1024, selectionMargin = 64, keepTiles = false)`;
  strength in (0, 1], tile 512–2048, reference 256–2048, window 256–4096,
  margin 0–512, a region at least 16 px a side and inside the image
  (`TiledJobs.regionOf`, `TiledJobs.checkArea`, shared with an edit, 39).
  `POST /api/outputs/{date}/{file}/redraw`.
- `shared/.../Tiling.scala`: `ImageRegion`, the tile layout and
  `Tiling.window` — integer geometry both sides share, so the browser lays out
  the very tiles the backend will run. `Tiling.Overlap` is the 256 px
  `TiledJobs.TileOverlap` forwards to. The image work stayed behind in
  `backend/.../postprocess/TileBlending` (`blend`, and `paste` for a region).
- A region redraws through `Tiling.window`: the box grown by the margin on
  every side, each side then at least `minimumWindowSide`, rounded out to 16
  and kept inside the image — shifted, not shrunk, at an edge. The tiles cover
  that window, `TiledJobs.runTiles` takes the blended result through its new
  `finish` step, and `TileBlending.paste` writes it back into the untouched
  source at full weight inside the box, ramping to nothing across the margin.
  The position sentence offsets each tile by the window's origin, so it still
  says where the tile sits in the whole picture.
- The box is **drawn, moved and resized**: a press on a corner resizes from
  it, a press inside moves it, a press outside draws a new one, and a click
  clears it. While it moves it says what it costs — `320×240 · 1 tile ·
  repaints 1024×1024` — because a tile is a whole pass of the model and that
  is the price of the job.
- Its sides **stick to the lengths where that price changes**:
  `Tiling.spanForTiles(n, tile, overlap) - 2 × margin` is the longest box that
  still fits in `n` tiles, and a drag within a handle's width of one lands on
  it instead of a pixel past (`ViewedImage.snap`, `RedrawGeometry`). Crossing
  costs a whole extra pass, so it takes a deliberate pull.
  `RedrawGeometry` is a `Var` the detail view owns: the redraw panel writes
  its tile size, margin and window into it as they are typed, and the viewer
  reads it — the picture and the panel agree because they read the same
  numbers.
- Frontend: `ViewedImage(width, height, selection)` in a `Var` the detail view
  owns — `GenerationMediaViewer` writes it (a drag in fractions of the
  picture, so a scaled-down view still names real pixels; every change of
  output clears the box), `RedrawPanel` reads it for the request and for the
  cost line (`RedrawPanel.costOf`: the area, the tiles, the steps each and,
  from the strength, how many of them are really sampled). The tile, window
  and margin fields, the grid controls, the plan and the geometry published to
  the picture are `TileAreaFields`, which the edit panel (39) holds its own of;
  only the task on screen publishes.
- `backend/.../postprocess/Redraw` (`Redraw.RestorationPrompt`,
  `PostProcessImages.position`), tiles through `TiledJobs.runTiles` with
  its configuration and tile request (26 has PiD on the same path): `SessionManager.startJobServer`
  launches the configuration's argv on a free port from the session range,
  never listed or counted as a session, output appended to the job log,
  stopped when the job ends and by `stopAll` at shutdown. Tiles are img_gen
  jobs through `sdserver/NativeJobs` over the server's capabilities defaults,
  every field written so a small tile cannot fall back to the server's
  default width.
- Tiling: the source padded to multiples of 16, `Tiling.layout` at most
  `tileSize` per tile, ≥256 px overlap, multiples of 16; a 4096² image at
  1280 is 16 tiles of 1216². The same seed for every tile. Absent steps and
  cfg come from the configuration's argv, so a turbo model keeps its 4 steps.
- `keepTiles` keeps `tile-NN-input.png`, `tile-NN-output.png` and
  `reference.png` in `logs/postprocess-<id>-tiles/`; the job log lists the
  server's img_gen features, the LoRAs applied (28) and every tile prompt.
- Configured LoRAs (28) are sent as `lora` on every tile; a missing one
  refuses the job.

## Notes

- **What each model does with a reference is a third thing, not a boolean**
  (`ReferenceImageUse`, measured 2026-09-19 on a purpose-made subject — a
  512×640 portrait generated for it, ESRGAN ×4, one 768 face tile, strength
  0.4, 20 steps):
  - `Context` — the reference informs without dictating. Flux.2, which redraw
    was built on.
  - `Edit` — the preset hands the reference to the diffusion model, which
    renders *that* image. Krea2 given the whole picture beside a tile returned
    mosaic corruption across the face; given the tile as its own reference it
    redrew cleanly, and given no reference at all it redrew cleanly and three
    times faster. So an `Edit` architecture is sent **no** reference, the
    tile-as-its-own-reference variant being a cost with no visible gain. The
    job note says which of the three cases it is — asked off, no preset at
    all, or an editing preset withheld — rather than reporting all three as
    "no reference preset in sd-cpp" (François, 2026-09-20, redrawing with
    Qwen Image 2.1).
    Classified there: krea2 (proven), and qwen-image, z-image-turbo, boogu ×2,
    mage-flow ×2 on the conservative side, unproven.
  - `Unused` — no preset at all: ernie-image, ideogram-4, hidream-o1,
    sensenova-u1.5.
  There is no way to soften a preset from drift: sd-server's API carries
  `ref_images` but no `ref_image_args`, so `pass_to_dit=false` cannot be asked
  for.
- **SenseNova U1.5 cannot be redrawn on at all**: it generates from a prompt
  normally, and returns pure RGB noise for any `--init-img`, in sd-cli as much
  as through drift (2026-09-19). It carries `initImage: false`, so the redraw
  panel refuses it by name instead of producing noise.
- **Every side is a multiple of the architecture's `sizeMultiple`** — the
  window, the padding it is run at, and each tile, on both sides of the wire:
  the panel counts tiles with the same number the backend lays them out with,
  and
  a tile size typed off the multiple is rounded up rather than refused. sd-cpp
  aligns an unaligned request *up* (`GenerationRequest::align_image_size`,
  `vae_scale_factor × diffusion_model_down_factor`) and answers with the bigger
  image, which `TiledJobs` then refuses as the wrong size — a redraw with Qwen
  Image 2.1 (multiple 32) died on "the image is 1280×1280, not 1280×1264"
  (François, 2026-09-20). PiD keeps its own numbers (64 for the layout, 16 for
  the window): a pixel decoder has no latent to align to.
- **The reference is a property of the architecture, not of the server.**
  sd-server answers `ref_images: true` for every model — `features_by_mode`
  is a hardcoded table (`make_img_gen_features_json`), computed from nothing —
  so the old gate on that feature refused nothing and told nobody anything.
  What decides is sd-cpp's `get_default_ref_image_preset(version)`: flux and
  flux2, qwen-image, krea2, z-image, boogu, mage-flow, longcat and anima have
  a preset; everything else falls back to "default", where a reference is
  latency that changes nothing. drift carries that as
  `Architecture.referenceImages`, seeded from the same list (23), beside
  `initImage` — the img2img the redraw itself is made of, and the one thing a
  redraw truly requires.
- What cannot redraw, then, is what the tags already said: video
  architectures, the LLM ones, and the PiD decoders — tagged `upscale`, and
  the only architectures that carry `initImage: false`, since a decoder
  reconstructs a reference rather than repainting an image handed to it. The
  flag is what a mis-tagged architecture would be refused by.
- So a redraw runs on **any** image architecture: with the whole image beside
  each tile where the family has a preset, and as plain img2img where it has
  none — the prompt then says the tile is part of "a larger photograph"
  instead of naming a reference image the model never received. `useReference`
  on the request overrides the architecture either way, and the *reference*
  group offers it as model default / always / never, hiding the reference size
  when none is sent.
- The trade, measured on Krea2 at 768 with strength 0.4, 8 steps
  (François, 2026-09-18): with the reference, composition holds far better
  (block luminance drift 26 against 41) but less texture is repainted (detail
  6.9 against 11.1) and the tile takes **2.8× longer** (152s against 55s).
  Worth it for the one or two tiles of a partial redraw, which is what the
  region is for; not obviously worth it for a whole 4K image.
- The negative prompt only acts when the configuration's cfg is above 1,
  which turbo models are not.
- The job never stops a live session: a job server loads its model beside
  whatever is resident, and the section's Stop offer is the only answer.
- Tile 1280 is the tested default (1536 shows overlap problems, 2048 risks
  the ROCm VAE crash seen with hires, 1024 makes a 4096² image 25 tiles).
- The reference's tokens join every step: about +56% image tokens at 768 on
  a 1024² tile, +25% at 512.
- Instructions stay a separate field so the result records them apart and a
  later step down the chain inherits no repair note.
- Seams: unlike PiD, tiles diverge as strength rises; the feathered overlap
  hides mild differences only. If seams persist, an overlap proportional to
  the tile is the next thing to try.
- A window is never scaled. Redrawing a 300×200 selection at 300×200 would
  come back as mush — models fall apart well below the size they were trained
  at — so the window is *grown* until the model has pixels enough, at the
  source's own scale. Nothing is resampled on the way in or out, and the
  detail density of a partial redraw is a full redraw's. Scaling a crop up
  instead would buy detail (ADetailer's `inpaint_width` does exactly that) at
  the price of a face rendered finer than the skin beside it; it is not what
  this does.
- 1024 is the default window because drift does not know any model's native
  size — a configuration's width and height are a hint, nothing more — and
  1024 is safe for the models it runs. With the 1280 tile, a face, a hand or a
  patch of background is one window, one tile, one inference.
- The margin does double duty: context for the model, and the width of the
  ramp `paste` blends over. A side where the window met the image's edge has
  no margin and keeps full weight — there is nothing beyond it to blend into.
- The reference is still the whole picture, which may tempt a model to repaint
  the whole scene into a small window; the position sentence is the defence.
  Dropping the reference under some window size is the fallback if it is not
  enough.

## Removing what an upscaler left behind

Measured on one 768 skin crop of a 4096 upscale, ERNIE Image (big-love fp8),
cfg 1, seed fixed, `PostProcessImages` detail at 1 px (grain, pores) and 4 px
(smudges, strokes) — source 4.24 / 8.12 (François, 2026-09-19):

| run | fine | coarse |
|---|---|---|
| strength 0.4, no soften | 4.74 | 9.50 |
| strength 0.4, soften 2 px | 1.45 | 5.21 |
| strength 0.6, soften 2 px | 1.60 | 6.18 |
| strength 0.8, 20 steps, soften 2 px | 3.29 | 11.19 |
| strength 0.8, 20 steps, no soften | 9.46 | 20.68 |

- An upscaler's leftovers are fine **structure**, and a low-strength img2img
  pass exists to preserve structure — so at 0.4 the model sharpens them: more
  detail than the source came in with, which is the complaint that started
  this. The prompt is not the lever it looks like.
- `softenRadius` blurs each tile before the model sees it, so there is nothing
  to sharpen back. Alone it is worse than nothing: at 0.4 and even 0.6 this
  model returns the blur. It pays only with the strength at 0.6–0.8 and
  16–20 steps, where the texture comes back new (3.29 fine, and coarse
  structure above the source's).
- Which is the recipe worth starting from for artifact removal — **strength
  0.55–0.8, 16–20 steps, soften 1–2 px on skin** — and the reason the region
  (partial redraw) matters: at that strength composition only holds because
  the area is small and feathered.
- The `redraw-skin-de-artifacting` template (32) names the artifacts
  concretely and describes skin as tissue rather than asking for "detail". It
  is a second built-in beside `redraw-restoration`, not a replacement: the
  restoration one stays the gentle default.
- **What an eager model does with a restoration prompt** — measured
  2026-09-22 on 1280² crops of a RealESRGAN ×4 of a 1024×1536 Qwen-Image 2.1
  beach portrait, strength 0.4, one seed, each model as François redraws with
  it. Krea 2 SNOFS + its turbo LoRA (8 steps, no reference) adds what the
  prompt names and more: the old default ("realistic skin texture with pores,
  fine hair and fabric detail") put freckles on the cheeks, moles on the belly
  and a dozen backlit hairs along the sunlit edge of an arm, none in the
  source. The hairs grow out of the upscaler's bumpy edge, which the model
  reads as backlit hair: no wording removes them ("no stray hairs" added
  some), soften 1 px brings them down to two or three faint ones — and flattens
  the skin texture with them, which no request for pores brings back; neither
  does a faint grain added after softening, nor strength 0.5 (the hairs
  return). Gentler prompts ("smooth, even tones, clear even skin") gave flat
  skin and a blotchy face where the upscaler's waxy patches were kept. Flux.2
  Klein 9B SNOFS (reference at 768, 1 sampled step of 4) and ERNIE Big Love
  (no reference, `--attn-scale` or the result is white) add no hairs to any
  prompt; they keep the bumps as small dents, and softening costs ERNIE its
  fine grain and warms its colour. So: `redraw-restoration` keeps its text
  minus "fine hair"; `redraw-women-portrait` asks for the same texture and
  says the arms, legs and body are "freshly waxed, bare and smooth" with "a
  clean, smooth highlight" where the light grazes the skin — with soften 1 px
  it was François's pick on Krea 2. Soften stays a per-job field: it helps an
  eager model and hurts a quiet one. Scripts and sheets were scratch work.
- **Context, measured 2026-09-22** on a ×2 zoom of the beach portrait's belly
  and arm (a 1024 tile covering a sixth of the body's width, as close as
  François's 8192 redraws), Krea 2 SNOFS with its turbo LoRA, strength 0.4,
  soften 1, the women's portrait text, two neighbouring tiles, one seed. The
  tile alone (≈130 s at 1280): the waist reshaped, a fold drawn across the
  belly, a red blotch on the arm — the "mess". A 1536 window, 256 px fixed
  around a 1024 tile (≈230 s): the silhouette where the source has it, the
  shading continued from the surroundings. A 1280 window, 128 px around 1024
  (≈145 s): the belly turned into a pair of knees — the view, not the fixed
  border, is what the model needs. The tile blanked white and inpainted from
  its surroundings (strength 1): a new scene, a face resting on knees. Flux.2
  Klein with the whole picture as reference kept the shape whether the tile's
  place was said in words, framed in red on the reference or blanked out on
  it (blanked moved the waist a little); it needs no context.
- **The reference keeps its shape.** sd-server before master-892 scales every
  reference to the tile's width × height while decoding the request
  (leejet/stable-diffusion.cpp#2004), so the whole picture sent beside a tile
  took the tile's shape — a portrait beside a square tile came out half again
  as wide (François, 2026-09-22). On such a build each tile's reference is
  letterboxed to the tile's shape with neutral grey, so the scaling is even; on
  a later one (`SdCppBuilds.keepsReferenceSize`) the request says
  `auto_resize_ref_image: false` and the reference keeps its own size. The job
  log says which.

## Post-v1

- A freehand mask rather than a box: sd-server's API takes `mask`/
  `mask_image` (1 channel, with an invert flag) and sd-cli takes `--mask`, so
  only the brush is missing.
- ADetailer (`sd-cli -M adetailer`) would find the regions a box is dragged
  around by hand. Assessed 2026-09-18 and not taken: it is CLI-only
  (`apply_adetailer` lives in `examples/cli/main.cpp`, and the server's
  routes and runtime carry no ad parameters), its detector is welded to its
  own inpaint pass (`adetail_image` takes the image and returns finished
  images — no detect-only call, no mask out), and its YOLOv8 detectors must be
  converted from Ultralytics `.pt` with upstream's Python script, or taken
  pre-converted from `exeterminal/adetailer-yolov8-safetensors`. Getting the
  boxes into drift needs a detect-only mode upstream, which François does not
  want to push for (2026-09-20): the lead was explored, not adopted. Closed.
- An assistant-written one-sentence summary of the source prompt if the
  identity sentence is not enough; the assistant writing the instructions.
