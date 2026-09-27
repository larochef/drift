# 39 — Edit: change part of a picture, keep the rest

**Status:** done in code 2026-09-22 — compiles, `backend.test` green
(`EditCompositeTests`), not yet run live
**Depends on:** 27, 32

Redraw repairs; Edit changes. The user says what should be different — "the
bikini top is red", "she wears a necklace" — over the whole image or a
selection, and drift makes the change while everything the instruction does
not touch stays the picture it was: same light, colour, grain and texture, no
seam where a tile or a selection ends.

## What it does

- A fourth task in the gallery's **Redraw & upscale** tab, beside Redraw, PiD
  and ESRGAN: a configuration, the **instruction** (required), a selection or
  none, and under *Advanced* the edit template, seed, steps, the tile size,
  window and margin that redraw has (27), and *keep tiles*.
- Each tile goes to the model as **the image to edit**: its only
  `ref_images` entry, no `init_image`, the tile's size, and a prompt of the
  edit template followed by the instruction. No framing sentence: there is one
  image, and it is the one to change.
- **Tiles run in order and are cut from the picture as edited so far.** A
  tile's overlap with the ones before it already carries their edit, so the
  model continues it rather than inventing its own. The whole picture is not
  sent as a second reference: Flux.2 Klein renders it into the tile (Notes).
- **drift keeps what the model did not change.** Per tile, against the crop it
  was given:
  1. colour match — a per-channel gain and offset, fitted by least squares on
     the pixels whose blurred difference is in the lowest 60 %, maps the edit
     onto the source's colours;
  2. change mask — the largest per-channel difference of the two images'
     6 px Gaussian means, ramped from 0 at 6 % to 1 at 16 %, grown by 4 px and
     feathered by 4 px;
  3. composite — the source where the mask is 0, the matched edit where it
     is 1.

  The skin, fabric and background the instruction did not mention are the
  source's own pixels, so their texture cannot change; what did change is
  brought to the source's colour balance. The composited tile is then blended
  into the picture over its overlap (27) and, for a selection, feathered back
  into the whole source as a redraw's is.
- The job log names each tile's **changed share** (the mask's mean). A share
  above one half is flagged in the log: a model that repaints instead of
  editing lands there (Krea 2: 75 %; Klein re-rendering the whole picture:
  77 %), but so can a large honest edit, so it does not fail the job.
- *Keep tiles* writes, per tile, the input, the model's raw edit, the mask and
  the composite, in the job's tiles directory (27).
- Only configurations of an architecture tagged **`edit`** are offered, and
  the endpoint refuses others by name: the tag already says "a model that
  edits an image", so it is the one question asked — no field of its own. The
  reference file tags `flux.2-klein-9B` (measured), `flux.2-klein-4B` and
  `flux.2-dev` (same family, not measured) besides the three it tagged before
  (`boogu-image-edit`, `mage-flow-edit-turbo`, `qwen-image-2.1`, not
  measured); `krea2` (measured: it draws another picture) and `ernie-image`
  (no reference preset) are not. `referenceImages` does not answer this: it
  describes a reference *beside* an init image, and Klein (`Context`) edits
  cleanly while Krea 2 (`Edit`) cannot.
- The result is a new gallery entry derived from the source, `operation =
  "edit"`, carrying the instruction and the region.

## Shape

- `shared`: `EditRequest(runConfigurationId, instructions, templateId, steps,
  seed, runtimeId, tileSize, gridOffsetX/Y, region, minimumWindowSide,
  selectionMargin, keepTiles)` — redraw's geometry, not its strength, soften,
  negative or reference; `POST /api/outputs/{date}/{file}/edit`
  (`editOutput`); `PromptKind.Edit` and the built-in `edit-seamless`, "drift
  seamless edit" (`PromptTemplate.DefaultEditId`); `Derivation.operation =
  "edit"` with `instructions` and `region`.
- `backend/.../postprocess/Edit`: checks the instruction, steps and area, and
  starts the job through `TiledJobs.tiledJob`, as Redraw does; each tile's
  img_gen request is the template and the instruction, the crop as its only
  `ref_images` entry, the tile's size, the seed, the configuration's LoRAs,
  the server's sample and VAE-tiling defaults with the request's steps.
  `Edit.finishTile` runs `EditComposite` and words the log line (changed share,
  flagged above `Edit.RepaintShare` = ½).
- `TiledJobs.runTiles(…, finishTile = Some(…))` is the edit path: a copy of the
  reference is the running picture, each tile is cut from it, judged, finished
  (`TiledJobs.FinishedTile(image, note, kept)`, the note optional) and painted in with
  `TileBlending.paint` — the ramps `blend` uses, over the overlaps with the
  tile before it in the row and the row above — and the picture is the result.
  Scale 1 only. The helpers redraw and edit share moved to `TiledJobs`:
  `checkArea`, `regionOf`, `templateTextOf`, `tiledArea` (the window, the
  padded reference, the rows).
- `backend/.../postprocess/EditComposite`: the colour match, mask and composite
  above, on float channels (Gaussian means as three box blurs, the growth a
  separable maximum). Tested in `EditCompositeTests` with `TileBlending.paint`.
- Frontend: `pages/gallery/EditPanel` — model, the *change* field on screen,
  then Advanced (template, steps and seed, area, keep tiles) and the cost line
  (`EditPanel.costOf`: every step sampled). The area fields, grid controls,
  plan and geometry are `TileAreaFields`, shared with `RedrawPanel`; both
  panels stay built, and only the task on screen publishes its geometry
  (`active`). `PostProcessSection.EditTask`, `TiledTasks` (where a box may be
  drawn); the configurations offered are those of sd-cpp architectures tagged
  `edit`; `PostProcessService.Command.Edit`; "Edit" in the job list and the
  recorded parameters ("Edited with … : instruction").

## Notes

Measured 2026-09-22 on 1280² and 1024² crops of a RealESRGAN ×4 of a
1024×1536 Qwen-Image 2.1 beach portrait, one seed, instruction "the bikini top
is bright red instead of light blue", each model as François runs it.

- **img2img cannot edit seamlessly.** At strength 0.6 Krea 2 SNOFS, ERNIE Big
  Love and Flux.2 Klein 9B SNOFS all ignore the instruction. At 0.8 Krea 2 and
  ERNIE turn only the straps and trim red and relight the skin (hard new
  shadows, new skin texture, a satin sheen); Klein with the whole picture as
  context paints the whole picture into the tile.
- **Klein with the tile as its only reference edits cleanly** — the whole top
  red, pose, shadows and framing kept — but re-renders the untouched skin
  greyer and olive with its own grain. The composite takes all of that back
  to the source (36 % of the tile changed: the bikini and a few hair strands).
- **Krea 2 is not an edit model**, whatever `referenceImages` says: given the
  tile as reference it draws a different composition covered in a mosaic
  pattern, in 384 s.
- **Across two overlapping tiles**, edited independently the halves already
  matched after the composite and the blend. Cutting the second from the
  picture as edited so far lowered its changed share from 24 % to 19 % (the
  overlap was already red) with an invisible join. Sending the whole picture
  so far as a second reference, before or after the tile, made Klein render
  that picture into both tiles (77 % and 40 % changed). A colour change is an
  easy case; an edit that invents detail (a pattern, an object) is where
  distant tiles could disagree, and the overlap is all that ties them.
- The composite thresholds (6 %/16 %, 6 px means, 4 px growth and feather) come
  from this one subject; they are constants until another subject shows them
  wrong.
