# 39 — Edit: change part of a picture, keep the rest

**Status:** done in code 2026-10-06 — compiles, `backend.test` green
(`EditCompositeTests`); the carried edit run end to end through an isolated
drift (Notes); not yet used by François
**Depends on:** 27, 32, 51

Redraw repairs; Edit changes. The user says what should be different — "the
shirt is dark red", "remove the shoulder bag" — over the whole image or a
selection, and drift makes the change while everything the instruction does
not touch stays the picture it was: same light, colour, grain and texture, no
seam where a tile or a selection ends.

## What it does

- A task in the gallery's **Redraw & upscale** tab: a configuration, the
  **instruction** (required), **carried up by** (a SeedVR2 upscaler, or *none
  — tile by tile*), a selection or none, and under *Advanced* the edit
  template, seed, steps, the tile size, window and margin that redraw has
  (27), and *keep tiles*.
- **Carried up** — the default as soon as a SeedVR2 configuration exists — the
  edit is made **once, where the model sees all of it**:
  1. the part to edit — the whole picture, or the window around the selection
     — is reduced to what the model takes in one pass: as it is up to 1536 px
     on its longest side, halved up to 3072, else brought within 1536
     (`EditRequest.passOf`);
  2. the model edits it in a single job: the reduced part is its only
     `ref_images` entry, no `init_image`, the prompt is the edit template
     followed by the instruction;
  3. drift reads what changed there (below) and cuts the box that holds it;
  4. the upscaler brings that box back up — ×2, or ×4 — in the tiles a SeedVR2
     upscale runs, each given its source's colours back;
  5. it is put into the picture through a mask made of both readings: the
     reduced one, enlarged and pulled in by its own margin, says *which object*
     changed, whole; the difference read at the picture's size says *where its
     edge is*.

  A shirt is then one decision, not nine tiles each deciding what a sleeve cut
  out of its shirt is; and what is new carries the upscaler's texture, the
  same as the rest of an upscaled picture.
- **Tile by tile** — no upscaler chosen — is what Edit was: each tile the
  model's image to edit, tiles in order, each cut from the picture as edited
  so far, composited, blended over the overlaps (27). Right for a change that
  sits inside one tile; across tiles the tiles disagree (Notes).
- **drift keeps what the model did not change** (`EditComposite`), against
  what the model was given:
  1. alignment — the edit is moved back by the whole-pixel shift at which its
     fine detail lines up best with the source's: an edit model hands the
     picture back a few pixels to the side;
  2. colour match — a per-channel gain and offset fitted on what the edit left
     calm (the calmest 60 %, never a pixel that moved by more than 8 %, and no
     fit when less than a fifth of the picture is left), then the slow drift
     that remains — a relit wall — is taken off: read where nothing changed,
     away from any real change, and applied inside one only as far as
     unchanged pixels near by vouch for it — a strap taken off a sleeve is
     filled at the sleeve's colour, the middle of a new sweater keeps the
     model's;
  3. change mask — a pixel changed if its 6 px mean moved by 16 % over a patch,
     or by 5 % while joined to such a patch; "moved" also counts a change of
     local contrast, so cream wool over a beige shirt is a change. Holes up to
     48 px are closed; the mask is grown by 6 px and feathered by as much;
  4. composite — the source where the mask is 0, the matched edit where it
     is 1.
- **A selection bounds the change, not what the model sees.** Carried up, the
  window around it is four times the selection's longest side, from 1536 to
  3072 px (`EditRequest.contextSide`): an edit model acts on a picture, not on
  a close-up, and the window costs one pass whatever its size. What comes back
  is kept inside the selection and across its margin only — not out to the
  window's edge.
- The job log names the pass's size, the share of it the edit changed (flagged
  above one half: a model that repaints lands there, and so does a large
  honest edit), the shift put back, and the tiles the upscaler ran.
- *Keep tiles* writes the pass as given, the model's raw edit, the mask, the
  composite and the mask it was carried through, in the job's tiles directory
  (27); tile by tile, each tile's input, raw edit, mask and composite.
- Only configurations of an architecture tagged **`edit`** are offered, and
  the endpoint refuses others by name. The upscaler must be a SeedVR2
  configuration on the drift runner, refused by name otherwise.
- The result is a new gallery entry derived from the source, `operation =
  "edit"`, carrying the instruction and the region.

## Shape

- `shared`: `EditRequest(runConfigurationId, instructions, templateId, steps,
  seed, runtimeId, tileSize, gridOffsetX/Y, region, minimumWindowSide,
  selectionMargin, keepTiles, upscaleConfigurationId)`;
  `EditRequest.passOf(width, height, multiple)` → `Pass(width, height, scale)`,
  which the panel prices a job with; `POST /api/outputs/{date}/{file}/edit`.
- `backend/.../postprocess/Edit`: checks the request, resolves the upscaler
  (`TiledJobs.resolved`), and starts either the tiled job (`TiledJobs
  .runTiles(…, finishTile = Edit.finishTile)`) or the carried one.
- `backend/.../postprocess/EditCarry`: the carried job, on its own thread, not
  a tiled job — one pass of the model and a few of the upscaler, so it is not
  paused. It asks a ready session of the configuration when there is one, else
  a server of the job's own, stopped before the upscaler's is started;
  `EditCarry.changedBox` is the box the upscaler is given (the mask's bounds,
  48 px around, on multiples of 16).
- `backend/.../postprocess/EditComposite`: `apply` (the composite above) and
  `carried` (the mask of a carried-up edit). Tested in `EditCompositeTests`.
- `TileBlending.paste(…, reach)`: the paste of a selection with its ramp kept
  within `reach` px of it; a redraw's still runs to the window's edge.
- Frontend: `pages/gallery/EditPanel` — model, *carried up by*, the *change*
  field, Advanced, and the cost line (`EditPanel.carriedCostOf`: the pass's
  size and the scale it is carried by).

## Notes

Measured the night of 2026-10-05 in
`~/dev/redraw-experiments/2026-10-05-edit` (`FINDINGS.md`, `sheets/`), on
generated 1k subjects upscaled ×4 — PiD in tiles, and SeedVR2 ×2 then ×2 —
with Flux.2 Klein 9B (SNOFS) on the drift runner, one seed unless said.

- **Tile by tile fails on the easiest garment edit.** "The shirt is dark red"
  over 3 × 3 tiles of 1024: the tiles that show the collar and the chest turn
  red, the ones that show a sleeve alone leave it beige, the middle one fades
  from one to the other, and later tiles re-change what earlier ones painted
  by 11 levels on average; 497 s. The same edit on the picture at a quarter
  of its size: the whole shirt, in 70 s.
- **The mask that read the colour alone broke the next case.** A cream
  sweater over a beige shirt moves the colour too little: half shirt, half
  sweater. It also called a relit wall an edit, and every edge of a picture
  the model had moved by 2–4 px.
- **Carried up by SeedVR2**, a sweater and a plaid flannel replace the shirt
  with their fibres and weave at 4k, the arm's own skin kept beside them:
  93–109 s for a 2560² window in the harness; in drift, 145 s for the same
  window (one pass at 1280², one tile ×2).
- **PiD is not the upscaler for this.** Klein's 4-step output is grainy before
  any upscale (where nothing changed: 4.0–5.3 levels of grain against the
  source's 1.7–2.3) and PiD turns the grain into crackle: a removed bag leaves
  a speckled sleeve and a wrinkled wall beside PiD's smooth original — "textures
  really too different from what was next to it". Taking the excess grain off
  before the upscale helps little. SeedVR2 restores what it is given and does
  not do it.
- **End to end in drift** (an isolated copy; pictures imported as JPEGs, Klein
  SNOFS, SeedVR2 7B): a 6144 × 4096 picture whole, "the dress has navy and
  white stripes instead of flowers" — one pass at 1536 × 1024, 15.9 % of it
  changed, a 736 × 704 box carried up ×4 in one SeedVR2 job: 185 s, a striped
  jersey with its knit and the arm it bared, the rest untouched. A selection
  larger than one pass (the man's shirt, light blue): 140 s. An upscaler that
  is not a SeedVR2: refused by name.
- **A close-up is not edited.** A 520 px box on a wrist, "remove the
  wristwatch": through the 1024 window a tiled edit uses, at the picture's own
  size, Klein hands the watch back on two wordings (one changed nothing, and
  the job says so; one moved a shadow). Through a 2080 window reduced to 1040,
  both wordings give a bare wrist with its hair and skin, 80–100 s. Hence the
  window four times the selection. A 1000 × 500 box over two earrings (seen
  through 3072², one pass at 1536²): both gone, the face the source's, 190 s.
- **RedQW21 does not edit**: given the picture as reference it draws another
  one (the whole pass changed). Klein is the edit model measured.
- **The same instruction gives the same edit on every seed**: eight
  instructions on three seeds at 1k, 21 right of 24 — sweater, plaid shirt,
  bag, lamp, sunglasses, watch, background woman, each 3 of 3 — and the three
  wrong are one sentence: "remove the woman's earrings" removed or replaced
  the woman every time. What fails is the wording, or a close-up, not the
  draw.
- **"Natural eyes" is a repair, not an edit**: Klein repairs the eyes and
  re-grades the whole picture (29 % of it changed). Spec 52's masked redraw is
  its tool.
- **Tile by tile with this composite**: the sweater over nine tiles goes from
  a patchwork with the shirt's buttons showing through to a whole sweater
  with two plain patches, in 511 s. The red shirt is not helped: a tile that
  shows a sleeve alone leaves it beige whatever reads the result.

## Open

- **A picture more than four times the pass** (an 8k one, whole) comes back
  ×4 and is enlarged the rest of the way: softer than its surroundings. A
  selection keeps the window small enough.
- **PiD as the upscaler**, for a picture upscaled by PiD: its crackle on an
  edit model's grain is the open point, not the wiring.
- **Several seeds**, and a choice among them before the upscale: the pass
  costs a minute, the upscale more.
- The carried job is not paused and has no picture on screen while it runs;
  the panel still draws the tile grid, which a carried edit does not use.
- **Fine structure over the edit** — strands of hair across a changed dress —
  keeps bits of the old picture between the strands: within the mask's margin
  the difference alone decides, pixel by pixel.
- **Wording.** "Remove the woman's earrings" removed the woman, on every seed;
  "remove the earrings" removes the earrings. The instruction is sent as
  written; the assistant could word it (32).
- **The smallest change the mask sees.** At the pass's size an earring is ten
  pixels wide and, replaced by hair, does not move a 6 px mean by 16 %: on a
  whole 4k picture the model removed both and the composite kept one, or
  none (the job then says so). A box around the thing makes the pass a window
  four times the box, where it is large enough. A finer reading finds it and
  brings back the false changes the mask was rebuilt to drop.
- The composite's constants come from one night's subjects.
