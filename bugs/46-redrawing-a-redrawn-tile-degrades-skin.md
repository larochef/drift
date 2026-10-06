# Bug 46 — Redrawing a part that was already redrawn makes the skin texture much worse

**Status:** open (reported by François 2026-10-05; not reproduced, not investigated — recorded for later, and
to be met again in the edit study of the night of 2026-10-05)
**Severity:** medium (a second pass is how a seam or a small defect gets repaired: it should not cost the
texture around it)
**Files:** `backend/src/drift/backend/postprocess/Redraw.scala`, `TiledArea.scala`, `TileBlending.scala`
(`paste`), `PostProcessImages.colourMatched`; the redraw's defaults (strength, steps, the 3×3 reference)

## Symptom

A picture upscaled, then redrawn; the redraw left the line between two tiles badly handled. He selected that
part and redrew it again, with Qwen Image 2.1: the skin texture came out "really worse" — "a bit like what I
had with FLUX.2". His reading: "something is still broken with re-repainting".

His case, step by step, as he gave it afterwards — everything with **Qwen Image 2.1**, at the default strength
(0.4, he thinks):

1. **SeedVR2** upscale, 1k → 4k (the ×4 that runs as ×2 then ×2). It left lines where its tiles meet
   (`bugs/42`).
2. A **full redraw** of the 4k picture. It removed some of those lines, not all.
3. A **Redraw of a selection** — one tile — (the Redraw task, not Edit: confirmed) to remove a line that was
   left. **This is the pass that made the skin much worse.**
4. Set aside; he then upscaled the full redraw of step 2 with SeedVR2 to 8k², and is very pleased with it.

So the title holds: the damaged part had already been redrawn once, and the third model pass over the same
skin (SeedVR2, Qwen, Qwen) is the one that broke it. Not recorded: the selection's size.

## Measured 2026-10-06 (`~/dev/redraw-experiments/2026-10-05-edit`, `FINDINGS.md` §8, `sheets/seam.jpg`)

His chain on a generated subject: a chest of a SeedVR2 4k with a line through it (U), a full redraw by Qwen
Image 2.1 at 0.4 (R1), then a 1024 × 256 selection across the line redrawn through a 1024 window, ten ways.

- **No redraw removes the line**: it is in U, in R1 and after every selection pass — 0.15 to 0.4, and under the
  selection's mask at 0.4 and 0.9. Redraw is not the tool for a seam.
- **The full redraw is what costs the skin**: U has freckles and pores, R1 neither (grain 2.46 → 1.98); passes
  made from U keep them at 0.25, lose most at 0.4. On a SeedVR2 picture a 0.4 redraw replaces a better texture.
- **The paste**: as drift pastes, the skin around the selection moves by 1.1–2.2 levels and loses grain; with the
  ramp kept within 64 px, 0.2–0.3 and none. `TileBlending.paste` takes a `reach` now (the carried edit uses it);
  the redraw's paste is unchanged — to decide.
- **What removes the line** (`sheets/seam-reupscale.jpg`, `seam-reupscale-1k.jpg`): not the 2k window upscaled
  again — the line is already in the 2k picture, drawn by the first ×2 pass (bug 40) — but **the same window taken
  from the 1k original and upscaled ×2 then ×2**: the line is gone, the freckles and pores are the picture's, and
  the selection pasted within 64 px shows no join (44 s). A shorter, fainter line appears elsewhere in the window:
  the boundary that draws them moved with the crop.

So the repair for a line an upscale left is not a redraw: it is **"upscale this selection again"** — the window
around the selection cut from the picture the upscale was made from (its derivation's parent), upscaled by the
same model, the selection kept. Not built. And a full redraw of a SeedVR2 picture is to weigh against what it
erases.

## Leads

- **The second pass works on the first one's output, not on a photograph.** A redraw's skin is the model's own
  texture; the restoration template still says "remove the upscaling artifacts": the model treats its own
  grain as the artifact and smooths or re-invents it. Each pass moves further from the source — a redraw is
  not idempotent, and nothing in drift measures that.
- **The same family as Klein's skin** (`specs/52`, measured 2026-10-05): a mottling of fixed period no lever
  removed, laid down by the model. A second pass lays its pattern over the first one's.
- **A selection is redrawn differently from the whole picture.** A window of at least `minimumWindowSide`
  around the selection, cut where the selection is rather than on the grid of the first pass; its 3×3
  reference is built from the picture as it now is (already redrawn); the fade back runs from the selection to
  the window's edge, wider than the selection (noted 2026-10-05). So the second pass also lays a second grid
  across the first, and the seam it was meant to repair sits in the middle of a window — the right place — but
  the skin on both sides is repainted with it.
- **The colour match** takes low frequencies from the picture given, here the first redraw's.
- **Strength and steps** of a partial redraw are the form's, the same as for a whole picture: a repair of a
  seam may want far less (0.2?) than 0.4, or a mask that holds the skin and repaints the line alone
  (`maskSelection`, runner only, Qwen Image 2.1 and FLUX.2).

## To try

- **His case first:** a generated subject upscaled by SeedVR2 ×2 then ×2, redrawn whole by Qwen Image 2.1 at
  0.4, then a selection across a seam left on skin redrawn again at 0.4 — against the same at 0.15–0.25, and
  under a mask a few pixels wide along the line. A seam is a low-contrast line: it wants a narrow, weak repaint, not a window of new skin.

- Generated subject, upscaled, redrawn once; then a selection across a tile seam on skin redrawn a second and
  a third time, at 0.2 / 0.3 / 0.4, with and without the mask — texture measured against the first pass and
  against the upscale (grain, the period of any pattern), judged alone and pasted back.
- The second pass cut from the *upscale* rather than from the first redraw (repaint from the source again,
  keep only the selection): does the seam go without the texture paying for it.
- The same for an edit made twice on the same part (the edit study: several edits one after the other on one
  thing, and what the texture is after each).

## Idea: the assistant on a selection (François, same message)

"Maybe we could also have the assistant's help on the partial tiles selection." Auto redraw (`specs/52`) reads
the whole picture with the job's tiles drawn on it, and a selection is redrawn without the reading. Wanted: the
assistant shown the selection's window — at its own scale, which is also the close-up 52 leaves open for what
exists only at full size — to say what is wrong there (a seam, a texture) and set that pass: its prompt, its
strength, a mask or not.
