# Bug 47 — Edit offers models that do not edit, and says nothing when one repaints

**Status:** fixed 2026-10-07 (found by François 2026-10-06 on his first use of the rebuilt Edit). The list:
migration 13 takes the `edit` tag from `qwen-image-2.1` and `boogu-image-edit`. The silence: an edit whose model
changed more than half of what it was shown says so on its gallery entry (`Derivation.warning`: "✎ edited ⚠"
on the card, a Warning row in the details). Measured the night of 2026-10-06 — see the end of this file.
**Severity:** high (the first model in his list for the job he wanted repaints the picture: every try looks like
Edit not working)
**Files:** `backend/resources/reference/architectures.json` (the `edit` tag),
`frontend/src/drift/frontend/pages/gallery/GenerationDetailHost.scala` (the configurations offered),
`backend/src/drift/backend/postprocess/Edit.scala` and `EditCarry.scala` (the share logged, `Edit.RepaintShare`),
`shared/src/drift/shared/PostProcess.scala` (`PostProcessJob`: nothing carries a warning)

## What happened

François, on an 8192 × 8192 picture (SeedVR2), "remove the line between the breasts" — a line the upscale's
tiling left — with **RedQW21** ("Qwen 2.1 Turbo, red edition"), three ways, 08:05 to 08:07:

| job | route | what the log says |
|---|---|---|
| `g1791266501908-1001` | selection 1953 × 613, carried up by SeedVR2 7B: a 3072² window, one pass at 1536², ×2 | "the edit changed 83.5 % of the pass; the model had moved the picture, put back by -3,7 px — more than half: the model may have repainted rather than edited" |
| `g1791266768156-1002` | whole picture, none — tile by tile: 64 tiles | tile 1: "changed 40 % of the tile, moved by the model and put back by -5,2 px" (stopped) |
| `g1791266812832-1003` | selection 1152 × 477, none: one 1280 × 1024 tile | "changed 37 % of the tile, moved … by -7,8 px" |

His words: "the line was still there and the image didn't blend in well". He read the third as "one tile, so the
case where the upscaler does not really work". It was not the route: on all three the model redrew what it was
shown, and drift pasted a repaint. The line of the log that says so is not shown anywhere he looks.

## Two faults

1. **The list.** Edit offers every configuration of an architecture tagged `edit`: `boogu-image-edit`,
   `flux.2-dev`, `flux.2-klein-4B`, `flux.2-klein-9B`, `mage-flow-edit-turbo`, `qwen-image-2.1`. Measured the
   night of 2026-10-05 (`~/dev/redraw-experiments/2026-10-05-edit/FINDINGS.md` §5, §7, §14):
   - `flux.2-klein-9B` (SNOFS): edits — 21 of 24 over three seeds, the misses one badly worded sentence;
   - `qwen-image-2.1`, RedQW21: **does not edit** — the whole pass changed on every case (a text-to-image
     fine-tune: given a reference it draws another picture);
   - `qwen-image-2.1`, the official weights with the turbo LoRA: edits a portrait picture (3 of 3) and redraws a
     landscape one at another scale (90 % changed, 3 of 3) — not reliable;
   - `flux.2-klein-4B`, `flux.2-dev`, `boogu-image-edit`, `mage-flow-edit-turbo`: not measured on this route.

   François: "maybe we should also only have in the list of models for edits the ones that should be used… in the
   same way that the list of models for redraw is now small."
2. **The silence.** A job whose model repainted completes like any other. The share and the "more than half"
   line are in the job's log only.

## To decide, then do

- **Which architectures keep `edit`.** By what is measured: `flux.2-klein-9B` yes; `qwen-image-2.1` no. The four
  others are unmeasured: measure them (one pass at 1k on the night's cases is a minute each) before keeping or
  dropping them, rather than guess. The tag is the architecture's, and RedQW21 against the official weights
  shows the answer can differ between two checkpoints of one architecture: dropping the tag from
  `qwen-image-2.1` also drops the official weights — the simple rule, and the one that matches "only the ones
  that should be used". (A reseed migration, as redraw's restriction was — migration 010.)
- **Say it on the job.** When more than half of the pass changed: the job shows it — a warning on the job in
  the gallery, with the share — instead of a line in a log. Whether such a result is still pasted: a large
  honest edit also passes one half (a sweater over a window that is mostly shirt: 93 %), so warn, do not refuse.
- **The line itself is not Edit's job** (bugs 42 and 46): no redraw removes it, an edit model leaves a mark that
  faint alone, and the composite does not see it. What removed it in the harness is the window around it
  upscaled again from the picture the upscale was made from. On his picture — 8k, upscaled from a 4k that was
  itself a full redraw — that is a re-upscale of the 4k window, if the line is not already in the 4k: to check
  on the first try.

## Measured the night of 2026-10-06 (`~/dev/redraw-experiments/2026-10-06-edit`, FINDINGS.md)

| architecture | on the 1k cases | a pass | kept `edit` |
|---|---|---|---|
| `flux.2-klein-9B` | edits (2026-10-05: 21 of 24) | 55–100 s | yes |
| `flux.2-klein-4B` | not measured: no checkpoint here; the 9B's family | — | yes |
| `flux.2-dev` (Q4_K_M, turbo LoRA, 8 steps) | edits: the sweater, 13.2 % changed | **520 s** | yes — slow, not wrong |
| `mage-flow-edit-turbo` | edits once flash attention is off (bugs/51): 12.8 % changed, no shift | 163 s | yes |
| `boogu-image-edit` (the full Edit checkpoint at the seeded 4 steps) | a blurred repaint, 39 % changed | 439 s | **no** |
| `qwen-image-2.1` | RedQW21 repaints; the official weights redraw landscape pictures (2026-10-05) | 110 s | **no** |

Flux.2 dev and Boogu were stopped after one case each, for their cost.
