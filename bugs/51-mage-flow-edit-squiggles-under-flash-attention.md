# Bug 51 — Mage-Flow Edit Turbo draws squiggles all over its edits under flash attention

**Status:** fixed in drift 2026-10-07 (the architecture no longer carries `--diffusion-fa`, migration 13); the
fault itself is sd-cpp's on ROCm and is not reported upstream yet
**Severity:** high for the model (every edit was unusable), none after the fix but the speed
**Files:** `backend/resources/reference/architectures.json` (`mage-flow-edit-turbo`),
`backend/resources/migrations/013-edit-models.json`

## What happened

François's edit with Mage-Flow Edit Turbo (2026-10-06, `mageflow.png`): the picture is edited and covered with
curved shards, on the background as much as on the subject. Reproduced on five cases of the edit study
(sd-cpp master-929, ROCm, gfx1151): the instruction is carried out — the sweater is there — and 56 to 86 % of
the picture counts as changed because of the lines.

One thing changed at a time (`~/dev/redraw-experiments/2026-10-06-edit/s3_mage.py`, `s6_mage_fix.py`,
`sheets/mage-nofa-vs-fa.jpg`):

| setting | result | a pass at 1024 × 1536 |
|---|---|---|
| `--diffusion-fa` (the seeded default) | squiggles | 79 s |
| no flash attention | clean: 12.8 % changed, the sweater alone, no shift | 163 s |
| `--diffusion-fa --attn-scale 0.0078125` (the cure for ERNIE's white pictures) | squiggles | 75 s |
| the stock Qwen3-VL-4B instead of the abliterated one | squiggles | 74 s |
| 8 steps | squiggles | 144 s |

The text-to-image Mage-Flow Turbo is clean under flash attention (1024², 23 s): the fault needs the reference
image's tokens.

## Fix

`mage-flow-edit-turbo` runs without `--diffusion-fa`: twice the time, and a real edit. To report upstream
with the two pictures; the drift runner would not have the fault (Mage-Flow is not on it yet).
