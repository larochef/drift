# 45 — Redraw: full steps at any strength, a 3×3 reference, a part that blends in

**Status:** done in code 2026-10-03 (compiled, backend tests green, not run live in drift)
**Depends on:** 27 (redraw), 42 (the drift runner), 43 (a runner per configuration)

Three changes make redraw finish what it starts, whatever the source:

- **Steps.** A redraw runs the number of steps the configuration (or the
  redraw panel) asks for, whatever the strength. Strength only picks how much
  noise the source gets. Today, strength also cuts the steps, and on turbo
  models this leaves a redraw unfinished.
- **Reference.** Every tile is painted beside the 3×3 block of tiles around
  it, and redraw is offered only on models that take a reference as context
  (Qwen Image 2.1 joins them). Today the whole source is
  squeezed into the reference, and on a large source a tile is a few dozen
  pixels in it: measured, that is no better than no reference at all — sand
  is repainted as rock, an areola is erased, a face changes.
- **A part that blends in.** A redrawn part keeps the source's colour and
  shading and only brings its detail, so a box redrawn alone no longer shows
  its edge. Today its colour drifts and the edge shows.

All three were measured on 2026-10-03 (Measurements, below) on a
purpose-made subject, through François's chain (PiD); the files are in
`~/dev/redraw-experiments/2026-10-03/`.

## Part 1 — steps as asked

### The problem

img2img runs only the tail of the schedule: `⌊steps × strength⌋` steps
(diffusers' `get_timesteps`; `FlowSchedule.firstStep` in the runner; sd-cpp
uses the same floor). At redraw's default strength of 0.4:

| Configuration | Steps | Run |
|---|---|---|
| Flux.2 dev + turbo LoRA | 8 | 3 |
| Flux.2 [klein] | 4 | 1 |

- **Too few steps.** A single step is one Euler jump from σ≈0.4 straight to
  the clean image. The result reads as unfinished: strange textures on
  Klein redraws, seen by François.
- **Strength is coarse.** At 4 steps, strength has four positions: 0.25–0.49
  runs 1 step, 0.5–0.74 runs 2, 0.75–0.99 runs 3. Moving the slider inside a
  band changes nothing.
- **Strength is not the noise level.** The start noise is the schedule's σ
  at the first step run, and FLUX.2's shift packs the schedule near 1. On a
  1024² tile (4096 tokens), strength 0.4 starts at:

  | Steps | First step run | σ there |
  |---|---|---|
  | 4 | 3 | 0.77 |
  | 8 | 5 | 0.85 |
  | 10 | 6 | 0.86 |

  So a "0.4" redraw on Flux.2 repaints from mostly noise. On 4-step Klein it
  is a single jump from σ 0.77 to 0, which likely explains the strange
  textures more than the step count alone. Other shifted schedules (Qwen
  Image 2.1) have the same effect, to a different degree.
- **The wrong fixes.** A minimum step count or a higher default strength
  would repaint more of the source than asked. They trade fidelity for steps.

### What it does

- **Steps as asked.** A redraw at N steps runs N steps, from the noise level
  that strength selects down to 0.
- **Strength is continuous.** Every strength value gives its own starting
  noise level.
- **The progress bar shows N steps.**
- **Cost follows the steps.** The redraw panel's estimate (`costOf`) counts
  N steps per tile, not N × strength. On default settings a redraw costs
  more than before.

### What sd-cpp does

Read from its source at `2bb7294` (2026-09-22); nothing was run.

- **The same floor.** `t_enc = ⌊sample_steps × strength⌋`, and the schedule
  kept is the last `t_enc` steps of the full one
  (`src/pipeline/image.cpp`, `prepare_image_generation_latents`). The init
  latent is noised at the first kept sigma (`diffusion_engine.cpp`,
  `noise_scaling(sigmas[0], …)`): for a flow model,
  `latent × (1 − σ) + noise × σ`.
- **Strength 1 trims nothing.** The trimming runs only below 1. Just below
  1, at most N − 1 steps run.
- **`strength_as_noise_level=true`** (a key of `extra_sample_args`, #1738):
  strength is the noise level to start from, which is σ itself on a flow
  model. The start is the first sigma of the N-step schedule at or below it,
  so fewer than N steps still run. With `force_first_sigma=true` beside it,
  that first sigma is replaced by the target, so 0.4 starts at σ 0.4. Neither
  key is in the `--extra-sample-args` help or in `api.md`.
- **The hires fix already inflates the steps** (`request.cpp`, "sd-webui
  behavior"): `scheduler_steps = ⌊steps / strength⌋`, then the same trim.
  img2img does not.
- **sd-server takes a schedule per request.** The native route reads
  `sample_params.custom_sigmas` and `sample_params.extra_sample_args`. A
  request with N + 1 sigmas over [σ_start, 0] and `strength: 1.0` is not
  trimmed, noises the init latent at σ_start and runs N steps. Custom sigmas
  replace sd-cpp's own schedule, per-model shift included.

### Shape

**Inflate the steps (option A).** drift sends `⌈N / strength⌉` steps, so
the floor runs N (4 at 0.4 → 10 sent, 4 run), on both engines with no
runner change — what sd-cpp already does for its hires fix.

- The σ values are the tail of the longer schedule, with its shift, so the
  steps are spaced the way the model's own schedule spaces them, and "0.4"
  keeps its meaning (a start near σ 0.86 on FLUX.2).
- `Redraw.scheduledSteps(N, strength)`: the fewest steps whose floor at
  that strength, in single precision as sd-cpp and the runner compute it,
  is N. N is the request's steps, else the configuration's under its LoRAs'
  sampling (49); a custom sigma list sets its own steps and is left alone.
- The job log says "N steps run, K scheduled"; the panel's cost line counts
  N per tile.

Not taken: **option B**, N steps over [σ(strength), 0] sent as
`custom_sigmas` with `strength: 1.0`. sd-cpp runs it as it is and it came
close behind A, but drift would need every family's shift, the runner's
`firstStep` would have to stop trimming a custom schedule, and the 0.4
default would have to be re-tuned.

### The check (done 2026-10-03)

Klein 9B (SNOFS) on sd-cpp, strength 0.4, the 3×3 reference, three tiles:
a face at the 4k stage, sand at the 4k stage, an areola at 16k. sd-cpp's log
confirms what ran (`target t_enc`): 4 steps sent run 1; 10 sent run 4 (A);
`custom_sigmas` with `strength: 1.0` run the 4 levels sent (B). Fine detail
of a 400² crop of the face (Laplacian σ): input 1.64, today 1.67, A 2.29,
B from σ 0.4 1.86, from 0.6 2.08, from 0.77 2.14.

- **One step does nothing visible**: the output is the input. Four steps
  repaint skin and freckles finer. Part 1 is worth building.
- **A started higher (about σ 0.86) and gave slightly more detail than B
  from 0.77**; B from 0.4 was gentle. The step-distilled model was no worse
  for stepping off its grid. A is chosen for its simplicity, not that margin.
- On the 16k areola, no step count at 0.4 removes the upscaler's banding
  (see Part 2, Sharpness).
- Not checked: the runner (its half of the run failed in the harness), and
  Qwen Image 2.1 or Flux.2 dev with a turbo LoRA.

## Part 2 — the 3×3 neighbourhood as reference

### The problem

Every tile gets the whole source as `ref_images`, fitted within the
reference size (768 by default), beside the tile as `init_image` (spec 27).
On a 16k source a 1280 tile is about 60 px in that reference. Models with no
context reference (`Edit` and `Unused` architectures) get nothing at all.

Measured on 16k-scale tiles (Measurements), a tile that knows nothing of its
surroundings is repainted as something else. Through PiD, RedQW21 (Qwen
Image 2.1, today's drift: no reference) erased an areola entirely, leaving
smooth skin and the nipple — what François saw on his 16k redraws — and
turned sand into rock. Klein made the face older and its skin rough. The
whole picture at 768 did no better: the areola faded, the sand was rock.
The tile is too small in it to say anything. Through ESRGAN, sand became
skin on every model.

### What it does

- **One reference per tile: the 3×3 block of tiles around it**, the tile
  at its centre, cut from the source and fitted within the reference size.
  With 1280 tiles and drift's 256 px overlap the block is 3328 px a side,
  so the tile is about a third of the reference whatever the source size.
  - At an edge or a corner the block is shifted to stay inside the picture,
    so the tile is no longer at its centre there.
  - A source no larger than the block is its own block: the reference is
    the whole picture, as today.
  - The framing sentence says the tile is at the centre of the reference
    and the reference shows it with its surroundings; the percentages of
    the whole picture go.
- **The whole picture is no longer sent.** Not measured beside the block
  (one reference was enough to fix every failure seen); see Open questions.
- **Redraw is offered only on configurations whose architecture takes a
  reference as context** (`ReferenceImageUse.Context`). François accepts
  fewer models for a better result (2026-10-03). The redraw panel's
  configuration picker lists only those; the endpoint refuses the others
  with the reason.
- **Qwen Image 2.1 becomes `Context`.** On the drift runner and on sd-cpp
  alike it used the block (or the whole picture) as context and never
  painted it into the tile; it had been classed `Edit` on the conservative
  side, unproven (27). Migration 010 reseeds it.
- **The reference size stays 768.** It was not really compared: the
  runner's Qwen Image 2.1 scales every reference to about a megapixel
  (`QwenImage21Pipeline.reference`), so 768 and 1280 reached the model
  alike, with the same result and time. Not measured on Klein or Flux.2 dev,
  where the size is sent as is.

### What was not taken

- **Per-tile captions.** With the block as reference they changed nothing
  visible on Qwen Image 2.1, hand-written or automatic. The automatic ones
  (Qwen3-VL 8B on the block with the tile framed) are also unreliable: the
  16k sand was "rough, textured skin on an animal's body", and on the tile
  alone the areola was "human skin and hair". A model that needs a caption is
  a model without a reference, and those are the ones this spec stops
  offering.
- **Neighbour tiles, already redrawn, as more references**, and with them
  `maxReferenceImages` and a token budget per architecture. One reference
  fixed every failure seen; the context margin (27) already hands the model
  the redrawn surroundings at the tile's scale.

### Sharpness

- **After PiD, a 16k redraw adds no sharpness.** PiD's tiles already carry
  skin texture, an areola's grain and individual sand grains. No variant of
  the redraw was sharper; RedQW21 with the block stays closest to the input
  and slightly smooths the finest skin texture. What a redraw at that stage
  is worth, against its hours, is open.
- **ESRGAN is where the softness came from in the first measurements**: it
  leaves smooth areas as flat bands of colour that img2img keeps at any
  strength. Grain added to the tile broke them there; on PiD tiles grain
  only added noise and smoothed the areola. Not taken: drift's chain is PiD.

### A broken base

The official Qwen Image 2.1 base had sand drawn as a woven grid at 1k; every
upscale and redraw carried it up. At the 4k stage, RedQW21 with the block
repainted it as sand at strength 0.9, not at 0.7. So a base that is wrong
is repaired by a region redraw at the 4k stage at high strength, where the
model still sees the scene, not by 16k tiles.

### Shape

- `Redraw.neighbourhood(tile, size)`: three tiles a side less the two
  overlaps, centred on the tile (or its context window), shifted to stay in
  the picture and no larger than it. Each request's reference is that block
  of the source, fitted within `contextSide`; the position sentence places
  the tile in the block. The whole picture is not kept on disk any more.
- `Architecture.referenceImages` stays one class for both engines: no model
  measured differs between them. The redraw endpoint refuses an
  architecture that does not take a reference as context, with the reason;
  the panel's picker lists only those.
- `useReference = false` stays possible on a job (a trade the user makes),
  on a model that takes a reference.

## Part 3 — a redrawn part that blends in

### The problem

A redraw of one part (a box, 27) pasted back into the untouched picture
often shows its edge, so François redrew whole pictures instead. Measured on
a single 1280 window at the 16k stage, pasted as drift does (full weight in
the box, a ramp across the 64 px margin): every raw redraw shifts the box's
colour, by 3 to 28 levels of 255 on average (blur 24 px, the box against
the source). On skin that is a visible edge: RedQW21 with no reference came
out yellower and smoother, both models at strength 0.9 greener or darker,
Klein with no reference with bluish blotches. The 3×3 reference reduces the
shift but does not remove it.

### What it does

- **The redrawn window keeps the source's colour and shading**: before the
  paste, its low frequencies are replaced by the source's —
  `redrawn + blur(source) − blur(redrawn)`, blur radius 24 px — so only the
  detail the model drew is kept. Measured: colour shift below 1 level
  (2.7–4.5 at strength 0.9), detail unchanged, and no edge left at 1:1 on
  either model, at 0.7 or 0.9.
- It applies to every redrawn tile, partial or full (a full redraw's tiles
  drift in colour the same way, which the feathered overlaps hide only
  partly). Always on, with no switch.
- **Not to Edit (39)**: an edit changes colour on purpose ("the bikini top
  red"), and taking the source's colour back would undo it.
- It is image work in drift, after the model, so it works on any engine.

### What it does not fix

- **A different texture.** Edit (Klein, the tile as its only reference)
  draws a fresh tile: much rougher, more detailed skin than its
  surroundings. Corrected, its colour matches, but the texture density still
  shows as an edge. That is the "good alone, does not blend in" result
  François remembers. Texture follows strength: keep a part's redraw at the
  strength its surroundings were redrawn at.
- **Not taken: noise anchored to the picture.** Each latent position's
  starting noise drawn from (seed, its place in the whole picture) instead
  of from the window's corner, so a part redrawn again sees the noise the
  full redraw used there (runner only: sd-server's API takes no noise).
  Measured on Qwen Image 2.1, a 3×3 full redraw of 16k skin, then a box
  re-redrawn off the grid at the same strength: the second pass changed the
  fine texture by 1.24 levels with anchored noise against 1.34 with free
  noise, and nothing differed by eye — in both, the box came out a little
  more textured, a second pass over redrawn skin adding texture whatever
  its noise. The lever is the strength of the second pass, not its noise.

### Shape

- `PostProcessImages.colourMatched(redrawn, source)`: two box passes of
  `ColourRadius` (29 px, about the Gaussian of 24 px measured) per image.
- `TiledJobs.runTiles(…, correctTile)`: each returned tile goes through it
  against the source's crop of the tile as it was before `prepareTile`
  (soften), before it is painted in or blended. Redraw passes
  `colourMatched`; PiD and Edit pass nothing.
- `keepTiles` keeps the uncorrected tile as `tile-NN-output-raw.png`.

## Measurements (2026-10-03)

### Through PiD (François's chain)

Base: RedQW21 + its turbo LoRA at 10 steps (François's configuration),
1024×1536, the same beach scene, its sand clean at 1:1. PiD 1.5 (FLUX.2,
the runner, 4 steps) to the 4k stage around three targets (face, chest,
sand); for three more (eye, areola, sand), a 1280 window there redrawn
with RedQW21 (no reference, as drift does today, strength 0.5, 10 steps
run) and PiD again to the 16k stage. 1280² tiles, strength 0.7, seed 42.

| Model (runner) | No reference | Whole picture | 3×3 block | s per tile: none → reference |
|---|---|---|---|---|
| RedQW21 + LoRA (10 steps run) | areola erased, sand → rock | areola faded, sand → rock | both kept | 88 → 107 |
| Flux.2 Klein 9B (8 run) | face older, rough skin; areola a little faded | face changed less | all kept | 73 → 129 |

- Eye and chest tiles held with every variant on RedQW21.
- Grain on the 16k tiles with the block: noise added, the areola smoother.

### Through ESRGAN (first, overnight)

The subject: official Qwen Image 2.1 (25 steps, cfg 6), 1024×1536, a topless
woman sitting on a beach (sand with footprints, rocks, sea, a pine, green
eyes). Its sand already shows a woven grid at 1k, before any upscale or
redraw: a redraw keeps it, being built to keep structure. François's chain
was approximated around six targets, with ESRGAN where he uses PiD: ESRGAN ×4 to the
4k stage (face, chest, sand tiles), then a 1280 window there redrawn with
RedQW21 (strength 0.5) and ESRGAN ×4 again for the 16k stage (eye, areola,
sand tiles). Every job went straight to sd-server or the drift runner, 1280²
tiles, seed 42, one seed only. Models: RedQW21 (Qwen Image 2.1, François's
redraw model), Flux.2 Klein 9B SNOFS, Flux.2 dev Q4, Krea 2 SNOFS + turbo
LoRA, all on the drift runner unless said.

| Model (runner) | Uses a reference as context | 3×3 fixes sand / areola | s per tile: none → one 768 reference |
|---|---|---|---|
| Qwen Image 2.1 (RedQW21) | yes, never painted in | sand yes; areola kept even without | 76 → 94 |
| Flux.2 Klein 9B | yes | yes / yes | 52 → 87 |
| Flux.2 dev | yes | yes / yes | 234 → 407 |
| Krea 2 | refused (runner: no reference images) | — | 88 (none) |

- Without a reference, or with the whole picture at 16k, sand became skin
  on all three models that ran, and Klein / Flux.2 dev faded the areola.
- On the 4k face, no reference changed the person (RedQW21, strength 0.7).
- With the block, the hand-written caption, an automatic caption, a
  restoration prompt naming no material, and the block at 1280 all gave the
  same picture on RedQW21.
- `--cache-mode spectrum` is ignored by the runner (outputs byte-identical):
  it is not the softness.
- RedQW21 on sd-cpp (through PiD, 16k areola and sand): no reference
  shrank the areola and turned the sand to clumps of rock, the whole picture
  smoothed the areola and left the sand rocky, the block kept both; nothing
  was painted in. 192 → 300 s per tile with a reference.
- Images, sheets and `results.csv` (one row per job: model, tile,
  reference, caption, strength, steps sent and run, seconds, a detail
  number) are in `~/dev/redraw-experiments/2026-10-03/`; `FINDINGS.md`
  there is the night's log.

## Open questions

- **The colour radius**: 24 px was the only value tried, on 1280 windows.
- **Is the 16k redraw worth it after PiD?** Compare a 16k picture straight
  from PiD with the same redrawn with the block, at strengths 0.3–0.7.
- **The whole picture beside the block**, for identity on a tile that shows
  no face. Not measured: one reference fixed every failure seen.
- **Two seeds** on the deciding comparisons (block versus none, sand and
  areola); everything here is one seed.
- Does Part 1 apply to the generation form's img2img and to Edit (39)?
  Redraw first.
