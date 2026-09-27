# 45 — Redraw: full steps at any strength, a reference at the tile's scale

**Status:** planned
**Depends on:** 27 (redraw), 42 (the drift runner), 43 (a runner per configuration)

Two changes make redraw finish what it starts, whatever the source:

- **Steps.** A redraw runs the number of steps the configuration (or the
  redraw panel) asks for, whatever the strength. Strength only picks how much
  noise the source gets. Today, strength also cuts the steps, and on turbo
  models this leaves a redraw unfinished.
- **Reference.** Every tile is painted beside a reference at a scale close to
  its own. Redraw is offered only on models that take a reference as
  context. Today the whole source is squeezed into the reference, so on a
  large source (above 4096 px) the tile's surroundings shrink to a few blurred
  pixels.

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

### Shape (to be settled)

Two ways; pick one after the check below.

- **A. Inflate the steps (both runners, no runner change).** drift sends
  `⌈N / strength⌉` steps, so the floor runs N (4 at 0.4 → 10 sent, 4 run).
  - It works on sd-cpp too, whose sampling drift can't change.
  - The σ values are the tail of the longer schedule, with its shift, so the
    steps are spaced the way the model's own schedule spaces them.
  - It lives in `Redraw.scala`, where `sampleSteps` is set per tile.
  - Round so exactly N run: check `⌊⌈N / s⌉ × s⌋ = N` and step up if not.
- **B. Schedule the range (runner only).** `FlowSchedule` builds N steps over
  [σ(strength), 0] instead of skipping the first ones. This is Automatic1111's
  "do exactly the amount of steps the slider specifies". It is exact, but it
  exists on the runner only, so sd-cpp keeps today's behaviour.

A keeps the shifted mapping, so "0.4" still starts near σ 0.86. B can
start at σ = strength itself, which is what the slider suggests; then 0.4
really is 40% noise, and the redraw panel's default must be re-tuned (the
current 0.4 was picked while it meant ~0.8). Preference: B on the runner,
with A as sd-cpp's fallback; to be confirmed by the check.

Whether sd-cpp maps strength the same way is to be read from its code, not
assumed.

### The check before building

A step-distilled model (Klein, the Flux.2 turbo LoRA) was trained on its
own σ grid; both A and B step through σ values off that grid. Compare one
Klein redraw at strength 0.4:

- today's single step (from σ 0.77);
- 4 steps over the same range;
- 4 steps from σ 0.4 and from σ 0.6 (B's meaning of strength).

Use a purpose-built source image, per the usual test-subject rule. If the
4-step version isn't better, Part 1 is dropped.

## Part 2 — a reference the model can use

### The problem

Every tile gets the whole source as `ref_images`, fitted within the
reference size (768 by default, 2048 at most), beside the tile's window as
`init_image` (spec 27). The ratio between the two grows with the source:

| Source side | Reference | A 1024 tile in the reference |
|---|---|---|
| 2048 | 768 | 384 px |
| 4096 | 768 | 192 px |
| 8192 | 768 | 96 px |

- **The reference loses its use above ~4096 px.** It tells where the tile
  is and who is in the picture, but at 1/5 scale or less the tile's own
  surroundings are a smear. It is a blurred copy of what the model is asked
  to sharpen, and it can pull the tile toward that blur.
- **Models without a context reference get none** (`Edit` architectures
  and those with no preset: Qwen Image, Krea 2, Z-Image turbo, …). On those,
  a tile of a large source is a close-up with no idea of the whole. This is
  where skin gets taken for another body part (spec 27, Notes).

### What it does

- **Redraw is offered only on configurations whose architecture takes a
  reference as context** (`ReferenceImageUse.Context`: today Flux.2 dev,
  Klein 4B and 9B). The redraw panel's configuration picker lists only
  those, and the endpoint refuses the others with the reason.
- **The reference is a neighbourhood, not the whole picture, once the
  source is large.** Each tile's reference is a crop of the source centred
  on the tile, a few tiles wide (for example 3× the tile side), fitted within
  the reference size. That keeps the tile at no less than about a third of
  its own scale in the reference, whatever the source size.
  - At or below the size where the whole source already fits at that ratio,
    the reference stays the whole picture, as today.
  - The position sentence (`Redraw.framing`) then places the tile in the
    neighbourhood, not in the whole picture.
- **Tile + context + reference fit the model.** The window (tile plus
  context, ≤ 2048 px today) and the reference together stay within what the
  architecture handles in one pass. That budget is a property of the
  architecture, not a global 2048.

- **Neighbour tiles as references, as many as the model takes.** An
  architecture declares how many reference images it accepts
  (`maxReferenceImages`). A tile's references are filled in order of use
  until the count or the budget runs out:
  1. the whole picture, small: identity and layout;
  2. the neighbourhood crop: where the tile sits;
  3. the neighbour tiles at their own scale, **already redrawn** ones first.
     Tiles run in order, so the tiles above and to the left are finished when
     a tile starts. Handing them over lets the model match their texture,
     grain and skin. That is the seam and consistency problem context margin
     handles today by inpainting, but with the whole neighbouring tile
     instead of a strip.
  - Every reference costs tokens and time. On a flow transformer attention
    grows with the square of the tokens, so eight full tiles beside one tile
    is a 9× longer sequence. The pixel budget governs, not the count alone.
    References may be downscaled to fit it (neighbour tiles at half scale
    still carry texture).

### Shape (to be settled)

- `Architecture.maxReferenceImages`: the vendor's number (Qwen Image 2.1:
  9; the Flux.2 families' from BFL's documentation), read from the docs, not
  measured. No default: every seeded architecture declares its own, 0 where
  no reference is sent. It is useful beyond redraw: the generation form's
  reference images and Edit can cap on it too.
- The filter is `architecture.referenceImages == Context`. `useReference =
  false` stays possible for a job, since it's a trade the user makes, but
  the model list no longer includes models that can never take a reference.
- Neighbourhood crop: in `Redraw.scala`, beside `fitWithin` / `letterboxed`;
  one reference per tile instead of one per job. The `keepTiles` output then
  keeps one `reference-<n>.png` per tile.
- The budget per architecture: a field on the architecture (a pixel count for
  window + reference), seeded for the three Flux.2 families; to be measured,
  not guessed.

### Open questions

- **The cost of the restriction.** Krea 2 and the other non-context models
  lose redraw.
  - Alternative: keep them, but only below a source size where their
    reference-less tiles still work.
- **Qwen Image 2.1 is the first candidate to add.** It takes up to 9
  reference images, so the whole picture and the neighbourhood both fit.
  - It is classed `Edit` on the conservative side, unproven (spec 27, Notes).
    The fear is Krea 2's failure, where a reference handed to the diffusion
    model gets painted into the tile.
  - Test it as a context reference: one face tile, with whole + neighbourhood
    references, on sd-cpp and on the runner. The runner decides for itself
    how references enter the model (its Qwen Image 2.1 editing path), so the
    two runners may answer differently.
  - If it passes, its class becomes `Context` (on that runner, if only one
    passes) and the filter includes it.
- **Neighbourhood or whole picture, or both** (two references: the whole
  picture small for identity and layout, the neighbourhood for detail). Flux.2
  and Qwen Image 2.1 (up to 9) take several references; each costs tokens and
  time.
- **Which references earn their cost.** Measure one face tile of a large
  source with: whole only; whole + neighbourhood; whole + neighbourhood +
  redrawn neighbours. Compare quality, seams and time per tile. A neighbour
  tile that isn't redrawn yet carries the upscaling artifacts the job is
  removing, so it may do more harm than good.
- **Measure before settling the ratio.** Same source above 4096, one face
  tile: whole-picture reference versus neighbourhood reference versus none.

## Open questions (both parts)

- Does Part 1's rule apply to the generation form's img2img, and to Edit
  (39) when it runs as img2img? Redraw first; the others after the check.
- Keep today's step behaviour as an option? Only if the check shows a case
  where one step is better.
