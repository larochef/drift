# 52 — Auto redraw: the assistant reads the picture and sets the redraw

**Status:** done in code 2026-10-05 (compiled, backend and runner tests green); measured in the experiment's harness and run once end to end through an isolated copy of drift — the reading, a redraw with it, the refusal, a masked repair (Measured, below); not yet used by François
**Depends on:** 20 (assistant), 27 (redraw), 32 (prompt library), 45 (redraw steps and reference), 11 of 42 (vision on the runner)

A redraw asks its user for a prompt, a strength and which part of the picture
to repaint. Every picture is different, and every part of a picture: sky,
skin, sand, a face. **Auto** hands the choice to the assistant: one call, on
the whole picture scaled down with the redraw's own tiles drawn on it. It
answers, tile by tile, with what the tile shows and how hard to repaint it,
and lists what looks broken. The form shows it, every value stays editable,
and the redraw starts on the user's click.

## What it does

- **🤖 Read the picture**, in the redraw panel, asks the first running
  assistant whose model reads images. One call, whatever the picture's size,
  never one per tile.
- With none running, the card offers to start one (`VisionAssistants`,
  `VisionAssistantStarter`): a chat configuration that reads images —
  `Architecture.readsImages`, a `--mmproj` slot with a file assigned — the
  last one run first, launched through the gallery's own launch-or-download;
  the reading is asked once it serves. A live chat model that does not read
  images is replaced only by the one button that names both. The assistant
  started stays loaded; its session's memory warning is shown in the card.
- The answer gives **each tile its own prompt and strength**:
  - the prompt names the materials the tile shows and the fine detail they
    should have ("skin with pores and fine hairs", "coarse sand with small
    pebbles"). It follows the restoration template and comes before the
    instructions written for the whole job;
  - the strength is 0 for a tile to leave as it is (an even sky, a blurred
    background), 0.2–0.3 for a face, 0.4–0.5 where detail should be added. A
    tile at 0 runs no job: it comes back as the source has it.
- It also lists **repairs**: parts a viewer notices as wrong (eyes that glow,
  a misshapen hand, garbled lettering), each with a box and what it should
  look like instead. **A repair is a proposal, never a default**: it changes
  what the picture shows, and that may be what the picture is about
  (François: "I might want to have a picture with glowing eyes"). **Set up
  this repair** selects the box and sets the form for it — strength 0.9, the
  fix as instructions, the selection repainted alone under a mask, the
  source's colours not put back — and the user starts it as a redraw of the
  selection.
- The prompt the picture was generated from goes with the question, when
  drift made the picture: what it asks for is not a defect.
- Nothing runs unseen: a line sums the reading up, **Tiles** lists every tile
  with its strength and prompt, editable; **use it for this redraw** turns it
  off without losing it; **Ask again** replaces it.
- A reading belongs to the tiles it was made for. A job whose tiles are cut
  elsewhere — another tile size, a moved grid — is refused, by name, rather
  than painted with a neighbour's prompt. A selection is redrawn without it.
- The result records that its tiles were set by the assistant
  (`Derivation.planned`).

## Shape

- **The call.** `POST /api/outputs/{date}/{file}/redraw-plan`
  (`RedrawPlanner`): the picture within 1536 px, a thin line down the middle
  of each overlap between tiles and each tile's name (A1, B1, …) in its
  corner; the tiles are `TiledArea.of`'s, the ones the job will cut. Asked
  whole, not streamed, **without thinking** (`chat_template_kwargs`) at
  temperature 0.2, through `AssistantProxy.ask`.
- **The question** is a template of its own kind in the library
  (`RedrawDiagnosis`, built-in `redraw-diagnosis`), editable like the others.
- **The answer** is JSON: `kind`, `cells` (name → holds, prompt, strength) and
  `repairs` (defect, fix, box in fractions of the picture). It is read
  leniently (`RedrawPlan.parse`): the last JSON object of the reply; a tile
  left out keeps strength 0.4 and no prompt; a strength out of 0–0.6 is
  brought back in; a repair without a usable box is dropped. Each of those is
  a note the form shows. Only an answer with no JSON object is refused.
- **The job.** `RedrawRequest.tiles` carries each tile's settings by its
  place; `Redraw` matches them to its layout or refuses. Steps are scheduled
  per tile, so every asked step runs at any strength (45).
- **A repair.** `RedrawRequest.maskSelection` sends, with a selection, a mask
  of it (grown by half the margin, softened): the model repaints the
  selection and keeps the rest of the window. `matchColour: false` leaves the
  tile's colours as painted. The runner takes masks on Qwen Image 2.1 and
  FLUX.2 (`Inpainting`): after every step the kept part of the latent is put
  back at that step's noise level.
- **Memory.** The assistant stays loaded beside the redraw's server; the call
  is made before the job starts.

## Measured

`~/dev/redraw-experiments/2026-10-04-auto/` (`FINDINGS.md`), on purpose-made
subjects upscaled by SeedVR2 7B at ×2 twice; the assistant is Qwen 3.8 27B on
the runner, the redraw RedQW21 with its turbo LoRA, one seed unless said.

- **The restoration template does not matter.** Another template with the
  same seed changes a tile by 0.5–1.5 levels; another seed with the same
  template by 5.8–6.5. Auto does not choose one.
- **Thinking off is what makes the call usable.** With thinking, 13 of 42
  answers came back empty (the tokens went into the thinking) and a call took
  26–120 s; without it 42 of 42 parse.
- **A question that quotes the templates is answered by reciting them**
  ("waxy, plastic, smeared pores") whatever the picture shows, and its verdict
  flips with the seed and the size. Asking for broken parts first, then for
  what each cell holds, gives the same answer on every ask.
- **Defects seen at the picture's scale are found**: glowing eyes on 5 asks
  of 5 with a box within two hundredths of the picture each time, garbled
  lettering on a street; nothing invented on four clean pictures.
- **What exists only at full size is not seen**: an upscaler's hatch on
  skin, strands across a small face. A picture scaled to 1536 px shows a
  tile as 256.
- **The grid is read correctly**: every cell listed on 20 answers of 21 at
  1536 (28 cells for 24 at 1024, hence 1536), contents right against the
  picture, an even sky at 0, about 145 s for 24 cells.
- **A repair needs a high strength and its instruction, and nothing less
  works**: glowing eyes are still glowing at 0.4 and 0.7, and at 0.9 without
  the instruction; at 0.9 with "natural human eyes…" they are natural.
- **The colour match undoes a colour repair**: it takes the low frequencies
  from the source, so the glow returned as a halo. And at 0.9 without a mask
  the whole window moves and darkens, so the pasted box shows its edge —
  hence the mask and the colours left as painted.

- **On thirteen subjects, six of them people at different sizes and in
  different positions**: 15 answers of 15 with every cell, 109–136 s for 24
  tiles; the glowing eyes proposed, nothing invented on the twelve others.
  The generation's prompt does its work: with "green eyes" in it the fix is
  "natural green irises", with "glowing neon green eyes" no repair is
  proposed. A street's garbled lettering, found by a question about defects
  alone, is not listed by this one.
- **End to end in drift** (an isolated copy, a 2048 × 3072 picture, 6 tiles):
  the reading came back in 26 s, every tile with its contents, a prompt and a
  strength, and the glowing eyes as a repair with its box; the redraw with it
  ran its 6 tiles in 341 s (345 s for the default one); the same settings
  with another tile size were refused by name. The repair, set up as the
  button does: natural eyes and the face around them untouched in 50 s —
  against a green halo over the eyes and a doubled mouth when repainted the
  old way, colour-matched and without a mask.
- **Each tile's own prompt changes little at 0.4**: the two redraws of that
  picture differ by 2.3 levels, less than a seed does. What the reading buys
  is the strength per tile, the tiles left alone, and the repairs.

## Open

- **Close-ups.** Defects that exist only at full size need a second look at
  the cells that hold a face, a hand or a nipple, sent at their own scale: a
  second call, on a few crops chosen by the first. Not built; it departs from
  "one call" and is François's to decide.
- **Large pictures.** The reading is one answer for every tile. François's
  first 8192² reading (2026-10-06; 64 tiles of 1248 px sharing 256) came back
  in 98 s with every tile, its contents, and strengths from 0.2 to 0.4 — not
  checked tile by tile against the picture. That is a second and a half a
  tile; the harness's 24 tiles took 145 s with another assistant, six seconds
  a tile. A 16384² picture is 289 tiles: eight minutes to half an hour. The
  browser, the server and the call to the assistant all wait an hour
  (`RedrawPlan.ReadingMinutes`): the browser's request gave up after one
  minute before, as "the operation was aborted", and the server would have
  after five. At 289 tiles a cell is 90 px wide: whether the grid is still
  read right there is not measured. Reading the picture in blocks of tiles,
  one call a block, each block at its own scale, is the way out if it is not
  — and it departs from "one call".
- **What a reading costs.** The assistant's own log of the 8192² reading
  (Qwen 3.8 27B, 2026-10-06) splits its 98 s in two: **13 s to read** — 2931
  tokens of picture and question, the same whatever the picture's size, since
  it is always shown within 1536 px — and **84 s to answer** — 2798 tokens,
  44 a tile, at 33.4 tokens a second. So about 13 s + 1.3 s a tile on that
  assistant: some six and a half minutes for the 289 tiles of a 16384²
  picture, if the speed holds on a longer answer. The harness's 24 tiles in
  109–145 s were the same model answering at 13.4 tokens a second (measured
  that night on the rebuilt runner); why it was two and a half times slower
  then is not explained. Each reading should record its tiles, the tokens
  read and written and both times (llama.cpp reports them), so the panel can
  say how long a reading will take before it starts.
- **The reading as a task** (53): asked for and fetched instead of one request
  held open for minutes — what the hour of waiting above stands in for.
- **Several repairs in one job**, and a repair followed by the whole-picture
  pass without the user chaining them.
- **Edit** (39) and **Upscale** are out of scope.
