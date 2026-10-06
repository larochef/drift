# Gallery

The **Gallery** page is everything drift ever made, rebuilt from the files on
disk, so it needs no running session and survives restarts.

An empty gallery explains what lands there and offers **Create a project**
(or **Just try a model**). While none of your results is in a project, a line
above them suggests starting one, with the same button.

## Browsing

- A grid grouped by day, newest first. One tile per image; a batch's images
  sit side by side with a "▦ 2/4" badge.
- Filters: project, **Show NSFW projects** (off by default), run configuration,
  images or videos, and a search over prompts. With nothing filtered, older
  days load with **Show n generations** or **Show all remaining days**. As
  soon as a filter may hide something — hidden NSFW projects count — the
  whole gallery is read and a day shows only for what it has that matches,
  with that count; a day with nothing to show does not appear.
- A generation completing anywhere appears here by itself.
- **☑ Select** switches the grid to selection mode: tick tiles, or a whole day
  from its heading, and delete them behind one confirmation. Deleting removes
  the files and everything recorded about them.

## Importing images

**⤓ Import images** in the toolbar, or dropping files anywhere on the gallery,
brings in PNG or JPEG images from your computer so you can upscale, redraw or
edit them. Each lands under today with an "⤓ imported" badge and its original
file name; one picked image opens straight in the detail view. Imports belong
to no project and have no parameters to reuse. Other formats (WebP, GIF…) are
refused — convert them to PNG first.

## The detail view

Click a tile to open it:

- The image or video player, with the thumbnails of the batch under it.
- Every recorded parameter, including the input images. Each image of a
  batch has its own seed — the first image's plus its place in the batch — and
  the seed shown is the one of the image on screen.
- **Made from** / **Made from this** links for derived entries, and an
  **Original** / **Result** toggle at matched zoom on an upscaled or redrawn
  image. **Inputs from** lists the gallery entries a generation's inputs were
  picked from (init image, references…), and such an entry lists the
  generation under **Made from this**. Only inputs picked with **From the
  gallery** are linked; a file from the disk is not.
- **🤖 Ask the assistant**, **🗑 Delete**, and the reuse buttons.

## Reusing parameters

- **↺ Reuse these parameters** into a live session of the same run
  configuration reproduces the request exactly, seed included. From an image
  of a batch it reproduces that image: its own seed, a batch of one.
- On another configuration it carries the *task*: prompt, negative prompt,
  input images, seed, size and the sampling fields the form shows, over the
  target's own defaults. Recorded LoRAs are left out, since a LoRA suits a
  model, not a task.
- With no session running, **▶ Launch … and reuse these parameters** launches
  a configuration with the task.

## Post-processing

The **Redraw & upscale** tab holds three tasks — **Redraw**, **Edit**,
**Upscale** — one on screen at a time, picked with the buttons at the top of
the tab. A video has one of them, **Upscale**, with the models that take a
video. Each makes a better or bigger copy **beside** the original: results
are new gallery entries linked to their source; nothing runs inside your
session, though a live one is offered a **■ Stop it** button to free memory
first.

Every task reads the same way, and reads short: a line saying what it does,
the **model** to do it with, an **▸ Advanced** section — folded, the way an
architecture's LoRAs are — holding every other parameter, and last of all what
the job will be and the button that starts it.
Nothing of a task sits after its own button, and Advanced stays open once you
open it, keeping what you typed while you move between tasks and images. The
tiled tasks — redraw, edit, and a SeedVR2 or PiD upscale — put the two
switches for the grid drawn over the picture beside that last line rather
than in the form: they change nothing about the job.

- **⬆ Upscale** is one task whatever enlarges the picture: its **model**
  select lists every upscaler you have — the SeedVR2 and PiD run
  configurations and the ESRGAN models of your upscaler store — and what
  follows the select is what the chosen model takes.
- An **ESRGAN** model (see [model-cache.md](model-cache.md)): seconds, no new
  detail; *Advanced* holds how many passes it runs. RealESRGAN x4plus works;
  x2plus does not load in sd-cpp.
- A **SeedVR2** model, for a picture **or a video**: it restores and enlarges
  in one diffusion step a pass, ×4 by default or ×2, on the drift runner (it
  does not run on sd-cpp). A ×4 is made as two passes of ×2, in about the time
  one pass of ×4 took: asked for ×4 at once, the model hatched skin and
  foliage and drew dark strands across small faces. The built-in **SeedVR2 7B upscale** is the default,
  **SeedVR2 3B upscale** is lighter and a little faster; *Advanced* holds the
  seed. There is no prompt. A picture larger than one pass (about 1088 px a
  side at ×4) is cut in tiles like a PiD's — the line before the button says
  how many, **show the grid** draws them, *Advanced* holds the **tile** size
  (see the tile grid below), and the job can be paused and
  resumed; the target stops at 16384 px on the longest side. A video keeps its
  frame rate and its soundtrack and comes back as a new video entry.
  **Original** / **Result** compares the two. A picture's colours
  are matched to its source; a video's are not yet.
- A **PiD** model: a diffusion decoder re-renders the image at ×4, sharper
  than ESRGAN on generated images. Pick a run configuration on a *PiD*
  architecture (Flux.2, Flux.1 or Qwen-Image VAE variants; use the
  `pid_1.5_…_4step_bf16` decoder files, and the Gemma 2 `tokenizer.json` in
  the *tokenizer* slot). *Advanced* holds the target size, steps, seed and the
  prompt — the source's by default; leave the size empty for ×4 of the source
  with its ratio kept, capped at 16384 px on the longest side. The line before
  the button says what that comes to — `→ ×4 of the source · 4096 × 4096` — so
  the size is never a surprise. The result takes the source's colours back,
  tile by tile: PiD on its own drifts — an illustration came out paler, its
  white paper greyed. **A target no larger than the source is
  refused** and the button greys out: PiD decodes a *quarter* of the target, so
  such a job would shrink the picture to that quarter and hand the same size
  back. That also means a source above a quarter of the cap is downscaled
  before the decode — a 8192² source asked for ×4 gets 16384², decoded from a
  4096² copy of itself, so it is a re-render at twice the size rather than a
  true ×4. Large targets are decoded in overlapping tiles and blended; the job
  reports "tile n of m". The model loads once per job — not
  at all if a session of that configuration is already running.
  - **On sd-cpp** it needs master-892 or newer (update the runtime if the job
    is refused), and tiles are at most 1536² (nine for a 4096² target).
  - **On drift's runner** (the *drift runner, images* runtime as the default)
    a 4096² target is a single pass: the whole image decoded at once, with no
    seams. It takes about 5 minutes for 1024 → 4096. Larger targets are cut
    into 4096² tiles. The tile grid shown before the job follows the default
    runtime; *Advanced* holds the **tile** size, and the grid can be moved
    (see the tile grid below).
- **✨ Redraw**: a low-strength img2img pass, tile by tile, that repaints
  texture (skin, hair, fabric) while the composition stays. Pick an image
  configuration, a strength (0.4 by default), and optionally *instructions*
  describing what should be seen ("clear even skin", "crisp fabric weave").
  Each tile is painted beside the block of tiles around it, so the model
  knows what it is looking at, and comes back with the picture's own colour,
  so it blends in — a box redrawn on its own included.

  **🤖 Read the picture** (Auto redraw) hands the choices to the assistant.
  With an assistant running whose model reads images, drift sends it the
  whole picture once, scaled down, with the job's tiles drawn and named on
  it — never tile by tile, whatever the picture's size. The answer sets,
  for every tile, a prompt naming the materials it shows ("skin with pores
  and fine hairs", "coarse sand with small pebbles") and its own strength:
  0 for a tile to leave as it is (an even sky), 0.2–0.3 for a face, 0.4–0.5
  where detail should be added. It takes two to three minutes for 24 tiles.

  With no such assistant running, the line under the button offers to start
  one: the chat configuration with a vision projector you ran last (a select
  when you have several), and **▶ Start … and read the picture** — the
  picture is read as soon as the model has loaded, so it is still one click.
  If a chat model that does not read images is running, the button says so
  and replaces it: **⏹ Stop …, start … and read the picture**. Nothing is
  stopped or started beside another without that click. The assistant stays
  loaded afterwards, for the next picture; stop it from the Sandbox or its
  project when you want its memory back.
  The line beside the button sums it up, **Tiles** lists every tile with
  its strength and prompt, all editable, and nothing starts until you press
  **✨ Redraw**; untick **use it for this redraw** to go back to one prompt
  and one strength. The reading belongs to the tiles as they were cut: after
  changing the tile size or moving the grid, **Ask again** — the job refuses
  settings made for other tiles. It is of the whole picture, so it is not
  used while a box is selected.

  The assistant also lists what looks **broken** — eyes that glow, a
  misshapen hand, garbled lettering — each with what it should look like
  instead. These are proposals only, because repairing one changes what the
  picture shows and it may be what you wanted. **Set up this repair** selects
  that part and sets the form for it: strength 0.9, the fix as instructions,
  **repaint the selection alone** (the model paints it under a mask and
  keeps everything around it), and **keep the source's colours** off (they
  would paint a glow back around the repaired eyes). Then **✨ Redraw
  selection**; clearing the selection puts the form back.

  The model is all the form shows; everything else is under *Advanced*,
  grouped by what it decides:
  *prompt* (a restoration template from Settings → Prompts, with your
  instructions in a text box under it) and *negative*, a text box too — a
  restoration prompt is prose, and a one-line field showed a sliver of it;
  *pass* (strength, steps, seed, soften); *reference* (whether the block of
  tiles goes along, and at what size, and the *context* around each tile); *area* (tile size 1280, window and margin —
  below); *options* (keeping the source's colours, repainting a selection alone
  under a mask, keeping the tiles on disk for inspection).
- **Which restoration prompt.** Some models add whatever detail the prompt
  names, and more: Krea 2 given "pores, fine hair" sprinkled freckles, moles
  and stray hairs over smooth skin; Flux.2 Klein and ERNIE barely react. The
  default, *drift restoration*, asks for skin and fabric texture but no longer
  for fine hair. *drift women's portrait* keeps the texture and says the skin
  is freshly waxed, bare and smooth, with a clean highlight where the light
  grazes it; with an eager model such as Krea 2, set *soften* to 1 px as well —
  the model reads the upscaler's bumpy edges as backlit hairs, and softening
  takes them away before it sees them (on a quiet model it only flattens the
  texture). *drift skin de-artifacting* describes skin as tissue (pores, vellus
  hair, moles): keep it for the stubborn waxy leftovers below.
- What drift says to the model: where the tile sits in its reference, and
  that the reference is context only; with the reference switched off, that
  the image is a close crop and to repaint only what is already there.
  Your *instructions* are added after that, and work best as descriptions of
  what should be seen ("natural skin texture, visible pores") rather than
  orders ("redraw the chest") — an order invites the model to draw that thing
  afresh instead of repainting what the crop holds.
- **Upscaling artifacts** (waxy skin, doubled pores, smudging) are the hardest
  thing to remove, because they are fine detail and a low-strength redraw is
  built to keep fine detail — at strength 0.4 the model sharpens them instead.
  What works, measured: **strength 0.55–0.8 with 16–20 steps**, and
  **soften 1–2 px** in *Advanced*, which blurs each tile before the model
  sees
  it so it cannot sharpen the leftovers back. Soften on its own, or at low
  strength, simply gives the blur back. Pick the
  *drift skin de-artifacting* restoration prompt with it, and keep the area
  small — at that strength the composition holds because the region is small
  and feathered, not because the model is being careful.
- **The tile grid** is drawn over the picture while a tiled task is open —
  **Redraw**, **Edit**, or **Upscale** with a SeedVR2 or PiD model: every tile
  the job would run, one pass of the model each, with the
  brighter bands where neighbours overlap and are blended back together. It
  follows the fields as you type them — a larger tile means fewer boxes — and
  once you drag a box it switches to the tiles of the *window* that box is
  repainted through, which is what would actually run. **show the grid** turns
  it off; it sits at the foot of the panel under the cost line, beside the
  reset, since neither of them changes anything about the job.
- **Move the grid** to choose where the cuts fall: the pointer over one of its
  lines turns into a move cursor, and dragging takes the whole grid with it. A
  face across a seam is repainted in two passes and blended, which is how eyes
  come back not quite looking together — slide the grid until the face sits
  inside one tile and it is painted once, whole.

  The shift snaps to the multiple your model aligns to, and moving the grid
  usually costs one tile more than leaving it alone: the even spread is the
  fewest tiles there can be. The two tiles at each end come out shorter, and
  drift keeps them at least two overlaps long — an end tile shorter than that
  is given up and its neighbour reaches the edge instead — so no tile is ever
  a sliver the model cannot paint.

  **↺ reset the grid**, beside that switch, puts it back where it falls on its
  own — which is also the fewest tiles. It sits there greyed out until you move
  the grid, and then names the shift it would undo, so the button doubles as
  the sign that the grid can be moved at all. Opening another image starts from an
  unshifted grid; stepping through a batch keeps the shift, the images being
  the same size.

  An **upscale** has the same grid, switch and reset, drawn where its tiles
  fall on the picture you are looking at. Its **tile** field (*Advanced*) is
  in pixels of the result: empty is the largest tile the model takes in one
  pass, which is the fewest tiles; a smaller number, from 1024, cuts more of
  them. An upscale always works the whole picture — no box can be drawn while
  it is open, and the one you drew for a redraw is kept for when you come back.
- **The tile size** is rounded up to what the model accepts — most take any
  multiple of 16, Qwen Image 2.1 wants 32 — so the tile count before the button
  may sit on a slightly larger tile than the one you typed.
- **Which models redraw.** Only those that take a reference image as
  context: Flux.2 [dev], Flux.2 Klein 4B and 9B, and Qwen Image 2.1. A tile that sees
  nothing of its surroundings is repainted as something else — measured on a
  16k picture, sand came back as rock and an areola was erased — so the other
  models are not offered. Krea 2, Qwen Image, Z-Image, Boogu and Mage-Flow
  *edit* what they are given a reference of, and ERNIE, Ideogram 4 and
  HiDream take none at all. **SenseNova U1.5 cannot redraw**: it returns noise for any
  input image.
- **The reference** is the block of three by three tiles around each tile,
  cut from the picture and fitted within the reference size (768 px): the
  tile is about a third of it whatever the picture's size. The whole picture
  is no longer sent — on a large picture a tile was a few dozen pixels in it,
  which told the model nothing. It costs about a fifth more time per tile on
  Qwen Image 2.1 on the drift runner (half more on sd-cpp) and three quarters
  more on Klein; *never send* in the
  *reference* group switches it off for a job.
- **Steps.** The steps you ask for (or the configuration's) all run, whatever
  the strength: drift schedules more so that the strength skips the extra
  ones. Before, a 4-step model at strength 0.4 ran a single step and gave the
  picture back nearly unchanged.
- **Colour.** Each redrawn tile is given the picture's colour and shading
  back — only the detail the model drew is kept. A redrawn tile drifts a few
  shades on its own, and pasted into the untouched picture that drift shows as
  an edge; this is what made a box redrawn alone look wrong. It is always on.
  It does not make a much stronger repaint blend in: texture follows strength,
  so redraw a part at the strength its surroundings were redrawn at.
- **Context** shows the model each tile inside its surroundings at full scale:
  a close-up tile seen alone can be taken for something else — on a zoomed
  belly Krea 2 reshaped the waist and drew a fold that was not there. With
  *context* above 0 the model is handed each tile inside that many px of its
  surroundings, as already redrawn, and a mask has it repaint the tile alone;
  the tiles run in order, each one painted in before the next is cut, so every
  tile sees its neighbours' result. The model paints the whole window, so it
  costs time: a 1280 tile with 128 of context is painted as 1536, about 1.8×
  the time per tile. Enough context is what matters — measured on the same
  belly, 256 around a 1024 tile held the shapes, while 128 around a 1024 tile
  (a 1280 window) still turned the belly into a pair of knees. The cost line
  says when context is on, and *keep the tiles* keeps each window beside the
  tile cut from it.
- The full-size view fits the window: the image takes the height that is
  left, the images of a batch run down its left side (with `2 of 8` in the
  title bar, so a batch is never a surprise), and the right-hand side shows
  one tab at a time — **Parameters**, **Redraw & upscale**, **History**, and
  **Inputs** where the generation had input images. Whatever you have typed
  into a panel stays where it is when you switch tab or task, and drift
  remembers which tab — and which task — you were on for the next image.
- **◨ Hide details** in the title bar folds the panel away and gives the whole
  card to the image — worth it for wide images, where the width is what limits
  them. The box for a partial redraw can only be drawn while the
  **Redraw** task of the **Redraw & upscale** tab is showing, so it never
  appears while you are reading parameters or setting up an upscale, neither
  of which has any use for it; what you selected is still there when you come
  back.
- The **prompt** sits above the parameters at full width, shortened to four
  lines; click it (or the ⤢) to read the whole of it over the page.
- The image you see is a copy scaled to the screen, not the file: drift's
  upscales reach 16384² and hundreds of MB, which a browser takes the better part of a
  second to decode and much longer on a busy machine. **⤢ Full size** under
  the image opens the original in a new tab, which is where full resolution
  is looked at. Everything else — redraw, upscale, the selection — works on
  the file itself, whatever the screen shows.
- **The address bar follows the image**: opening one gives it a URL, in the
  gallery and in a project alike, so a refresh — after a repaint lands, say —
  comes back to the same image rather than the grid.
- **Redrawing one part**: drag a box over the image and the button becomes
  **✨ Redraw selection** — only what you framed is repainted, the rest of the
  file is untouched. A click on the picture clears the box, and so does
  switching output or looking at the original.

  The box can be adjusted: drag a corner to resize it, drag its middle to move
  it, click the picture to clear it. It says as it moves what it will cost —
  `320×240 · 1 tile · repaints 1024×1024` — and its sides stick to the sizes
  where the tile count changes, so it takes a deliberate pull to buy another
  pass of the model.

  The model is never shown just your box: drift widens it to at least the
  *window* size (1024 px a side, in *Advanced* → *area*) plus a *margin*
  (64 px), because a
  model handed a 300×200 crop paints mush. The extra area is context only —
  the result is blended back over the margin, so nothing outside your box
  changes and the repaint has no visible edge. The last line before the button
  says what it will do and cost, for instance
  `selection 312×248 → window 1024×1024 · 1 tile · 24 steps each, about 10
  sampled`, and it reads the same way with no selection, for the whole image.

  Same instructions, same restoration prompt, same everything else as a full
  redraw. Good for a face, a hand, a bit of background that came out soft.
- **✨ Edit** changes the picture instead of repairing it: write what should be
  different under *change* — "make the bikini top red", "she wears a thin gold
  necklace" — and the model makes that change while the rest stays what it
  was. It needs a configuration on an architecture tagged **edit**, a model
  that edits an image by instruction: Flux.2 Klein works; Krea 2 does not (it
  draws a different picture), so it is not offered.

  **Carried up by** says how the edit reaches the picture's size. With a
  SeedVR2 upscaler — the default when one is installed — the edit is made
  *once*: the picture, or the window around your selection, is reduced to what
  the model takes in one pass (as it is up to 1536 px, else a half or a
  quarter), the model edits that, and only what changed is brought back to the
  picture's size by the upscaler and put in. The model sees the whole shirt it
  is asked to change, so the change is one decision; what is new has the
  upscaler's texture, like the rest of an upscaled picture. The line before
  the button says the pass's size and the scale it is carried by.

  With *none — tile by tile*, each tile is handed to the model on its own, in
  order, each cut from the picture as edited so far. That is fine for a change
  that sits inside one tile; a garment that spans several comes out different
  from tile to tile — a red chest and beige sleeves.

  Either way the model is given *the image to edit*, with the *drift seamless
  edit* prompt (Advanced → *prompt*) and your instruction; there is no
  strength — at any strength a redraw either ignores the instruction or
  relights what it should keep. The model re-renders everything it is given,
  the untouched skin and background included, and there drift takes the
  original back: wherever the edit did not really change anything, the result
  is the source's own pixels, so its grain, tone and texture cannot drift;
  where it did, the change is brought to the source's colours and feathered in.

  A box works as for a redraw — the same window, margin and grid — and the
  button becomes **✨ Edit selection**; carried up, nothing changes beyond the
  box and its margin, whatever the model did in the rest of the window. The job
  log says how much changed; more than half is flagged, since a model that
  repaints rather than edits lands there (a large edit can too). *keep the
  tiles* keeps what the model was given, its raw edit, the change mask (white
  where the edit was taken) and the composite.

  What Edit is not for: a defect to repair (eyes, a hand) — the model tends to
  re-grade the whole picture around it; set the repair up from **🤖 Read the
  picture** in Redraw instead. And a PiD upscale is not offered to carry an
  edit: it turns the edit model's grain into a crackle beside the original.

**While a job runs you watch the result appear**: each tile the model finishes
is drawn over the original where it belongs, the tile being worked on pulses,
and the rest is the plain grid. The picture rebuilds itself piece by piece, so
you can judge a long job without waiting for it — and stop it early if the
first tiles are not what you wanted. **⤢ Full size so far** in the header
opens it at full resolution; on a very large picture the first opening after a
tile takes a while (a 16384² picture is ~20 s to prepare and heavy for the
browser to show), opening it again before the next tile is immediate. A paused
job keeps showing how far it got. While a job is on
that image, it is its grid you see, not the panel's — **show the grid** turns
it off.

**The card says what is left** once the first tile is done — "tile 7 of 21,
about 12 min left" — from what the tiles of this run have actually cost, so it
gets truer as the job goes.

**One job at a time per image.** Starting a second one on the same picture is
refused, naming the job in the way: stop it, or cancel a paused one, first.

**Pause a long job** with the **Pause** button on a running tiled job: it stops
after the tile it is on — never losing that tile — keeps everything it has
already made, and stops its server, so the GPU and its memory are free. The
card says "Pausing — it stops when this tile is done" as soon as you click, and
the button becomes **Force pause**: if you would rather not wait for a tile
that takes minutes, that one drops the tile in flight (after asking), and the
tile runs again when you resume. The
card then shows how far it got, with **Resume** and **Cancel**. Resume starts a
server again and carries on at the first tile it does not have, with the same
seed and the same tiles as before. A paused job survives closing drift: it is
listed again on the next start, so you can recompile, run something else on the
GPU, or come back tomorrow. Cancel drops its tiles. Resume says no if the job
could no longer be the same one — the picture is gone, or the settings now lay
out different tiles — and nothing is thrown away when it does.

Jobs are listed under the panels with their progress — how many tiles are done,
and under it what the model is doing right now: loading its weights, then the
sampling steps of the tile in flight, exactly as the inference page shows them,
with the last log line whenever no bar is running — and, on failure, the tail
of their log. A running job has a **Stop** — it kills the sd-cli behind an
upscale, or asks the server to drop the tile a PiD, redraw or edit is
generating;
what the job had written is removed, and a stopped job keeps nothing. A
finished job's **Open result** opens the new entry, and its **×** closes the
card once you are done with it — the result stays in the gallery, and the next
redraw or upscale of the same image starts from a clean panel rather than
beside the last one's notice. Logs live in
`~/.cache/drift/logs/postprocess-<job>.log`.

Typical chain: generate at base size, PiD upscale ×4, then redraw the result.
