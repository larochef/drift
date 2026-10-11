# Generating images and videos

Generation happens inside a **session**: a model server drift launches from a
run configuration. You start one from the **Sandbox** or from a project's
workspace (see [projects.md](projects.md)); **Launch** on a Models page card
opens the Sandbox on it.

## Starting a session

- On a run configuration card, pick a runtime in the select beside **▶ Launch**
  (the default runtime is preselected) and launch. The card's command-line
  preview is exactly what runs.
- The session starts as *Starting*. Big models take minutes to load: watch the
  progress bar (tensors loaded, MB/s) and the activity line under it, which
  shows the last thing the server said when there is no bar to show.
- Any notes under the card come from resolving parameters against the runtime:
  a flag this build cannot take was dropped, or one was added. See
  [models.md](models.md).
- The session is *Ready* once the server answers. A crash shows *Failed* with
  the error line and the tail of the log; the full log is in
  `~/.cache/drift/logs/`.
- If the loaded models leave almost no memory — typically an
  assistant model beside a large video model — the card and the generating
  panel warn that memory is nearly full. Generations then crawl, because the
  weights are read back from disk over and over; stop one of the loaded
  models.
- The **Machine** box at the bottom of the sidebar shows, on every page, the
  last five minutes of the GPU's load and of the memory: a curve each, green
  in the lower third, yellow in the middle, red at the top, with a line each
  minute; point at a curve for the value at that moment. The memory curve
  rises as memory fills, and counts what is really left (the loaded weights
  are not free, unlike what `free` calls available). Under them, the memory
  the GPU holds. The current figures also sit in one line under the bar of a
  loading model, a generation or a post-processing job. When a run seems stuck, read that line: "GPU idle for
  40 s" means it is not computing — it is reading from disk because memory is
  full, or working on the processor.
- Only one generation session runs at a time by default. Launching a
  configuration that already runs focuses it. **⏹ Stop** ends it; quitting drift
  stops every session too.
- A log view under the panel shows the raw server output, including what
  happened before you opened it.

## The form

The form is built from what the loaded model reports it supports, seeded with
the run configuration's parameters. You adjust a known-good starting point.

Always visible:

- **Prompt** and **negative prompt**, then **✨ Generate** right under them.
- **Width**, **height**, **steps**, **CFG**; on two-expert video models (wan
  2.2) also **high-noise steps** and **high-noise CFG**; **frames** on video.
- **↺ Reset parameters** puts the size, steps, CFG, sampler, schedule, flow
  shift and guidance — a video's frames and FPS too — back to what the
  configuration and the selected LoRAs set, after a reused recipe or your own
  changes. Prompts, seed, inputs and LoRAs stay.
- **Seed** with a **Random** toggle: drift draws the seed itself and shows it,
  so every result is reproducible. Untick Random to keep a seed.
- The LoRA picker, prefilled with the configuration's default LoRAs, with
  **+ Add LoRA** to install more. See [loras.md](loras.md).
- **↻ Restart**, beside **Stop session**, stops the model and starts it again
  on the same runtime, keeping the form as it is: what a LoRA installed since
  the launch needs, and the quick way out of a server gone wrong.

Folded sections, opened on click. A closed section says when it is doing
something (an image attached, hires on, a non-default sampler):

- **Sampling**: sampler and scheduler; **Flow shift**: how far the steps lean
  toward the noisy end (empty: the model's own). And **Sigmas**: the noise
  levels to step through, typed as a list (`1.0, 0.9375, 0.875, 0.75, 0.5,
  0.25`), in place of the scheduler's. A turbo LoRA is trained on its own few
  levels and its page lists them; with a list, the steps are the list's,
  whatever **Steps** says. sd-cpp takes them for every model, the drift
  runner for its image models — Qwen Image 2.1, Krea 2, FLUX.2 [klein],
  HiDream O1, Mage-Flow, Nucleus-Image and LLaDA-Image — and not for video or
  PiD, where it refuses them.
  On Qwen Image 2.1 the drift runner reads the list as the model's own
  pipeline does: the levels are shifted for the image's size, so a list
  published for it is typed as written and holds at every size. A flow shift
  beside the list replaces the size's; a flow shift of 1 takes the list as the
  final levels (Fun-Acc's are given that way). sd-cpp takes every list as
  final. A turbo LoRA wants CFG 1, its own levels and at least its steps; the
  ones drift offers, and Viggle's, Pruna's and Fun-Acc's once installed, come
  with theirs filled in.
- **Inputs**: init image (img2img) with a strength, reference images (edit),
  and a mask once an init image is attached (inpaint). On video: start and
  end images, references, guides and a control video, as far as the model
  takes them (see [Video inputs](#video-inputs)).
  Each slot takes a file from the disk or, with **From the gallery**, one of
  the gallery's entries: a picker with the gallery's tiles and its project,
  configuration, NSFW and prompt filters, opened on the project's own results
  in a workspace. It offers what the slot takes — images, videos or both —
  and a slot that takes several lets you tick them in order. A result made
  from a picked entry links back to it (see [gallery.md](gallery.md)).
  **🤖** on an image input stages it for the assistant's next message, so a
  file you browsed for is not browsed for again.
  A reference keeps its own shape whatever the output's: drift sizes it the
  way sd-cli would (about a megapixel) before sending it, or — on an sd-cpp
  older than master-892, whose server stretches references to the output —
  pads it to the output's shape with grey so it is scaled, not squashed.
- **Hires**: a second pass through a latent mode or a plain filter, with
  its own denoising strength and steps. Upscaling with a model (ESRGAN, PiD,
  SeedVR2) is the **Upscale** task on the result, in the gallery.
- **VAE tiling**: fixes high-resolution decode crashes.
- **Guidance & SLG**: image CFG, distilled guidance, skip-layer guidance.
- **Batch**: images per run (bounded by the model), output format and
  compression. A batch is one job; sd-cpp makes the images one after another.

## Modes

- **Image**: text to image; add an init image for img2img; add reference
  images for an edit model. "Edit" is not a separate mode.
- **Video**: text to video, or image to video with a start or end image.
  Tabs appear only when the model offers both modes.

## Video inputs

The **Inputs** section offers what the loaded model reports it takes; a
picker previews each file as a still, a player or a sound.

- **Start / end image**: the first and last frame. Both are fitted to the
  video's size.
- **References**: images, clips (with their sound) and audio files, in
  order — the prompt names them by position, so **←** / **→** move one.
- **Guides**: a file held at a frame of the video; the frame index counts
  from 0, and a negative one from the end (-1 is the last frame, the default).
- **Control video**: a pose, depth or edge video steering the motion, with a
  **strength** (default 1). Folded under **Control range and mask**: the
  fraction of the steps it applies over (0 to 1) and an optional **mask**
  (image or video, white is regenerated) over an optional **source video**.

References, guides and control videos keep their own size. The gallery shows
every input a video was made with, and **reuse** loads them back.

Per model, on the drift runner:

- **MiniMax H3**: text alone, start and end images and guides, or references
  (not beside start, end or guides). Any checkpoint runs every way: the usual
  (fl2va) one is trained for the first three, the **ref2va** one for references,
  and community merges (DaSiWa and Eros MAX are built on ref2va) are made for
  all of them; the log warns when a file runs the way it was not trained for.
  Start and end images and references are read by the text encoder's vision
  tower. The MiniMax H3 encoder file carries it; a text-only GGUF of Qwen3-VL
  does not — give the tower then in the configuration's **llm vision** slot
  (the `mmproj` file published beside such a GGUF, or the Qwen3-VL-32B one in
  drift's model list). A reference clip
  is costly — about three times the step time for one second. An int8
  checkpoint takes no more GPU memory than its file. A
  control video needs the **Fun ControlNet
  union** in the configuration's **control-net** slot. Both files are in drift's
  model list, ready to download.
- **Wan 2.2 A14B (I2V)**: start and end images; LoRAs, a pair's high-noise
  file applied to the high-noise expert (see [loras.md](loras.md)).
- **LTX 2.5**: start and end images; LoRAs.

## Results

- Each submission shows its queue position, then the sampling progress from the
  log, then the image or video. Wan 2.2 with two experts shows a bar per expert,
  marked "high noise" then "low noise"; a bar moves once a step is done, which
  takes minutes for a long or large video (the log gives the token count and
  each expert's steps before the first one). After the last step an LTX 2.5
  video shows a second bar, "decoding", while its frames are made from the
  latents, which takes minutes for a long video. A batch shows two bars: "Image 2
  of 4" with the images already done, and under it the steps of the image
  being made. A video on the drift runner shows two bars as well: "Step 2 of
  5: sampling" over the whole generation (reading the references, the prompt
  and the inputs, sampling, decoding, writing the file), and under it that
  step's own bar with the time it still needs ("about 4 min left for this
  step"). On the drift runner a sampling step that lasts more than a second
  moves the bar while it runs, for pictures and videos alike — from the second
  step on (the first one measures how much work a step is; MiniMax H3 moves
  from the first).
  A picture also shows itself while it is sampled, on the drift runner and
  on sd-cpp alike:
  under the bar, a small **preview** of what the model has so far, redrawn
  after each step and captioned with that step. It is rough at first and
  settles as the steps go, soft all along (it is read straight off the
  model's latents, not decoded), and it leaves when the picture arrives. A
  batch previews the image being made. On the drift runner FLUX.2, Krea 2,
  Qwen Image 2.1, Mage-Flow, Nucleus-Image and HiDream O1 have one; sd-cpp
  draws its own for every image model.
  A video on the drift runner (Wan 2.2, LTX 2.5, MiniMax H3) previews as a
  clip that plays in a loop at the video's own pace, silent, coarser in
  time than the video (one picture for every few frames) and smaller on the
  page than a picture's preview, being rougher; it leaves when the
  sampling ends, before the decoding. **⏸ Pause**, under it, holds the clip
  on one frame — the first steps flicker more than they move — and every
  later step then shows as a still, until **▶ Play**. On sd-cpp a video
  previews as a still of its first frame.
  **Preview while generating**, the checkbox over the results, turns all of
  this off: unticked, drift asks the server for no preview from the next
  generation on, and hides the one on screen. drift remembers the choice.
  When the server could not apply a LoRA, or only part of it — a file made
  for another model, or named in a way it does not read — the picture is
  still made, and a yellow note above it says what was left out; the gallery
  keeps it as a **Warning** in the entry's parameters. The drift runner
  reports this; sd-cpp says nothing of it.
  Submit several and a short list
  appears above the result with the jobs waiting behind the one on screen, each
  with its own **⏹**, so a queue can be trimmed without touching the rest — jobs
  queued from another project included, since they hold yours up just the same.
  A stop takes a moment to land — the button spins until it has. **⏹ Cancel**
  stops a queued or running job, in a project workspace as well as here. Some
  models and builds cannot stop a job once sd-cpp has taken it; drift then asks
  whether to stop it by killing sd-cpp and starting the same configuration
  again. Say yes and the generation ends, the model reloads (as long as the
  launch took), and the session comes back by itself — there is no need to
  restart drift to get out of a run you did not want. - A batch lays its images
  side by side. - **🤖 Ask the assistant** on a result sends the image and its
  parameters to the assistant. See [assistant.md](assistant.md).

## Sandbox or project

- The **Sandbox** (sidebar) is for trying a model: nothing is saved. Tabs pick
  **Image**, **Video** or **Text**, each laid out like a project workspace of
  that kind: the model pickers at the top, with **Log**, **Restart** and
  **Stop session** beside the live model. Image and Video also have an
  **Assistant** picker: its chat opens as a drawer on the right, with **Apply
  to form** and **Apply and run** on its proposals, as in a project. Text is a
  chat with the raw model. Results show their duration and a **Save into**
  row: "the gallery only" or an existing project. Saved images become gallery
  entries (and a version of the project, if one is picked).
- Everything not saved goes when you leave the page (drift asks first), when
  you switch tabs (which also stops the image or video model and clears the
  chat, after asking), when the session stops, or with **Clear results**.
- A model a project already runs is shared rather than loaded twice: the
  Sandbox says *Running for* that project. Picking another model of the same
  kind asks before stopping the project's.
- From a project workspace every generation is recorded, becomes a version of
  the project and appears in the gallery. See [projects.md](projects.md) and
  [gallery.md](gallery.md).
