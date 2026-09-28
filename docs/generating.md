# Generating images and videos

Generation happens inside a **session**: a model server drift launches from a
run configuration. You start one from the Models page (free play) or from a
project's workspace (see [projects.md](projects.md)).

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
- **Seed** with a **Random** toggle: drift draws the seed itself and shows it,
  so every result is reproducible. Untick Random to keep a seed.
- The LoRA picker, prefilled with the configuration's default LoRAs, with
  **+ Add LoRA** to install more. See [loras.md](loras.md).
- **↻ Restart**, beside **Stop session**, stops the model and starts it again
  on the same runtime, keeping the form as it is: what a LoRA installed since
  the launch needs, and the quick way out of a server gone wrong.

Folded sections, opened on click. A closed section says when it is doing
something (an image attached, hires on, a non-default sampler):

- **Sampling**: sampler and scheduler.
- **Inputs**: init image (img2img) with a strength, reference images (edit),
  end image on video, and a mask once an init image is attached (inpaint).
  A reference keeps its own shape whatever the output's: drift sizes it the
  way sd-cli would (about a megapixel) before sending it, or — on an sd-cpp
  older than master-892, whose server stretches references to the output —
  pads it to the output's shape with grey so it is scaled, not squashed.
- **Hires**: a second pass through an upscaler model or a latent mode, with
  its own denoising strength and steps.
- **VAE tiling**: fixes high-resolution decode crashes.
- **Guidance & SLG**: image CFG, distilled guidance, skip-layer guidance.
- **Batch**: images per run (bounded by the model), output format and
  compression. A batch is one job; sd-cpp makes the images one after another.

## Modes

- **Image**: text to image; add an init image for img2img; add reference
  images for an edit model. "Edit" is not a separate mode.
- **Video**: text to video, or image to video with an init or end image.
  Tabs appear only when the model offers both modes.

## Results

- Each submission shows its queue position, then the sampling progress from
  the log, then the image or video. Submit several and a short list appears
  above the result with the jobs waiting behind the one on screen, each with its
  own **⏹**, so a queue can be trimmed without touching the rest — jobs queued
  from another project included, since they hold yours up just the same. A stop
  takes a moment to land — the button spins until it has.
  **⏹ Cancel** stops a queued or running job,
  in a project workspace as well as here. Some models and builds cannot stop a
  job once sd-cpp has taken it; drift then asks whether to stop it by killing
  sd-cpp and starting the same configuration again. Say yes and the generation
  ends, the model reloads (as long as the launch took), and the session comes
  back by itself — there is no need to restart drift to get out of a run you
  did not want.
- A batch lays its images side by side.
- **🤖 Ask the assistant** on a result sends the image and its parameters to
  the assistant. See [assistant.md](assistant.md).

## Free play or project

- From the **Models** page you are in free play: nothing is saved. Results
  show their duration and a **Keep into** row: keep "the gallery only" or an
  existing project. Kept images become gallery entries (and a version of the
  project, if one is picked). **Clear results** or stopping the session throws
  the rest away.
- From a project workspace every generation is recorded, becomes a version of
  the project and appears in the gallery. See [projects.md](projects.md) and
  [gallery.md](gallery.md).
