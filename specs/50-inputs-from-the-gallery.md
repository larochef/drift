# 50 — Inputs from the gallery

**Status:** partial — the picker, the link and Ask the assistant are done in code, not yet run live; a video as a still and *Use as input* remain
**Depends on:** 12, 14, 21, 47

Every input a model takes — an init image, references, a start or end frame,
a control video, the pictures a message to the assistant carries — is picked
from the disk through the browser's file dialog, even when the picture
wanted is one drift made a minute ago and shows in its gallery. And a file
picked for the generation form has to be browsed for a second time to show
it to the assistant. Inputs come from the gallery as easily as from the
disk, and a picture already on the page goes to the assistant in one click.

## What it does

- **From the gallery.** Every input slot of the generation form, and the
  assistant's composer, offers **From the gallery** beside **Browse**. It
  opens a picker over the page: everything the gallery holds that matches,
  in one grid, newest first — no days to open — with the
  gallery's filters — project, configuration, kind (image or video), NSFW —
  and a search on the prompt. In a project it opens on that project's
  results; the filter can be cleared.
- The picker shows only what the slot takes: stills for an image slot,
  videos for a video slot, both where the slot reads both. A slot that takes
  several (references) lets several be ticked, in the order ticked; a
  generation made as a batch offers each of its outputs.
- **A video as a still.** For an image slot, a video offers one of its
  frames: the first, the last, or the one paused on — the last frame of a
  clip is what the next clip starts from.
- A picked entry behaves as a browsed file does in the form: same thumbnail,
  same removal, same request.
- **A link back.** A result made from a picked entry remembers it: its detail
  lists the gallery entries its inputs came from, each opening that entry,
  and the entry lists what was made from it — as an upscale and its source
  do. A browsed file leaves no link. Deleting the source leaves the result
  and its own copy of the input; the link then says the source is gone.
- **Saved results only.** The picker shows the gallery: a Sandbox result is
  there once it is saved, and not before.
- **To the assistant.** An image input's thumbnail in the generation form
  carries 🤖, as a result carries **Ask the assistant**: the picture is staged
  for the next message, without browsing for it again — a gallery entry with
  its parameters, a file as it is. Not on a mask, a clip or a sound. The
  composer has **From the gallery** too, for pictures.
- **From the assistant.** A picture staged in the composer offers the
  reverse, **Use as input**, into the form's first free slot that takes it.

## Shape

- A picked entry is the **URL its file is served from** in the slot, where a
  browsed file is a data URL: the form's state, its previews
  (`MediaPreview` reads both) and the request stay as they were, and a video
  never passes through the browser.
- `GenerationFiles.servedInput` reads such a URL: the file as the data URL
  sd-server takes, and the entry it is an output of, found by the sidecar
  its name starts with. `ServedInputs`, in `GenerationSubmissions`, runs
  every input slot of an image or video request through it before the
  request is forwarded and its inputs are copied beside the outputs. A URL
  whose file is gone refuses the submission.
- `Generation.inputSources: List[InputSource]` — slot (the names the
  persisted inputs carry: `init`, `ref0`, `reference1`, `control`…),
  generation id, output index, date and file name. Sidecars are not a
  migrated collection: one written before has no such field and reads as an
  empty list.
- `pages/gallery/GalleryPicker`: a `BrowserModal` with one grid of
  `GenerationCard` tiles, one per output the slot takes; opening it reads
  every day of the history. `GalleryFilter` holds the filter rules, for the picker
  and `GalleryPage`. `WorkspaceBody` hosts one picker for its generation
  panel and its chat, opened on the project in a workspace.
- `pages/generate/MediaPickers`: `InputTools` carries the two things a slot
  can do besides browsing — the gallery and the assistant — down
  `GenerationForm` → `InputsSection` → the pickers.
- The assistant: a picked entry is staged as the gallery's own **Ask the
  assistant** does (`AssistantService.outputAttachment`); any other input is
  fetched where the form holds it and uploaded
  (`AssistantAttachments.uploadFrom`). The composer's **From the gallery**
  stages pictures the same way.
- `GenerationLineage` shows **Inputs from** and, on the source, the
  generations that took it under **Made from this**; `GenerationDetailHost`
  resolves both in the loaded days.

## Remaining

- **A video as a still**: a frame of a video for an image slot.
- **From the assistant**: *Use as input* on a picture staged in the composer.
- A reused generation loads its inputs back as files: the result of the
  reuse is not linked to the entries the original's inputs came from.
- Open, to settle with use: whether imported pictures need a filter of their
  own, so that a library of reference pictures can be kept there.

## Post-v1

- **A part of a video.** A video slot takes a whole file, or a gallery
  video, and a range within it — from 1:20 to 1:25 — rather than the whole
  clip: two time fields under the thumbnail, the preview playing the range.
  The cut is made before the request (ffmpeg on the backend, which the
  runner's webm already needs), so the model receives the range alone.
