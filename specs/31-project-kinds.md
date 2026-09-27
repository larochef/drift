# 31 — Project kinds: image or video

**Status:** done in code, not yet run live
**Depends on:** 19 (projects and the workspace), the architecture tags

A project says whether it makes images or videos, and its workspace offers
only the run configurations whose models make that. Every project saved
before kinds existed is an image project.

## What it does

- The new project form has a "Makes" choice, Images by default; the
  workspace header has a select to change it. The model picker narrows at
  once and is titled "Image model" or "Video model".
- Project cards show the kind. A video project's cover is its newest video,
  the file itself (no transcoder): the first frame shows, and it plays muted
  while hovered.
- A live session of the other kind stays in the picker, marked "(live, not a
  video model)", since sessions belong to the machine, not the project.
  Nothing prevents generating with it.
- A video generated in a video project's workspace shows its progress and
  result and starts or joins a version, whose card shows a video thumbnail.
  The compare view diffs prompts either way.

## Shape

- `ProjectKind = Image | Video`; `Project.kind: ProjectKind = Image`.
  `ProjectKind.accepts(architecture)` in shared: an image project takes an
  architecture tagged `image` or `edit`, a video project one tagged `video`;
  `upscale` and `llm` match neither. Tags are derived once at startup for
  architectures registered before tags existed (`ArchitectureTags`).
- `PromptVersion.kind: "img_gen" | "vid_gen"` with `imageParameters` or
  `videoParameters`. The video submit takes the same `project`, `version`,
  `origin` query parameters as the image one; keeping a free-play video into
  a project joins or starts a version like a kept image.
- A video recipe leaves out the seed, the start and end images and the
  control frames. A recipe of another kind than its parent's is a new version
  noted "changed kind".
- The cover endpoint checks the chosen cover's type against the kind and
  falls back to the newest result of that kind. Changing the kind rebuilds
  the covers.

## Notes

- `Project.kind` defaults on read and is omitted on write (jsoniter's
  transient defaults), so an older file loads as an image project untouched.
- `PromptVersion.kind` has no default: a project file whose versions still
  carry the older image-only `request` field does not decode, and the listing
  skips it with a warning.
- The gallery filters on each generation's own kind; the assistant is not
  told the kind (the brief says it if the user wrote it); the post-process
  pickers work on images whatever the project.
