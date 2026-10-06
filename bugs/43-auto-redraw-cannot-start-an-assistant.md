# Bug 43 — Auto redraw has no way to start an assistant when none is running

**Status:** fixed in code 2026-10-06 (Done, below) — compiled, gate green, not seen in a browser
**Severity:** medium (the feature stops at a message; the user has to leave the gallery, start an assistant on
another page and come back)
**Files:** `frontend/src/drift/frontend/pages/gallery/AutoRedrawCard.scala`,
`backend/src/drift/backend/postprocess/RedrawPlanner.scala`,
`AssistantProxy.visionSession` (the refusal), the assistant drawer's own start of an assistant

## Symptom

**🤖 Read the picture**, in the redraw panel, asks the first running assistant whose model reads images
(`specs/52-auto-redraw.md`). With none running the call is refused — "no running assistant reads images:
start one whose model has a vision projector" — and the card can do nothing more.

## Wanted

When no assistant that reads images is running, the card offers to start one, from where the user is:

- which one: a run configuration of an assistant whose model reads images — the one the user last ran, else
  a choice among those installed; none installed is said, with the way to the page that installs one;
- the card shows the assistant loading and asks the question when it is ready, so one click still reads the
  picture;
- an assistant that is running but does not read images is not the same case: the card names it and
  offers to **stop it and start one that reads images** in its place, as one action — never a second one
  started unasked beside it, and never the first stopped without the user's click (a conversation may be
  going on with it).

## Done (2026-10-06)

- `VisionAssistants` (built by `GenerationDetailHost`): the chat configurations that read images
  (`Architecture.readsImages`: a `--mmproj` slot with a file assigned), the last one run first; the live chat
  sessions; what stands between the card and a reading (`State`: ready, loading, another model live, none
  running, none installed); and the start of one, a `Stop` of the session it replaces then a `Launch`.
- `VisionAssistantStarter`, in `AutoRedrawCard` under the button: the select (hidden with one choice) and
  **▶ Start … and read the picture** through `LaunchOrDownload`, so missing weights or a missing runtime are
  offered as every other launch from the gallery offers them; **⏹ Stop …, start … and read the picture** when
  a chat model that does not read images is live; the way to the Models page when no configuration reads
  images. The card waits and asks when the assistant serves; a start that fails ends the wait.
- Settled as built, to change if wanted: the assistant **stays loaded** after the reading (52 keeps it beside
  the redraw's server); **memory** goes through the launch's own check, and the session's warning is shown in
  the card (`MemoryWarningView`); **reuse** is of the gallery's launch path (`SessionService` commands,
  `LaunchOrDownload`), not of the workspace's `ModelPicker`, whose "new configuration" entry needs services
  the gallery does not hold.
- "Reads images" is read off the configuration before a launch, and off `/props` by the backend once it
  serves: a projector that fails to load would leave the card asking a model that refuses.

## Settled before building (kept for the record)

- **Memory.** The assistant stays loaded beside the redraw's server (52), and an assistant beside a video
  model already oversubscribed memory once (`MemoryHeadroom`, `Session.memoryWarning`): starting one from the
  card should go through the same check and show its warning.
- **Stopping it.** Whether an assistant the card started is stopped once the reading is in, or left for the
  next picture.
- **Reuse.** The workspace's assistant drawer already starts an assistant; the card should use that, not a
  second way of launching one.
