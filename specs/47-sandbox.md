# 47 — Sandbox

**Status:** done in code, compiled and backend-tested, not run live
**Depends on:** 22, 41, 46

Free play (22) lives inside the Models page: any live session, including one
a project started, turns the page into a "nothing here is saved" generator,
so one model shows up in two places under two rules. The Sandbox moves free
play to a page of its own and leaves the Models page to configurations.

## What it does

- A **Sandbox** item in the menu opens a page shaped like a project workspace
  without its project: no brief, no versions, no project gallery. An
  **Image / Video / Text** switch at the top picks the kind; each kind shows a
  configuration picker filtered to it, then the same generation panel (image,
  video) or chat (text, the raw model, no system prompt) a workspace mounts.
- The panel shows the latest result, with its duration, and the queue, as
  22's did. A result offers **Save into** the gallery only or a project
  (22's Keep). Nothing else survives.
- **Not persistent.** Leaving the page by a link, a reload or closing the tab
  warns that unsaved results will be lost; the browser's Back is not asked
  about. However it is left, the results and the chat are cleared. They are
  also cleared when the session stops and at startup, as before.
- **Switching kind** stops the image/video session and resets the assistant
  (clears the chat) if it was started, after a confirmation naming what goes.
  One exception: a chat reply's *Apply to generation form* opens the Image tab
  without stopping or clearing anything, since the proposal has to reach the
  form.
- **Sharing a loaded model.** When the configuration picked is already
  running for a project, the Sandbox uses that session rather than reloading
  it, and says so: *Running for* ⟨project⟩. Its results are still scratch.
  Picking another configuration of the same kind says it will stop the
  project's model before doing it.
- **The Models page shows configurations only.** A live session no longer
  takes the page; its configuration's row shows *Running in Sandbox* or
  *Running in* ⟨project⟩ with **Open** and **Stop**. **Launch** on a row opens
  the Sandbox on that configuration's kind with it picked. "← All
  configurations" and "↩ Back to N running" go.

## Shape

- `Session.projectId` / `LaunchSessionRequest.projectId`: the workspace that
  launched it, none from the Sandbox, the Models page and the gallery; a
  restart keeps it. Who launched it, not who uses it.
- `pages/sandbox/SandboxPage` at `/sandbox/<image|video|text>`; without a
  kind it follows the live session (the image/video one first). It mounts
  `GenerationPanel` without a project (so `scratch` still derives from the
  missing project) and `AssistantPanel` with `freePlay`.
- `pages/projects/ModelPicker`, cut out of `WorkspaceHeader`, serves both the
  workspace's model bar and the Sandbox; it takes the launching project and a
  `confirmReplace` the Sandbox uses to ask before stopping a project's model.
  `ModelPicker.newConfiguration` is the modal both open from the select's
  last entry.
- Leaving: a capture-phase click listener on `window` asks before an in-app
  link leaves `/sandbox` (it runs ahead of frontroute's `LinkHandler`), and
  `beforeunload` covers reload and close. Unmount sends `ClearScratch` and
  `AssistantService.clearScratch()`, which empties the set-aside scratch chat
  if a workspace has bound its own first.
- The Models page lost its session takeover (`generationElement`,
  `assistantElement`, `activeSessionKey`, `showList`); **Launch** launches and
  opens the Sandbox on the architecture's kind (`SandboxPage.kindOf`). Its
  cards show *Running in* the Sandbox or the project, with **Open** and
  **Stop**.
- The gallery's reuse and *Just try a model* open the Sandbox, staging for
  the assistant opens its Text tab. A scratch result offers no *Ask the
  assistant*: the Sandbox has no chat beside its images.
- Keep is labelled **Save** now; the backend (`Generation.scratch`,
  `outputs/scratch/`, `ScratchRoutes`, `ScratchGenerations`) is spec 22's,
  unchanged.
