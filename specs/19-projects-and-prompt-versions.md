# 19 — Projects and prompt versions

**Status:** done
**Depends on:** 08 (the generation panel), 12 (reuse, the gallery), 21 (the chat panel)

A **project** is one thing the user is trying to make. It holds the prompt as
a list of versions and every generation made from them, across whatever
models they were tried on, so "which recipe on which model gave that" is a
recorded fact. The workspace puts the whole loop on one page: versions, the
generation panel, the assistant. Projects are what drift is for: `/` shows
them.

## What it does

- **Projects page**: a grid of tiles, each led by the project's cover (its
  newest result, or a chosen one), then the name, kind, version count and a
  brief clamped to two lines. A form creates one with a name, a brief, the
  kind ([`31`](31-project-kinds.md)) and an NSFW flag. NSFW projects are
  hidden unless "Show NSFW projects" is ticked, with a count of what is hidden.
- **Deleting a project** asks twice: the project and its versions always go;
  its images only when the second confirmation says so.
- **New Project** opens `NewProjectModal` (name, brief, kind, NSFW); a created
  project opens its workspace. With none yet (`ProjectService.projectsLoaded`
  and an empty list) the page shows an invitation in its middle — what a
  project is, **Create a project**, **Just try a model** → Models. The gallery
  shows the same invitation when it has nothing, and a one-line suggestion
  with the button while no project exists (`12`). Under the invitation,
  `Showcase` lays out eight SFW pictures, each a different style, made with
  the Krea 2 Turbo starter at 1024² and shipped as 640px WebP (~250 KB) in
  `frontend/public/assets/showcase/`: the only static path the backend serves.
- **Workspace** `/projects/{id}`:
  - Laid out by what is loaded: no image model and no version → the model
    bar alone, prominent, in the middle of the page; versions and no image
    model → the bar on top, the version history at `Size.Large` as the
    page; an image model live → the bar on top, the generation panel, the
    history at `Size.Small` under its result (`ProjectBinding.history`).
    Until the project has loaded it is not taken for an empty one.
  - Header: name and brief editable in place, the NSFW flag, the kind.
  - Model bar: two pickers — the image (or video) model and the assistant —
    listing that tool's configurations with the live one marked; picking
    another stops the live session and launches the pick. The live image
    session's Log/Restart/Stop (`GenerationPanel.sessionControls`) sit beside
    its picker, the panel drawing no header of its own in a project; the
    chat drawer's toggle beside the assistant's.
  - **+ New** beside each picker opens `NewRunConfigurationModal` (the run
    configurations page's own), its architectures narrowed to the picker's
    tool and the project's kind. The created configuration is picked as a
    select change would — launch, or download and wait — once
    `LaunchPrerequisites.unsettled` no longer lists it: a model registered
    with it has no cache state yet, which would otherwise read as nothing
    missing.
  - `VersionHistory`, one component at two sizes: newest first, a row per
    version — number, note, configuration, **use this recipe** (or
    *selected*), prompts in the tooltip — then its results, each with
    **Ask** (stages it for the assistant), **🖼** (make it the cover; lit on
    the cover, click again to clear), and **🗑** (deletes the whole
    generation, naming the batch size). Clicking a result opens the
    gallery's full detail view. **use this recipe** selects the version, and
    seeds a live panel. A **compare** link diffs any two versions' prompts.
  - The assistant is a drawer on the right, shrinking the body rather than
    covering it, kept mounted while closed; it opens when an assistant
    launches. Its proposal card's **Apply to form** fills the panel's prompts
    in place; **Apply and run** also submits, and is disabled until an image
    session is ready.
- **Every generation creates a version.** A submission whose recipe equals
  the selected version's (same configuration; seed and input images aside)
  belongs to it; anything else appends v(N+1) with the selected version as
  parent and a note naming what changed ("changed prompt, sampling; on
  krea2-turbo"). A submission whose prompts equal the last applied proposal
  is of assistant origin.
- **Seeding from a version is best effort**, through the gallery's reuse: on
  the same configuration field for field, LoRAs included; on another, the
  prompts, size and seed carry over, LoRAs and model-specific numbers take
  the target's defaults, and the notice names what was dropped. The seed is
  rolled anew.
- In an SFW project the LoRA picker starts with NSFW off; in an NSFW project
  on.
- The gallery shows a project badge on cards and a project filter.

## Shape

- `shared/Projects.scala`: `Project(id, label, brief, createdAt, lastUsedAt,
  versions, nsfw = false, kind = Image, cover: Option[ProjectCover])`;
  `PromptVersion(id, number, note, parentId, origin: Manual | Assistant,
  createdAt, runConfigurationId, kind: "img_gen" | "vid_gen",
  imageParameters | videoParameters)`; `ProjectCover(date, fileName,
  mimeType)`. Snake case like the sidecars. Version ids are
  `<projectId>-v<n>`.
- One document per project in `~/.config/drift/projects/`.
- `Generation.projectId` / `promptVersionId` in the sidecars; derived
  entries inherit. A project's generations are a filter over history;
  deleting a project keeps its images unless the sweep is asked for.
- Endpoints under `/api/projects`: list, get, create, update (`mergeUpdate`:
  the stored `versions` and `lastUsedAt` are kept, never taken from the
  client), delete, `GET/DELETE …/{id}/generations`, `GET …/{id}/cover`
  (downscaled JPEG bytes, 404 when none, a video project's cover is the file
  itself), `GET/PUT …/{id}/conversation` ([`20`](20-prompt-assistant-conversation.md)).
- Submit carries `?project=&version=&origin=` beside the native body
  (`SubmitContext`); `backend/projects/ProjectManager.versionFor` decides on
  the recorded request (`sameRecipe` blanks seed and inputs, `describeChange`
  diffs fields by name). A vanished project refuses the generation before it
  reaches sd-server.
- `backend/projects/ProjectCovers` walks the outputs once for all projects,
  newest day first, memoised against the newest day directory; a chosen cover
  is used while its file exists and is of the project's kind.
- Frontend: `pages/projects/{ProjectsPage, ProjectWorkspacePage,
  WorkspaceHeader, VersionHistory, WorkspaceSessions}`,
  `services/ProjectService` (folds the status socket's completions, reloads
  projects when a pushed generation names an unknown version).
  `GenerationPanel` takes a `ProjectBinding` and exposes `applyVersion`,
  `applyProposal`, `applyProposalAndRun` and `sessionControls`; `pages/gallery/GenerationDetailHost` serves both the
  gallery and the workspace, parameterized by the pool it resolves in.

## Notes

- **Selecting a version and seeding the form are two acts.** The selection
  follows a version that was *appended* by a run (so the next run compares to
  what just ran) but never a plain project reload; the form is seeded only
  when the panel appears and when a version card is clicked. A generation
  must not move the form.
- **A recipe must round-trip or every run appends a version**: a recorded
  request with no `hires`/`vae_tiling` block means off, not defaults; a LoRA
  whose NSFW flag moved its folder resolves by the id in its path; pending
  LoRA selections are sent as recorded. LoRAs are resolved from values in
  hand inside the seeding transaction, never read back from a `Var` just set.
- "Reuse these parameters" on an older result moves the selection to that
  result's version.
- The workspace's result column filters the session's generations by project
  id; the session outlives any one workspace.
- Inline editors keep their draft in a `Var` the input is `controlled` by,
  not in the signal that builds them (else focus is lost per keystroke).
- The newest version is read off the emitted project signal, not a mirror
  `Var` (which lags one propagation). A `Var` declared after the `val` that
  reads it is null at construction.

## Post-v1

- Editing a version's note; comparing two versions' images at one seed;
  forking a project from a version; favourite generations.
