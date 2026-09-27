# 29 — Code structure: facades, components, rebuild rules

**Status:** done in code (the later splits compiled and formatted, exercised
only through the ordinary walk-through)
**Depends on:** nothing

No file should be so large that it hides a render bug. This spec is the map
of what the big files became, and the rules that keep them that size.

## Rules

- A component is a plain class with constructor attributes and a
  `lazy val element`, never `object X { def apply }`. One feature, one file;
  shared code over parallel copies.
- **State stays where it is read.** A component gets the `Var`s or signals
  it needs from its parent and never grows a second copy of the parent's
  state.
- **Never rebuild a large component from a frequently changing signal.**
  `child <-- signal.map(build)` recreates the DOM, the images and every inner
  `Var` on each emission. Build once per identity (`split`, `distinct` on a
  key) and pass the changing parts in as signals. Session progress ticks
  several times a second; nothing big may hang off it.
- Roughly 600 lines, or a render bug, is what earns a file a split — but the
  test is single responsibility, which is what François judges by
  (2026-09-23): a file worth reading says one thing. A class with a dozen
  `private def someUi` is several components sharing a file.

## Backend facades

Each keeps its public API over package-private pieces:

- `postprocess/PostProcessManager` over `PostProcessJobs`, `EsrganUpscale`,
  `PidUpscale`, `Redraw`, `Edit`, `PostProcessImages` (+ `TileBlending`; the
  geometry moved to `shared/.../Tiling`). A tiled job is four of them, one
  responsibility each: `TiledJobs` decides what a job is and records it,
  `TileRun` is one run of its tiles — server, tiles, blend — `TiledArea` what
  the job works on (the fields checked, the selection, the template, the window
  and its tiles), `TileWindow` one tile's exchange with the model (what it is
  handed, what becomes of what it returns). A paused job carried on later
  (`40`) is another `TileRun` over the same tiles.
- `lora/LoraManager` over `LoraInstalls` (what a request ticked, from each
  site or a folder), `LoraPlacement` (which entity those files land in),
  `LoraDownloads` (the queue and the fetching), `LoraAdoption` (a folder the
  store holds and no entity claims), `LoraSidecars` (what sits beside the
  weights); the manager itself lists, edits, moves and deletes.
- `runtime/RuntimeManager` over `RuntimeInstalls`, `RuntimeArchives`,
  `TheRockPairing`, `RuntimeValidation`, `RuntimeSelections`, `RuntimeCleanup`.
- `sdserver/GenerationManager` over `GenerationRegistry`, `GenerationFiles`,
  `GenerationMonitor`, `GenerationSubmissions`, `ScratchGenerations`.
- `session/SessionManager` beside `SessionSettings`, `SessionOutput`,
  `ServerProcesses`, `LaunchArguments`, `JobServers`.

## Frontend split

- The gallery's picture: `GenerationMediaViewer` draws it — the media, the
  overlay, the tile grid, the batch strip — and `PicturePointer` decides what a
  press, a move and a release mean, since all of that lives in mouse events
  and none of it is drawing.
- Gallery detail: `pages/gallery/GenerationDetailHost` builds one
  `GenerationDetail` per opened (generation, output) and feeds sessions,
  configurations, derivatives and jobs in as signals; the detail is
  `GenerationMediaViewer`, `GenerationParameters`, `GenerationLineage`,
  `PostProcessSection` (with `EsrganUpscaleRow`, `PidUpscaleRow`,
  `RedrawRow`, `PostProcessJobList`, `LiveSessionNotice`); the tiled tasks
  share `TileAreaFields` (the tile, window and margin fields, the grid and the
  plan read off the box). What is on screen
  and what is drawn over it — the output shown, the box, the grid and its
  offset, the tiles a job is painting — is `DetailPicture`, one chain of
  signals in one file: in the detail's own body, among its view fragments, a
  new link landed above the one it read and was `null` when the class was
  built (François, 2026-09-23). A component's derived signals belong with the
  ones they derive from, in a file small enough that the order is visible.
- Assistant chat: `pages/assistant/AssistantPanel` is the shell — the
  session's header, the system prompt, the switch offer, the context meter and
  the archive — over `AssistantTurns` (the conversation: every turn, its
  reasoning, its timings, its proposals) and `AssistantComposer` (the draft,
  the staged pictures, the conversation's controls). The draft belongs to the
  composer; the system prompt's switch is the panel's, since the panel places
  the editor.
- Every page has a package of its own: the Model cache page and its views
  (`pages/cache`: `ModelCachePage`, `OnDiskCacheView`, `ConfiguredModelsView`,
  `MissingModelsView`, `UpscalerSection`, `ConvertModelModal`,
  `ConversionJobList`) were loose in `pages/` until 2026-09-23.
- Generation form: `pages/generate/GenerationPanel` is the shell over
  `GenerationFormState`, `RecipeSeeding`, `GenerationSubmission`,
  `GenerationForm`, `CoreFields`, the `{Sampling, Inputs, Hires, VaeTiling,
  Guidance, Batch}Section`s, `CollapsibleSection`, `FormFields`,
  `ReuseNotice`, `GenerationResult`, `SessionLog` (the progress view is
  `components/LogProgressView`, shared with the gallery); the
  LoRA picker is `components/LoraPicker`.
- Settings: `pages/settings/SettingsPage` over `RuntimesSection`,
  `RuntimeRow`, `AddRuntimeModal`, `InstallRuntimeForm`, `AdoptRuntimeForm`,
  `TheRockBuildField`, `RuntimeInstallJobs`, `RuntimeOptions`,
  `AuthenticationSection`.
- Assistant: `services/AssistantService` with `ChatStreaming`,
  `ConversationStore`, `AssistantPrompts`.
- Workspace: `pages/projects/ProjectWorkspacePage` with `WorkspaceHeader`,
  `VersionsColumn`, `ProjectResults`, `WorkspaceSessions` (the picker reads
  only the live session's ids and status, so a progress tick does not
  rebuild the select).
- Browsers and cache: `components/CivitaiBrowser` with `CivitaiMedia`,
  `CivitaiTags`, `CivitaiModelAbout`; `pages/ModelCachePage` with
  `OnDiskCacheView`, `ConfiguredModelsView`, `MissingModelsView`.
- Left whole at about 600 lines: `AssistantPanel`, `AssistantService`,
  `Downloader`, `LoraManager`.

## Notes

- **Name shadowing**: `drift.shared.SessionProgress` exists. A frontend class
  of that name in a package importing `drift.shared.*` silently wins and
  breaks its callers; the view is `components/LogProgressView`.
- Audited and fine as they are: the gallery grid (keyed per tile), the run
  configurations page (`activeSessionKey` is the list of live ids), the model
  cache rows (rebuild only on reloads), the assistant panel's status tag
  (deduplicated).
