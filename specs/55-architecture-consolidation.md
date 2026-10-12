# 55 — Architecture consolidation: one piece of code per shape

**Status:** built but for what each step lists as left and what "Not taken" says; kinds as enumerations (15) wait for a migration of the outputs
**Depends on:** nothing; touches most of the code

The foundations hold: storage and migrations, the generic entity endpoints,
the shared tiling maths, one parameter resolution for the preview and the
launch, the runner's op contracts and fixtures. What the review found is one
kind of debt: the same shape written several times — per model family in the
runner, per kind of download or job in the backend, per page in the frontend.
Each step below replaces the copies by one piece with a parameter, and is
compiled and tested on its own before the next. Runner steps that change what
runs on the GPU are checked there (`gpuTest`), at night.

## Steps

0. **Small corrections** (done 2026-10-11): the prompt library
   mounted once; the built-in rule and the `/api` prefix written once; the
   stale texts on MiniMax H3's inputs and on the runner's progress (44).

### Across the modules

1. **A protocol module shared by the backend and the runner** — done
   2026-10-11 (`protocol`, package `drift.protocol`, built on by both):
   - **Done: the requests.** `img_gen`, `vid_gen` and `upscale` are types
     whose every field is optional — a field left out is the launch's, as
     sd-server reads it. The runner reads a request through them and refuses
     a field they lack by its name; a field it knows and does not read
     (`scheduler`, `eta`, `clip_skip`, `img_cfg`, skip-layer guidance, another
     output format) is said in the job's warnings when it is set. The
     backend's requests are checked against the same types by test, and its
     upscale requests are built from them. Seen live on Mage-Flow Turbo.
   - **Done: the job document and the routes** (`NativeJob`, `JobRoutes`) are
     the protocol's: the runner writes its answers through them — a
     submission's, a poll's with its preview, a result, a partial result —
     and the backend reads them. `JobRoutes.Version` is 1. Seen live.
   - **Done: the version.** The runner says its protocol version in its
     capabilities (`drift.protocol`), and a session or a job server whose
     runner says another, or none, fails at once with the reason and what to
     do (reinstall the runner).
   - **Done: the launch flags** of the image server (`ImageServerFlags`): one
     list the runner parses with, and a test that every flag the reference
     data gives a runner launch is one it takes.
   - **Done: the routes** (`JobEndpoints`): the job protocol as endpoints in
     the module, served by the runner as they are.
   - The chat server's flags are in the module (30); its requests are
     llama-server's own.
2. **The runner's own progress report** — done 2026-10-11 (44): the job
   document carries `progress` (`JobProgress`), which drift shows as it
   polls, for generations and for post-processing jobs. Left: the runner's
   log still draws sd-cpp's bars, and drift's patterns for them stay for
   sd-cpp.
3. **ffmpeg said where it matters** — done 2026-10-11: drift looks for it
   once at start and the Machine box says when it is not installed; the
   README and the getting-started page say what it is for — every video's
   still, and the runner's videos. Left: LTX's own call to it for its image
   conditions, beside the runner's video files.
4. **Reference data checked whole** — done 2026-10-11, by test: a built-in
   model is a checkpoint of some architecture's slot; a starter fills slots
   its architecture has, with models of the slot's family; an architecture
   the runner is offered for is of a kind the runner runs
   (`RunnerModelKinds`, the list the runner answers with); every flag such a
   launch carries is one it takes. The tag fix-up at startup is gone.
5. **Entity fields without defaults** — done 2026-10-11 (38): the stored
   types carry none, a stored record says every field and does not load
   without one, the reference files are spelled out, and the migrations
   written so far are retired.
6. **Paths said once** — done 2026-10-12: the backend says where it keeps
   its files (`GET /api/locations`, `FileLocations`: each folder as resolved
   at startup, the home written `~`), and the page's texts and the launch
   preview read them there (`LocationsService`) — a relocated drift names
   the folders it really uses.

### Backend

7. **One child-process mechanism** — done 2026-10-11 (`process/ChildProcesses`):
   every process is started there and known while it lives, so stopping
   drift stops them all, an sd-cli included; one way to stop a child
   (asked, then killed after the grace period); a long run followed to its
   end with its output read in the background, so a helper the child leaves
   holding its output cannot hang the run; a short tool run for its answer
   through a file, with a time limit. Sessions, job servers, the sd-cli of an
   upscale and of a conversion, runtime validation, archive unpacking and
   video stills all go through it. Left: sessions and job servers still
   write their log header and wait for readiness each in their own way.
8. **One client of the job protocol** — done 2026-10-11
   (`nativejobs/NativeJobClient`): submit, follow, cancel and capabilities,
   used by a generation and by the jobs a post-processing drives. One policy
   for following a job — a silent server is given time, five failures in a
   row are not, a document is counted only once it has been read and taken, a
   dead or forgetful server ends it at once — tested against a stub server.
   Run through drift on a copy of the configuration: a session on the
   runner, a generation of two pictures, a SeedVR2 upscale through a job
   server.
9. **One download queue** — done 2026-10-11 (`download/DownloadQueue`): a
   transfer's turn, its progress, its cancel and its end are written once and
   tested; the models', the LoRAs' and the upscalers' managers each say only
   what a transfer fetches and where to. The models' manager is in `models`,
   so `download` is the generic part and the backend has no import cycle.
   The three kinds of job the page reads are one shape with a key each
   (`TransferJob`), and the downloads panel lists them by one function.
10. **A package per site** — done 2026-10-11: `civitai` (its client, which
    also says where a file is fetched and what token goes with it — to
    Civitai's hosts only) and `huggingface` (its calls, its examples, its
    downloads) beside `modelscope`; what the sites share (descriptions
    cleaned, the root reason of a failure) in `sites`. The routes only
    delegate, and no package imports them. Left: a failed search or detail
    still answers nothing rather than why (14).
11. **`sdserver` split by what it holds** — done 2026-10-11: `outputs` (the
    files, the history, the imports, the previews, and `Outputs`: the layout
    of the outputs root and its media types), `generation` (a generation's
    life: the manager, the registry, the monitor, the submissions, scratch,
    moves between projects) and `nativejobs` (the job protocol's client, the
    requests, the reference sizing); the picture helpers are in `images`. The
    history is told of a deletion by a function, not by the manager.
12. **One tile pass** — done 2026-10-12 for what was written twice: the
    edit's carry runs its tiles through `postprocess/TilePass` (cut, made,
    judged, corrected, blended; tested), and a returned tile is judged by
    one rule there for every pass — a black tile is refused in the carry
    too. A running job is one record (`RunningJob`: what a cancel reaches,
    its log, its report, its picture, its work) instead of six maps, removed
    only by the run that put it. Left: a tiled job of its own (`TileRun`)
    keeps its own loop, which holds its tiles on disk to be paused and
    shown. Every kind of job gets its server from one place
    (`ChosenServer`): a ready session's, else one of its own.
13. **Collections and settings documents declared once** — done 2026-10-11
    (`storage/Collections`): each collection is a value carrying its folder
    and how its records read, and the generic entity endpoints take one; each
    settings document carries its name and what it starts as. A run
    configuration is resolved for a launch by one function
    (`LaunchArguments.launch`: its architecture, its argv, or what stops it
    in one sentence), which sessions, job servers and the tiled jobs' checks
    all call.
14. **Named failures** — done 2026-10-11 for the sites: a search or a
    model's detail on Civitai or HuggingFace that fails answers 502 with the
    reason — the site's status, its own words, or that it could not be
    reached — which the browser shows; a model's install or download says the
    same reason instead of "or the lookup failed". Seen through drift. Deleting a
    cached file says which file and why it stays, where it answered `false`
    (2026-10-12). Left: the bare 404s of the endpoints a picture is fetched
    by, which no one reads.
15. **Files by entity, and the tests** — done 2026-10-12 but for the
    enumerations: `Api.scala` and
    `PostProcess.scala` are one file per entity (`Architectures`, `Models`,
    `RunConfigurations`, `HuggingFaceModels`, `Cache`; `UpscaleRequests`,
    `Redraw`, `Edit`), `Api.scala` keeping what every file builds on. Left:
    kinds and operations as enumerations, which changes what a generation's
    sidecar holds and so comes with a migration of the outputs. The launch
    refusals and a post-processing job's states are tested; the page's four
    entity services are one (`EntityService`), each saying its endpoints,
    its words and where a saved entity goes in its list.

### Frontend

16. **Services mounted once, in the shell** — done 2026-10-11: every
    service's pipelines are mounted by `AppShell` and by no page; a service's
    `effects` are its own (none brings another's along); a page keeps its
    loads (`LaunchPrerequisites.loads`). Seen in a headless browser on every
    page: lists load. Left: the search services of the browsers, mounted by
    the browser that shows them.
17. **One store of generations** — done 2026-10-12 (`GenerationStore`):
    every generation the page knows is held once, by its id; the gallery's
    days, a project's results and a session's panel are views of it, so a
    generation that completes, moves to a project or is deleted changes in
    one place. Each service keeps only what is its own: which days and
    projects are loaded, which generations a session lists. The downloads
    panel lists the three kinds of transfer by one function (9).
18. **Packages by what they are** — done 2026-10-11: under `pages` only what
    a route opens (projects, sandbox, gallery, models, model cache,
    settings); beside it `gallery` (the cards, the picker, the detail view
    and its tasks), `generate` (the form and its result), `assistant`,
    `workspace` (what a project's page and the Sandbox share) and
    `models.architectures` / `models.inference` (the two tabs of Models). No
    page imports another and no package imports one that imports it. Two
    rules that pages lent each other are the project's own now
    (`Project.versionFor`, `ProjectKind.of`).
19. **A form field declared once, in layers** — done 2026-10-12 for the
    sampling fields: each is one declaration (`generate/SamplingField`) of
    how its text is read from a block of sampling parameters — a session's
    defaults, a recorded request, the LoRAs' settings over the defaults —
    and how what the form holds is written to the request; seeding, reuse,
    the LoRAs' layer and the submission all go through it. Each holds the
    user's value when they gave one, else its default (`LayeredField`), so a
    default that changes shows in a field left alone and never replaces what
    was written, with no list of touched fields kept beside the form. Left:
    the other fields (size, seed, hires, VAE tiling, the inputs), each still
    said in the seeding and in the submission.
20. **Lists that tick are not rebuilt by it** — done 2026-10-12: the run
    configurations' cards read the sessions' state without
    what ticks (`SessionService.states`), and a job's row is rebuilt alone
    (split by id); the open detail is keyed on its generation's id and the
    output shown, and reads what can change under it — its project — from
    the record as a signal, so a move rebuilds nothing (2026-10-12).
21. **The page's large files** — done 2026-10-12: `GenerationPanel` is the
    wiring of what it held (`FormAssistantLink`, `SessionLoras`,
    `SessionFormSeeding`, `GenerationSessionControls`,
    `GenerationResultColumn`); `GalleryPage` likewise (`GalleryFilters`,
    `GalleryToolbar`, `GallerySelection`, `GalleryDaySection`,
    `GalleryImport`, `GalleryLabels`); `AssistantService` is a facade over
    its context, its transcript, its replies and its session's properties.
    The gallery's filters are one state for the page and the picker
    (`GalleryFilterState`). The stylesheet is one file per area under
    `frontend/styles/`, imported in the order the rules had.

### Runner

22. **Buffers reused** — done 2026-10-12: `HipOps` keeps released
    buffers for the next allocation of their size (2 GiB at most, the oldest
    given back first, all of them when the device is full;
    `-Ddrift.hip.spare=<MiB>`, 0 for none), and finds a buffer by its address
    instead of searching a list. No caller changed. Measured on SeedVR2 3B, a
    1024² picture to 2048², one run each way: the same file to the byte,
    decode 31.9 → 28.1 s, encode 15.4 → 14.3 s, system time 8.2 → 3.9 s. Not
    measured: a long LTX decode under memory pressure, which is what it is
    for (`bugs/README.md`, bug 4). One scope type (`ops/TensorScope`) holds what a step, a
    frame or a model's buffers allocate, released together in the order
    made, in place of fifty lists kept by hand; the seven that hand single
    tensors on or release them early keep their list.
23. **The runner on the backend's stack** — done 2026-10-11: its image
    server and its chat server are tapir endpoints on Netty, on ox; the jobs'
    worker, the model's loading and the reading of a tool's two streams are
    forks of a scope. No thread pool and no thread of its own is left, but
    the one the JVM asks for at shutdown. Seen live on both: jobs, previews
    and refusals on Mage-Flow Turbo, from the packaged jar too; a chat whole
    and streamed, and a stream dropped by its reader ending the reply.
24. **No cycle** — done 2026-10-11: `lora` (the reader, the updates, the
    report) and `weights` (`WeightSource`, `WeightNames`) are packages of
    their own under the models, the cache descriptor is with the ops, the
    random draws with the tensors. `models` imports nothing of `diffusion`,
    `ops` nothing of `state`.
25. **One sampling loop** — done 2026-10-11 for five image families
    (`diffusion/FlowSampler`): Krea 2, FLUX.2, Mage-Flow, Qwen Image 2.1 and
    LLaDA-Image give their model's velocity and keep their prompt, their
    noise and their decode; the steps, the guidance, an img2img's start, the
    mask, the preview and the progress are written once. LLaDA Turbo's
    stochastic step is the loop's with another way to move. Checked through
    drift against pictures made before: the same file to the byte for Krea
    2, FLUX.2 klein, Mage-Flow Turbo and Qwen Image 2.1 — plain, guided,
    shifted, img2img and masked — and for HiDream O1 and GRN, which were not
    touched. LLaDA cannot be checked so (`bugs/README.md`, bug 5). Left:
    Nucleus (its guided step is its own), HiDream O1 (UniPC), the video
    families. Krea 2 takes a mask now, which the loop gave it for the asking
    (run through drift, not looked at).
26. **Transformer pieces written once** — done 2026-10-12 for what is the
    same: the timestep's
    sinusoid is one function (`models/TimestepSinusoid`) with what the
    families differ by as parameters — the scale, which of the sines and the
    cosines come first, the largest period, the precision of the
    frequencies, and for the models released in BF16 what is rounded on the
    way — used by every family. Rounding to BF16 on the host is one object
    (`tensor/Bf16`). The same pictures to the byte through drift for Krea 2,
    FLUX.2, Qwen Image 2.1, GRN, Mage-Flow and HiDream O1; and for the five
    that moved last (HiDream O1, LTX, Mage-Flow, Nucleus, LLaDA) the same
    bits as their own formulas gave, over two thousand timesteps. A linear
    layer with its bias and its LoRA site is one type (`models/Affine`),
    used by fifteen models; the buffers a model keeps for a token count and
    remakes when a call needs more are one mechanism (`WorkBuffers`,
    `GrownBuffers`), used by seven; full attention with rotary positions is
    written once for the families that share its order exactly
    (`RotaryAttention`: FLUX.2 and Krea 2, GRN and LLaDA). The same pictures
    to the byte through drift for the six families that repeat one. Left:
    the joint block, and the attention of the families whose order of
    operations is their own.
27. **VAEs on one kit** — done 2026-10-12 for what is the same
    (`models/VaeLayers`): what a causal video VAE does to one frame — its
    pixels as rows, the norm and SiLU, a 1×1 convolution, the residual block
    around its two convolutions over time — is `FrameLayers`, with a form
    for the encoders that hold a whole clip (MiniMax H3's, LTX's); one
    output frame of a convolution over time is one function (`summed`) for
    Wan, SeedVR2 and MiniMax H3, each giving its order of terms; the frames
    a convolution keeps from one chunk to the next are `FrameHistory` for
    Wan and SeedVR2. Left as their own: LTX's convolution over time (one
    reused part, another order), each VAE's first chunk, and tiling, which
    only SeedVR2's decode has.
28. **Families declared in one list** — done 2026-10-11
    (`Pipeline.Families`): the upscaler, the video families and the image
    families each say how their file is recognised and how they open from
    the launch's files (`ModelFiles`), in the order they are tried; the file
    is read once to find its family; a file none recognises is refused with
    their names, where it was opened as Krea 2, and a ControlNet for a family
    that takes none is refused by the family's name. A pipeline says its
    family, its job mode and its default steps (`Pipeline`). Tested on every
    family's tiny model. Each family names the model kinds it runs: the
    runner reports those, the help line names the families, and a test holds
    them to the protocol's list.
29. **One progress object** — begun 2026-10-11: `diffusion/ProgressReport`
    is where a generation says its image, stage, pass and bar, to the log
    and to the job at once. Left: handed to a generation rather than reached
    as an object, and the load timings through it.
30. **The runner's large files** — done 2026-10-12: `ImageServer` is the
    wiring of the routes, and what it held is one class each — the queue
    and its worker (`JobQueue`), the launch's defaults (`LaunchDefaults`),
    the capabilities document (`ServerCapabilities`), the requests read per
    kind of job (`ImageJobs`, `VideoJobs`, `UpscaleJobs`), the results made
    (`JobResults`). One reader of a command line (`LaunchFlags.read`) for the
    image and the chat server, each with its own list of flags in the
    protocol (`ImageServerFlags`, `ChatServerFlags`). `Ops`, `HipOps` and
    `CpuOps` are one trait per concern each — memory, elementwise,
    normalisation, linear, experts, attention, convolution, layout,
    sequence — over a base that holds what they share (`HipBase`: the kernel
    modules, loaded in the order they were, the buffers, the launches), and
    closing the GPU backend closes every kernel module. Weights are read
    through `WeightSource` everywhere, and a text encoder's tokenizer by one
    function (`GgufTokenizer.of`). The operations no test compares on the two
    backends are a list a test holds (`OpsComparisonCoverageTests`, 20 of
    74): an operation added without a comparison fails it.

## Order

Small and independent first: 7, 16, 20, 22 (first half), 24. Then the
protocol module (1), which 2, 8 and 28 build on. Then the merges, one at a
time: 9, 10, 11, 12, 17, 25, 26, 27. File splits last: they conflict with
everything else while it moves.

## Not taken

- The backend's calls to a job server are not built from the endpoint
  values: the paths (`JobRoutes`) and the documents (`NativeJob`) it uses are
  already the protocol's, what it posts is the form's request, held to the
  protocol's by test, and its one HTTP client carries the policy of a poll —
  what a silence means, what a failure counts for — which a generated client
  would have to be given again.

- The capabilities document does not carry what drift knows of a runner
  before it runs — the largest tile, whether references keep their size, a
  family's size multiple: a job is planned, and the page lays its tiles,
  before any server of it exists, so these stay with the runtime's engine
  and the architecture. The document's head is typed and versioned
  (`CapabilitiesHead`).

- Runtime installs keep their own queue: an install is several steps — a
  TheRock tree, a release, its unpacking, a validation — with states of its
  own, not one transfer waiting its turn.
- A delete that answers whether there was something to delete is an answer,
  not a failure: those endpoints keep their boolean.

- The container image is kept (`Dockerfile`): it serves a later publication.
  It carries neither ffmpeg nor the runner's kernels today.
