# 55 — Architecture consolidation: one piece of code per shape

**Status:** planned 2026-10-11, from a review of the whole code — nothing built but step 0
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

0. **Small corrections** (done 2026-10-11, uncompiled): the prompt library
   mounted once; the built-in rule and the `/api` prefix written once; the
   stale texts on MiniMax H3's inputs and on the runner's progress (44).

### Across the modules

1. **A protocol module shared by the backend and the runner** — begun
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
   - Left: the capabilities document as a type, carrying what the backend
     infers today (the largest tile, whether references keep their size, a
     family's size multiple, frame rate and default steps) — it holds the
     form's sampling types, so it moves with them, and the module becomes one
     the page builds on too; the backend's calls built from the endpoints
     (8); the chat server's flags and requests.
2. **The runner's own progress report** on the job document (44), replacing
   the bar patterns kept in step on both sides.
3. **One ffmpeg wrapper per side**, a check at startup shown with the
   machine's status, and the README and getting-started saying when it is
   needed.
4. **Reference data checked whole**: every id named by a starter, a
   migration's map or a retired built-in resolves. (The tag fix-up at
   startup is gone, 2026-10-11.)
5. **Entity fields without defaults** — done 2026-10-11 (38): the stored
   types carry none, a stored record says every field and does not load
   without one, the reference files are spelled out, and the migrations
   written so far are retired.
6. **Paths said once**: the page's texts and the launch preview read the
   locations from the backend.

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
9. **One download queue**, for models, LoRAs and upscalers, with one job
   type and one status topic; runtime installs reuse its queue and cancel.
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
    history is told of a deletion by a function, not by the manager. One
    cycle is left in the backend: `download` and `huggingface`, the generic
    downloader and the models' download manager sharing a package (9).
12. **The tile loop runs every tiled pass**, the edit's carry included, and
    picks its server in one place; a running job is one record instead of
    six maps.
13. **Collections and settings documents declared once**; a run
    configuration with its architecture and models resolved by one function.
14. **Named failures** in the shared endpoints: the site failed, refused with
    a reason, not found.
15. Kinds and operations as enumerations; `Api.scala` and `PostProcess.scala`
    split by entity; tests for the launch refusals, the job state machine,
    the job client and the download queue as each is rewritten.

### Frontend

16. **Services mounted once, in the shell** — done 2026-10-11: every
    service's pipelines are mounted by `AppShell` and by no page; a service's
    `effects` are its own (none brings another's along); a page keeps its
    loads (`LaunchPrerequisites.loads`). Seen in a headless browser on every
    page: lists load. Left: the search services of the browsers, mounted by
    the browser that shows them.
17. **One store of generations**, by session, by project and by day derived
    from it; the global downloads derived from the services' own lists.
18. **Packages by what they are** — done 2026-10-11: under `pages` only what
    a route opens (projects, sandbox, gallery, models, model cache,
    settings); beside it `gallery` (the cards, the picker, the detail view
    and its tasks), `generate` (the form and its result), `assistant`,
    `workspace` (what a project's page and the Sandbox share) and
    `models.architectures` / `models.inference` (the two tabs of Models). No
    page imports another and no package imports one that imports it. Two
    rules that pages lent each other are the project's own now
    (`Project.versionFor`, `ProjectKind.of`).
19. **A form field declared once**: its value, how it is read from defaults,
    a record and a LoRA, how it is written to the request — the layers shown
    as "the user's, else the LoRA's, else the defaults" instead of writes
    guarded by a list of touched fields.
20. **Lists that tick are not rebuilt by it** — done 2026-10-11 for two of
    three: the run configurations' cards read the sessions' state without
    what ticks (`SessionService.states`), and a job's row is rebuilt alone
    (split by id). Left: the open detail, still keyed on the whole record, so
    a move to a project rebuilds it — it takes the record as a value all the
    way down.
21. `GenerationPanel`, `GalleryPage` and `AssistantService` split by
    responsibility; the gallery's filters written once for the page and the
    picker; one generic entity service; the stylesheet split by page.

### Runner

22. **Buffers reused** — first half done 2026-10-11: `HipOps` keeps released
    buffers for the next allocation of their size (2 GiB at most, the oldest
    given back first, all of them when the device is full;
    `-Ddrift.hip.spare=<MiB>`, 0 for none), and finds a buffer by its address
    instead of searching a list. No caller changed. Measured on SeedVR2 3B, a
    1024² picture to 2048², one run each way: the same file to the byte,
    decode 31.9 → 28.1 s, encode 15.4 → 14.3 s, system time 8.2 → 3.9 s. Not
    measured: a long LTX decode under memory pressure, which is what it is
    for (`bugs/README.md`, bug 4). Left: one scope type for what a step or a
    frame allocates, in place of the lists kept by hand.
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
25. **One sampling loop** for the flow models: schedule, first step of an
    img2img, guidance, preview, mask, progress; a pipeline keeps its prompt
    encoding, its noise and its decode.
26. **Transformer pieces written once**: work buffers, a linear with its
    bias and LoRA site, the timestep embedding, the joint block; attention of
    a non-causal model through one wrapper.
27. **VAEs on one kit**: the causal video layers shared by Wan, SeedVR2,
    MiniMax H3 and LTX; tiling offered to every decode.
28. **Families declared in one list**: how a file is recognised, what it
    needs, how it opens. A file no family recognises is refused. The server
    holds one kind of pipeline.
29. **One progress object** handed to a generation: phases, bars, load
    timings — every phase that lasts says so.
30. `ImageServer` split (queue, capabilities, request parsing, results); one
    flag parser for the image and chat servers; `Ops`, `HipOps` and `CpuOps`
    split by concern; one weight loader; a test listing the GPU ops that no
    hand-run comparison covers.

## Order

Small and independent first: 7, 16, 20, 22 (first half), 24. Then the
protocol module (1), which 2, 8 and 28 build on. Then the merges, one at a
time: 9, 10, 11, 12, 17, 25, 26, 27. File splits last: they conflict with
everything else while it moves.

## Not taken

- The container image is kept (`Dockerfile`): it serves a later publication.
  It carries neither ffmpeg nor the runner's kernels today.
