# Projects

A project is one thing you are trying to make. It keeps every prompt you
tried as a numbered version, and every image or video those versions made,
whatever model ran them. Projects are the first page of the sidebar and what
`http://localhost:4321` opens on.

## Creating a project

- Give it a name and, optionally, a brief: what you are making, in your own
  words. The assistant reads the brief.
- Choose what it **makes**: images (the default), videos or texts. The
  workspace only offers models of that kind. You can change it later in the
  workspace header. A text project is a conversation; see
  [Text projects](#text-projects).
- Tick **NSFW** for a project you want hidden by default. The list hides NSFW
  projects unless "Show NSFW projects" is ticked, and says how many are
  hidden. The gallery hides their results the same way (see [gallery.md](gallery.md)).

Each project shows as a tile led by its cover: its newest result, or one you
chose. Deleting a project asks twice: the project and its versions always go;
its images only if you say so.

## The workspace

Open a project to get its workspace, in three columns.

- **Header**: the name and brief, editable in place; the NSFW flag; the kind;
  and two pickers. The **Image model** (or Video model) picker lists your run
  configurations of that kind and launches the one you pick, stopping the
  current one if needed. The **Assistant** picker does the same for chat
  models (see [assistant.md](assistant.md)). A model that is live but of the
  other kind is still listed, marked as such.
  **+ New** beside each picker creates a run configuration without leaving the
  project: the same form as the run configurations page, offering only the
  architectures that make what the project makes (a chat model for the
  assistant). Once created it is picked, as if chosen in the list: it
  launches, or its weights start downloading and it launches when they are
  on disk.
- **Versions**, on the left, newest first. Each card is the version number
  plus thumbnails of what it made; hover for its note, origin, model and
  prompts. Click a version to load its recipe into the form. **compare**
  shows the prompts of any two versions side by side as a word diff.
- **Generation panel**, in the middle: the same form as everywhere else (see
  [generating.md](generating.md)). Under it, the project's results grouped by
  version. Each result has three buttons: 🤖 **Ask** sends it to the
  assistant, 🖼 makes it the project's cover (click again to clear), 🗑
  deletes the whole generation. Click a result to open its full detail view,
  with upscaling and redraw.
- **Assistant**, on the right: the project's conversation.

## Versions

You never save a version by hand. Every generation makes one:

- If the recipe equals the selected version's, on the same model, the
  generation joins it. Only the seed and the input images are ignored, so a
  re-roll stays in its version.
- Anything else appends the next version, with the selected one as parent and
  a note saying what changed, such as "changed prompt, sampling".
- A version whose prompts came from an applied assistant proposal is marked as
  the assistant's.

A version is the whole recipe: prompts, size, sampling, LoRAs and their
strengths, hires, tiling, and the model it ran on.

## Loading a version into the form

Click a version card. On the model it ran on, every field comes back as it
was, LoRAs included. On another model, the prompts, size and seed carry over
and the rest takes that model's defaults; a notice lists what was dropped,
such as a LoRA the other model cannot load. The seed is rolled anew so runs
do not all share one.

The form is seeded only when you click a version or switch model. Generating
never changes what you typed.

## Projects in the gallery

Gallery cards carry a project badge, and the gallery toolbar has a project
filter. "Reuse these parameters" on a result inside the workspace also moves
the selection to that result's version.

## Text projects

A text project is a conversation with a chat model, kept with the project.
Its workspace has one picker, **Chat model**, and the chat below it. There
are no versions, no generation form and no results.

The model gets your messages and nothing else: no system prompt, no brief,
no drift rules. Replies are never read as prompt proposals. The conversation
is saved after every message. **Compact** replaces it with a summary when the
context fills. **Restart** archives it and starts over. With a vision model
you can attach images or videos. See [assistant.md](assistant.md) for the
chat itself.
