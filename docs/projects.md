# Projects

A project is one thing you are trying to make. It keeps every prompt you
tried as a numbered version, and every image or video those versions made,
whatever model ran them. Projects are the first page of the sidebar and what
`http://localhost:4321` opens on.

## Creating a project

**New Project** opens a window for it; with no project yet, the page explains
what one is and offers **Create a project** in the middle, beside **Just try a
model** for trying a model without one, over eight pictures in different
styles made with Krea 2, one of drift's starter models (hover one for its
prompt). Creating it opens its workspace.

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

Open a project to get its workspace. What it shows depends on what is
loaded:

- **A new project with no model loaded**: the two pickers, in the middle of
  the page. Pick an image model to start.
- **A project with versions and no image model loaded**: the pickers at the
  top, and the whole page for the version history, with large thumbnails.
- **An image model loaded**: the pickers at the top, the generation form on
  the left, the result on the right, and the version history under the
  result, with smaller thumbnails.

The parts:

- **Header**: the name and brief, editable in place; the NSFW flag; the kind.
- **Model bar**: two pickers. The **Image model** (or Video model) picker lists your run
  configurations of that kind and launches the one you pick, stopping the
  current one if needed. The **Assistant** picker does the same for chat
  models (see [assistant.md](assistant.md)). A model that is live but of the
  other kind is still listed, marked as such.
  The last entry of each picker, **+ New … configuration…**, creates a run
  configuration without leaving the
  project: the same form as the run configurations page, offering only the
  architectures that make what the project makes (a chat model for the
  assistant). Once created it is picked, as if chosen in the list: it
  launches, or its weights start downloading and it launches when they are
  on disk.
  While a model runs, **Log**, **Restart** and **Stop session** sit beside
  its picker. Beside the assistant's (or a text project's chat model):
  **Show chat** / **Hide chat**, **Stop session**, the context size the
  server applied and whether it has **vision** — the chat itself keeps its
  room.
- **Generation panel**: the same form as everywhere else (see
  [generating.md](generating.md)).
- **Versions**, newest first. Each version is a line (its number, what
  changed, the model) and what it made. **use this recipe** loads its recipe
  into the form, now or when an image model is loaded; the version in use is
  marked **selected**. Hover the line for the prompts. **compare** shows the
  prompts of any two versions side by side as a word diff. Each result has
  three buttons: 🤖 **Ask** sends it to the assistant, 🖼 makes it the
  project's cover (click again to clear), 🗑 deletes the whole generation.
  Click a result to open its full detail view, with upscaling and redraw.
- **Assistant**: the project's conversation, in a drawer on the right that
  opens when the assistant is loaded. The page narrows to make room for it.

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

Any picture or video can be moved to a project — an import, something made in
the Sandbox, or a result of another project. The header of its detail view has a
**Project** select showing where it is: pick another project and press **Move**. To move
several, tick them in the gallery (**☑ Select**), pick the project in the
selection bar and press **Move N selected**. *No project* takes them out of
any. The button spins while the move runs, then a line says where they went
and how many upscales, redraws and edits went with them. The recipe of a moved
generation becomes one of the project's versions; an imported file has no
recipe and shows under *untagged*.

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
