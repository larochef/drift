# The prompt assistant

The assistant is a chat model that runs on your machine. drift runs it
through llama.cpp the way it runs image models through sd-cpp: you describe
the model once, and drift downloads it, launches it and talks to it. It helps
you write and refine prompts, and can look at what you generated.

## Setting one up

1. On **Models → Architectures**, drift ships three chat families: Qwen 3.6
   35B-A3B, Qwen 3.8 27B and Gemma 4 26B-A4B. Each has a required model slot
   and an optional vision projector (mmproj) slot; Qwen 3.8 27B and Gemma 4
   also take an MTP head in a file of its own (the `mtp` slot), for a model
   file without one. Register the GGUF files from the
   HuggingFace browser (see [browsers.md](browsers.md)) and download them from
   Model Cache (see [model-cache.md](model-cache.md)).
2. On **Models → Run configurations**, create a configuration for the chat
   architecture and assign the files. Assign the projector if you want the
   assistant to see images; vision and MTP drafting work together on
   llama.cpp b9240 and later.
3. You need a llama.cpp runtime, installed from **Settings** like an sd-cpp
   one (see [settings.md](settings.md)).

One assistant can run beside one image model. A chat configuration can carry
default LoRAs, applied to every reply (GGUF adapters only; see
[loras.md](loras.md)).

## Chatting

Launch a chat configuration from the Models page, or pick it in a project's
**Assistant** picker (see [projects.md](projects.md)). The panel shows the
context size the server applied and whether it has **vision** or is **text
only** — in a project, beside the picker, with **Stop session**.

- The composer is at the top; replies appear right under it, newest first.
  A reply comes in as plain text and is formatted (headings, lists, code,
  tables) once it is finished.
  Ctrl+Enter sends. **Stop** cuts a reply short.
- The model's thinking is folded away under "thinking…"; open it if you want.
- Under each reply: how many tokens it read and wrote, and how fast.
- The **context meter** shows how full the model's memory is. Past 75 % it
  warns you; at 100 % sending is refused until you compact or restart.

## Showing it an image

Images are never sent on their own. You attach one when you want to:

- Click 🤖 **Ask** on any result, in the workspace, the generation panel or a
  gallery detail. The image is staged with the prompt and parameters that
  made it, so the model compares what you asked for with what came out.
- Use the file input in the composer for any other image or video.

Staged files go with your next message only. A text-only model gets the
parameters and a note that it did not see the picture.

## Proposals

When the assistant has a concrete suggestion, it ends with a prompt and a
negative prompt in fenced blocks. drift shows them as a card:

- The prompt is shown as a **word diff** against the prompt you asked about,
  so you can see exactly what was added or removed. If most words changed,
  the card opens on the plain text and lists the words that did not make it.
- **Apply to form** fills the generation form's two prompt fields, and
  nothing else: adjust, then press Generate. In a project, **Apply and run**
  fills them and generates at once; it needs an image model loaded.

By default the assistant edits your prompt and keeps everything else as
written. Ask for a rewrite when you want the whole prompt said another way;
it should still keep every detail you asked for. The diff is there to check
that it did.

## The system prompt

Open **System prompt** in the panel header to pick the template the model is
told, and read it. Templates live in Settings → Prompts: drift's own helper,
an **Ideogram 4 caption writer** that turns a plain idea into the JSON
caption Ideogram needs, and any copies you edited. drift adds after the
template, when it allows: the project brief, how the target image model
likes its prompts, your form's current prompt, a note when CFG is 1 (the
negative prompt does nothing then), its rules for editing rather than
describing, and the form's aspect ratio. A project starts with what its
image model suggests: the run configuration's own choice, else the default
its architecture names (the caption writer for Ideogram 4, drift's helper
elsewhere), and keeps your own pick once you make one. When you start a
model whose suggestion differs from the project's pick, the panel offers to
**Switch** or **Keep**.

With the caption writer, a reply's JSON object is the proposal: **Apply to
form** puts it in the prompt and, when the caption names an aspect ratio,
sizes the form to it. Expect most of the help, not all: a local model may
drop parts of the contract or misplace boxes, and the negative prompt stays
empty since Ideogram ignores it.

## Per-project conversations

Inside a project the conversation is kept and comes back when you reopen it.

- **Compact** asks the model to summarise the conversation so far and
  replaces it with that summary. Use it when the context meter fills. The old
  messages stay readable under "show the archived transcript".
- **Restart** archives everything and starts fresh. Versions and generations
  are untouched.
- **Retry** appears on a reply that failed or was stopped.

## Sandbox

The Sandbox's **Image** and **Video** tabs have the assistant a project has:
pick a model in the **Assistant** row and its chat opens as a drawer beside
the form. It reads the prompt in the form, uses the system prompt the image or
video model is set up for, and its proposals offer **Apply to form** and
**Apply and run**. The chat is not kept: it goes when you leave the page or
switch tabs.

The **Text** tab gives a scratch chat with no template by default: the raw
model. Pick one from the System prompt select to change that. It goes when you
leave the page or switch tabs; **Clear chat** empties it. Apply to
generation form hands a proposal to the Sandbox's image form.
