# Models

The Models page describes what drift can run. It has two tabs: **Run
configurations** (what you launch) and **Architectures** (the shapes they are
built from). Both share the same search box and tag filters.

## Architectures

An architecture describes one model family: which weight files it needs (its
**checkpoint slots**, each mapped to a command-line flag and a **family** of
files that can fill it), the default parameters, the runner (sd-cpp or
llama.cpp), and **tags** saying what it is for — `image`, `video`, `edit`,
`audio`, `llm`, `upscale`, or any word you add. Tags drive the filters here and
which configurations a project offers. A slot can be optional: Qwen Image 2.1
generates from three files and wants its fourth, the vision projector, only
when you hand it a reference image to edit.

drift ships built-in architectures for Flux.2 (dev, Klein 9B and 4B), Z-Image
Turbo, Krea 2, Qwen Image, Qwen Image 2.1, Ideogram 4, Mage-Flow Turbo and
Edit Turbo, SenseNova U1.5, Boogu Image and Boogu Image Edit, ERNIE Image,
HiDream O1 Image, Wan 2.2, LTX 2.3 and 2.5, MiniMax H3, HunyuanVideo 1.5, the
PiD upscalers and two chat models. They are refreshed on every start and cannot be
edited or deleted; **New Architecture** creates your own — where, besides the
slots and defaults, **Size multiple** says what the model rounds every side up
to (16 for nearly everything, 32 for Qwen Image 2.1 and a few others), which is
what a redraw lays its tiles out on. An architecture can
carry **prompting notes**, a sentence or two the assistant reads about how the
model likes its prompts.

An architecture also says which **runners** can run it. Its tool's own engine
(sd-cpp or llama.cpp) always can. **Runs on the drift runner too** adds drift's
own engine. The form offers that only for a **model kind** the installed drift
runner runs. The kind is what the model is: the GGUF's `general.architecture`
for a chat model (`qwen35`, `qwen35moe`, `qwen4exp`), the family for an image
model (`krea2`, `flux2-klein`, `flux2-dev`). The built-ins say theirs; on your own
architectures, fill it in.

## Models

A model is one weight file registered in a family, so any architecture with a
slot of that family can use it. Expand a slot on an architecture card and click
**Add Model**: the browser opens straight away, with the source as a row of
buttons in its header — HuggingFace, ModelScope, Civitai or this machine (see
[model-cache.md](model-cache.md) for how files are found and downloaded). A
file you have already registered is marked **✓ registered** there, with the
name it carries in drift, so you do not add it twice. The site's mark then
shows beside the model wherever it is listed.

The search depends on the slot. The architecture's own model (the diffusion
model, Wan 2.2's high- and low-noise models, a chat model) is looked for as
the architecture, Civitai first. Every other slot (the LLM, a text encoder,
a VAE, a vision projector) holds a model many architectures share, so it is
looked for by its family on HuggingFace and ModelScope: **Search for** chips
under the search field offer the family (`qwen3-vl-8b-instruct` finds Qwen's
releases and their GGUF quantisations) and the architecture (`Qwen Image 2.1`
finds the repositories that repackage its text encoder and VAE). A VAE starts
on the architecture, the rest on the family. Civitai is not offered there: it
has no type for these files.

Picking a file closes the browser and opens a window for the rest: the model's
ID and label (suggested from the file), its source with **Change model** to
pick again, and the parameters it asks for. **Add** registers it. Editing a
model opens the same window with **Save**.

Pick a file and the panel appears with what you picked already filled in:

- the **source** is shown as it stands — the site, the repository or model, the
  file, and the weight format read off its name — and cannot be edited, since
  it is whatever you picked. **Change model** reopens the browser on another
  file;
- **Model ID** and **Label** are suggested from the file and are yours to
  change before saving; an id you have typed is kept when you change the file;
- **Parameters** are yours as before — a Turbo checkpoint's step count, for
  instance — and override the architecture's defaults wherever the model is
  assigned. The flags the architecture sets are offered as chips under the
  list, each saying which layer set it: ✎ on a chip overrides it here, 🚫
  removes it (the row then reads **not passed**). 🗑️ drops a row instead,
  which leaves the layer below deciding again.

### Models split into several files

Official releases often split a model into shards —
`model-00001-of-00008.safetensors`, `model-00002-of-00008.safetensors`, … —
with a small `model.safetensors.index.json` that says which shard holds which
tensor. Register the **index file**, not a shard:

- From HuggingFace, pick `model.safetensors.index.json` in the file list
  (sometimes in a subfolder, such as `transformer/`). Downloading the model
  fetches the index and every shard it lists, into the same folder.
- From a local folder, pick the index; the shards must sit beside it, under
  the names the index gives them.

The model counts as downloaded only once every shard is on disk, its size is
the shards' total, and sd-cpp is given the index, from which it loads the
shards. Registering a single shard gives sd-cpp that shard alone: the model
fails to load or loads incomplete. SenseNova U1.5's official release comes
seeded this way (`SenseNova-U1.5-8B-MoT bf16 (official, 8 shards)`).

Every built-in architecture comes with a vendor model per slot, chosen to run on
the GPU (GGUF, bf16 or fp8, never int8). SenseNova U1.5's community turbo
merge, an fp8 file, brings its own 8 steps and CFG 1. Shared files such as VAEs and text
encoders are registered once and used by every architecture that needs them.
Built-in models are marked **built-in** and cannot be edited or deleted: drift
restores them from its reference on every start, so an edit would not survive
one — change what they pass on the run configuration instead. To free the disk
space one takes, delete its file in Model Cache → On disk; the model stays
listed as not downloaded. Models you add have ✏️ and 🗑️ beside them:
the edit panel is the one that added them, with the label, the file
(**Change model**) and the parameters yours to change. The **Model ID** is
not — run configurations assign by it — so renaming one is still a delete and
a re-add.

An architecture can also name the **default assistant prompt** a project
starts with when it targets it, chosen from the prompt library in Settings
(Ideogram 4 comes with its caption writer). LoRAs are managed per
architecture in a folded section of its card; see [loras.md](loras.md).

## Run configurations

A run configuration is what you launch: an architecture, a model for each slot,
any parameter overrides, default LoRAs, and optionally the assistant prompt
it prefers over its architecture's default. Its **Runner** is the engine it
runs on, among its architecture's runners. Every launch uses it: the launch
control, projects, text projects, the assistant, PiD, redraw and edit. The card
says which one, and a runner that isn't installed is marked so. Chat models get
configurations the same way. **New run configuration** creates one; the card's **Edit** changes it.

A slot whose model isn't registered yet doesn't need a trip to the
architectures page: **+ Add model** beside the slot opens the browser on that
architecture, and the model you pick is registered and assigned to the slot.
**+ Add LoRA** above the default LoRAs does the same for LoRAs.

drift adds six starter configurations on first start: Flux.2 Klein 9B,
Krea 2 Turbo and Qwen Image 2.1 for images, MiniMax H3 for video, PiD 1.5
(FLUX.2) for diffusion upscaling and Qwen 3.6 35B-A3B for chat. After that
they are yours like any other: edit them, delete them — a deleted one does not
come back.

Each card shows whether the configuration is **ready** or **incomplete**, and
names what blocks it — an empty slot, a deleted model, weights not downloaded yet.
While weights are missing, the **Launch** button is replaced by **Download the
weights**; it says so while they download, and turns back into **Launch** once
they are on disk. With no runtime to run it, the same place says so and lets you
pick one to install — drift's runner too, where the architecture runs on it,
which switches the configuration's runner (see
[getting-started.md](getting-started.md)). The same swap
happens wherever a model is started: a
project's model pickers (the entry reads *download the weights*, and picking it
downloads rather than launches), the gallery's reuse and *try this task*
buttons, and the redraw, edit and PiD panels. A collapsible preview shows the exact
command line it would run, with **Copy**.

### How parameters resolve

Flags merge in four layers, each overriding the previous: the architecture's
defaults, then the assigned models' parameters, then the runtime's defaults, then
the configuration's overrides. The runtime can also **veto** a flag its build
cannot take (flash attention on Vulkan). A veto is never silent: the preview and
the launched session list it as a note.

A layer can also **remove** a flag rather than give it another value — a
model, for one of its architecture's defaults, and a configuration, for
anything the three layers below it set. In either editor, 🚫 on a row turns
it into a removal (the row reads **not passed**) and ↩ gives it a value again;
the chips under the list are the flags already set below, each naming its layer
— `--attn-scale 0.0078125 · model` — with ✎ to override one here and 🚫 to
remove it in a click. Overriding a flag with the value it already had changes
nothing, which is why the chips say where a value comes from — and drift does
not keep it: saving stores only what actually overrides something, so a row
repeating the value below, or a removal of a flag nothing sets, is gone when
the card reloads. Nothing is lost, because the effective command line is
computed from the layers every time; what is gone is a line in a file that
looked like a decision and was not. An empty value is not a removal — it is how a flag with no value,
such as `--diffusion-fa`, is written — and 🗑️ only drops your row, letting
the layer below decide again. A removal that actually takes a flag off the
command line is listed as a note beside the vetoes, naming the layer that
wanted it and the one that removed it. The architecture form has no 🚫: it is
the bottom layer, where removing a default means deleting it.

### Launching

The card's launch control preselects the configuration's runner. It offers
the other installed runtimes its architecture can run on, for one launch.
Click **Launch** and the Sandbox opens on the model; progress, notes, the log
and **Stop** are described in [generating.md](generating.md). A running
configuration's card says where it runs — *Running in the Sandbox* or in a
project — with **Open** to go there and **Stop**; the page itself only lists
configurations. One image session and one assistant session
run at a time: a second model would not fit in VRAM. There is no UI for the
limits; a file
`~/.config/drift/settings/sessions.json` with `maximumConcurrentSessions` and
`maximumConcurrentAssistantSessions` overrides them.

Trying a model happens in the **Sandbox** (see [generating.md](generating.md));
real work lives in projects: see [projects.md](projects.md).
