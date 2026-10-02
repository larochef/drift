# LoRAs

A LoRA belongs to one architecture: a Flux LoRA means nothing to Wan. So LoRAs
are managed on the architecture cards of the **Models** page and picked in the
generation form of a session running that architecture.

## Installing

On an architecture card, open the folded **LoRAs (n)** section and click
**+ Add LoRA**. One browser opens, with the source as a row of buttons in its
header: **Civitai**, **HuggingFace**, **ModelScope**, **This machine**, each
with the site's mark. Switch source whenever you like; the search, the filters
and the results are that site's own ([browsers.md](browsers.md)). The same mark
shows beside every LoRA in the list, so you can see where each came from.

- **Civitai** opens in LoRA type, filtered to that architecture's base models.
  It is greyed out when the architecture declares no Civitai base model
  (SenseNova U1.5, which Civitai files under "Other"); hover it for the reason,
  and it is not offered at all for a chat model. Here the unit is a **version**,
  not a file.
- **HuggingFace** lists the LoRAs trained on the architecture's model, most
  downloaded first. The **trained on** choice switches between that model's
  variants (Qwen Image 2512 or the original, Wan 2.2 text- or image-to-video…)
  or to **Any repository**; typing narrows the list. A repository opens on its
  **Examples**, so you can see what a LoRA does before picking a weight file
  from **Files**; it installs as one LoRA, named after the file. Gated
  repositories need your HuggingFace token in Settings.
- **ModelScope** opens on the LoRAs of the architecture's base models, with
  their cover images.
- When drift knows no base model for the architecture on HuggingFace or
  ModelScope (Wan 2.2 on ModelScope, for instance), that browser starts on a
  search for the architecture's name instead of every LoRA the site hosts.
  Clear the search field to see them all.
- **This machine** picks a file on the drift host and copies it into drift's
  LoRA folder; the original is left where it is.

In every browser, a LoRA this architecture already has is marked **✓
installed**: on its result tile (for Civitai, when any version of the model is
installed) and on the version or file row itself, the chip naming the LoRA it
became. The mark appears as soon as the install starts, so the browser can
stay open without you losing track of what you already took. A LoRA installed
on another architecture is not marked — installing it here makes a separate
LoRA with its own copy of the files.

Under the installed LoRAs, **Official LoRAs** lists the ones drift knows about
for that architecture — the vendor's own, such as SenseNova's 8-step LoRA, or
a well-known distillation such as Boogu Image's Turbo as a rank-128 LoRA —
with what they are for and an **Install** button. Once installed it is a LoRA
like any other; delete it and it is offered again.

Installing works the same way whichever source you are in:

- Open a model or a repository and tick the weight files you want — **nothing
  is ticked to begin with**, and on Civitai the ticks may cross versions.
- The bar over the list says how many files are ticked and how much they
  weigh, and **Install into** says where they go: **one LoRA** of all of them
  (the default when you ticked several — this is how a wan 2.2 high/low-noise
  pair becomes one LoRA, each file tagged with its stage), **N separate
  LoRAs**, or **files of 'X'** for a LoRA this model or repository already
  made. One ticked file makes a LoRA of its own by default.
- Ticked one half of a pair yesterday and the other half today? Choose **files
  of 'X'** and the second joins the first.
- Want to compare two versions of the same Civitai LoRA? Install them one
  after the other: a LoRA taken from a single version of a model that
  publishes several is named after that version, so both sit in the picker
  side by side.
- Only the versions made for this architecture's base models are listed. Tick
  **Versions for other base models (N)** to see the others: they install just
  as well and usually do nothing, which is why they are out of the way.
- The browser stays open after an install so you can queue several from one
  search; progress shows on the card and in the Downloads panel.
- A file the running session has not seen is unusable until you restart the
  session: sd-server scans its LoRA folder at launch.

## Tuning on the card

Each installed LoRA shows:

- its **name** — click it to rename: what an install could go on is the file's
  own name, and `high_noise_model` says nothing a week later. Enter or
  clicking away saves, Escape puts it back;
- its **trigger words** from Civitai (none for the other sources);
- an **sfw / nsfw** tag — click to move it between the two folders;
- a **default strength** — remembered, seeds the picker every time; useful
  for LoRAs that need an unusual weight;
- per-file **stage** chips (general, low noise, high noise) — drift guesses
  from the file's path, its folders, and the name of the Civitai version or
  the repository it came from ("high noise", "low noise", and the short
  `_HIGH` / `_LOW` / `hn` / `ln` the repository sites use); click to cycle if
  the guess was wrong, hover one to see the file and where it came from;
- **⇄ Pair**, on a LoRA whose files are all high noise or all low noise: some
  wan 2.2 pairs are published as two repositories, so they install as two
  LoRAs. Choose the other half, give the two a name, and they become one LoRA
  holding both stages — the other entry goes;
- **⚙ Sampling**: the settings the LoRA was made for — steps, CFG, flow
  shift, and where they apply the sampler, the scheduler, the distilled
  guidance, the exact sigmas, or on wan 2.2 the high-noise steps and CFG.
  Type them from the LoRA's page (drift does not read them from Civitai or
  Hugging Face); leave empty what the LoRA does not care about. A turbo LoRA
  typically wants its step count, CFG 1 and a flow shift around 3. The LoRAs
  drift offers come with theirs filled in;
- **delete**, which removes the entity and its files.

## Using one in a generation

The generation form has a LoRA picker:

- Search by name, trigger word, tag or description. The **NSFW** checkbox
  (off by default) includes LoRAs filed under nsfw.
- Adding a LoRA shows a strength input seeded from its default and its
  trigger words as one-click inserts into the prompt. Nothing is inserted for
  you; the prompt is yours.
- A LoRA that carries sampling settings (**⚙ sampling**) puts them in the
  form when you add it: they are one more layer of defaults, over the run
  configuration's. The fields you have changed yourself keep your value, the
  others follow the LoRA; removing it gives them back to the configuration.
  Click **⚙ sampling** on a selected LoRA to type or change its settings
  right there — the same fields as on its card, saved on the LoRA itself, so
  they hold wherever it is used; a LoRA with none yet shows the mark in grey.
  With two such LoRAs the one added last wins where both set a field. Nothing
  is locked: change any value and generate. The line under the picker says
  what each LoRA sets, **Apply the LoRA settings** puts them back over what
  the fields hold (after reusing a gallery entry, for instance, whose own
  values are kept), and a warning shows when the steps are fewer than the
  LoRA was made for.
- "⚠ needs restart" on a LoRA means the running session cannot see its file
  yet (installed or moved after launch). **↻ Restart** at the top of the
  form starts the model again with your form as it is.
- **+ Add LoRA** in the generation form's picker installs one for the running
  model's architecture without leaving the page, from Civitai, HuggingFace,
  ModelScope or this machine.
- A wan 2.2 pair is sent as both files automatically, the high-noise half
  applied to the high-noise stage.

## LoRAs for a chat model

The assistant's chat models take LoRAs too, installed on their architecture
card from HuggingFace or the disk (Civitai hosts no chat model LoRAs, so it is
not offered), with two differences:

- llama.cpp loads LoRA adapters as **GGUF** only. Most chat LoRAs on
  HuggingFace are safetensors (PEFT) and must first be converted with
  llama.cpp's `convert_lora_to_gguf.py`; drift refuses a non-GGUF file for a
  chat architecture and says so.
- A chat session loads every installed LoRA of its architecture at launch
  without applying any. Add the ones you want as **default LoRAs** on the chat
  run configuration: every assistant reply applies them at their strengths. A
  LoRA installed after the session started makes the assistant refuse to
  answer until you restart the session, rather than silently leaving it out.

## Default LoRAs on a run configuration

A run configuration can carry default LoRAs with their strengths — the place
to encode "this checkpoint always runs with its turbo LoRA at 0.7". Edit them
on the configuration's card with the same picker.

- Launching that configuration pre-fills the picker with the defaults. Reusing
  a gallery entry on the same configuration keeps the entry's own LoRAs
  instead.
- The defaults' sampling settings apply too, as for a LoRA added by hand.
- Tiled post-processing jobs (PiD upscale and redraw, see
  [gallery.md](gallery.md)) run without a form, so they apply the
  configuration's defaults. Redraw and edit take the LoRAs' sampling settings
  with them (the steps you ask for still win); PiD keeps its own 4 steps. A job whose default LoRA is missing on disk refuses
  to start and says which.

## On disk

`~/.cache/drift/loras/<architecture>/<sfw|nsfw>/<lora>/` holds the weight
files and, for a Civitai LoRA, its metadata and two previews. The **LoRAs** sub-tab of the
Model Cache's on-disk view lists this tree; deleting a row removes the whole
folder, and an orphan folder can be re-attached to an architecture with
**Adopt** ([model-cache.md](model-cache.md)); a folder you filled by hand is
adopted with its files as they are.
