# HuggingFace, ModelScope and Civitai browsers

Models, LoRAs and upscalers are found without leaving drift. The browsers
open from wherever something is registered: a checkpoint slot on an
architecture card, the LoRAs section of a card, the Upscalers tab of the
Model Cache. Picking a file registers it; nothing downloads until you ask
([model-cache.md](model-cache.md)).

When you opened the browser to add a model or a LoRA, its header says where it
is looking: switch between Civitai, HuggingFace, ModelScope and your own disk
there at any time, without closing it.

Three things hold in all three. Results are tiles, five to a row, showing the
preview, the name, who published it, its numbers, and how many examples it has
in the corner; a video tile plays muted, with a pause button of its own, and
**Play videos** at the top of the browser turns them all off. **← Back** from
an opened model or repository returns to the results scrolled where you left
them, so a long browse carries on where it stopped. And what drift already has
is marked, on the result tile and on the version or file row it came from: a
LoRA this architecture holds as **✓ installed** ([loras.md](loras.md)), a file
already registered as a model as **✓ registered** ([models.md](models.md)), an
upscaler you installed as **✓ installed**. Hover the chip for the name it
carries in drift.

## HuggingFace

- Type a query; sort by most downloaded, most likes, recently updated or
  recently created; **Load More** pages further.
- Opening a repository shows **Files** and **Model card** tabs. The card is
  the repository README, rendered. A gated repository shows its card and its
  files only with a token whose account accepted the terms; the tab says so
  instead of erroring, and the Files tab carries a **🔒 Gated repository**
  line — accept the terms on huggingface.co (or wait for the author, where
  access is granted by hand) and set a token in Settings, otherwise the
  download fails with 403.
- Results are image tiles, like Civitai's: the repository's first example
  fills the tile, with its name, its owner and its downloads and likes over
  it, and how many examples it has in the corner. Hover a tile for its full
  `owner/name`. **With examples only** hides the repositories that have none,
  which is where most test uploads end up.
- Opened to install a LoRA, it starts on the LoRAs trained on the
  architecture's model, with a **trained on** choice for its other variants
  and **Any repository** ([loras.md](loras.md)), and with **With examples
  only** ticked.
- An opened repository has an **Examples** tab: the images and videos of its
  card, with the prompts that made them, then any image files it holds. Click
  one to see it large with its prompt. The **Model card** tab shows the card's
  gallery too.
- Choose a file to register it as a model of the family you came from. For a
  model split into shards, choose its `model.safetensors.index.json`: it
  brings every shard with it ([models.md](models.md)).

## ModelScope

ModelScope (modelscope.cn) is Alibaba's model hub: the Chinese labs publish
there, and its community shares LoRAs with cover images, much like Civitai.

- Results show at once, without typing: the same image tiles as the other two
  browsers — the repository's first example, its name, its owner, its
  downloads and stars, and how many examples it has in the corner. Hover one
  for its full `owner/name`, whether ModelScope files it as a LoRA, and what
  it was trained on. Sort by downloads, stars, update or creation date;
  **Load More** pages further.
- **LoRAs only** and **With examples only** narrow the results; both start
  ticked when you are installing a LoRA.
- Installing a LoRA, the **trained on** choice starts on all the base models
  the architecture's LoRAs name on ModelScope at once, and can pick one or any.
- An opened repository has **Examples** (its cover images, with prompts when
  their authors gave them), **Files** and **Model card**. Choose a file to
  register it as a model, or to install it as a LoRA; a split model is chosen
  by its `model.safetensors.index.json`.
- Private and gated repositories need a ModelScope token in
  [settings.md](settings.md).

## Civitai

- Type a query; sort by most downloaded, most liked, newest or oldest. The
  browser is already filtered to the type you came for (checkpoint, LoRA or
  upscaler) and to the base models the architecture declares.
- **Include NSFW** (on by default) is in the toolbar.
- An opened model shows only the versions made for the architecture you are
  adding to — installing a Wan 2.1 LoRA into Wan 2.2 is the easiest mistake to
  make. **Versions for other base models (N)** at the top of **Files** shows
  the rest, each keeping its ⚠ flag; a version that names no base model is
  always shown, and nothing is hidden when the architecture declares no Civitai
  base model.
- A version Civitai charges for is marked beside its name in **Files**:
  **💲 paid**, or **💲 early access until <date>** where the charge lapses on
  that date. It is per version — a model often has one paid version and several
  free ones — and drift cannot buy it: the download works only for an account
  that owns it (the one whose Civitai token is in Settings), while an early
  access version simply becomes free on its date.
- Each file row carries its precision tag. **⚠ runs on CPU** marks int8
  files, which sd-cpp cannot run on AMD GPUs — convert them after downloading
  ([model-cache.md](model-cache.md)). fp8 files run on the GPU. A version published for a
  base model the architecture does not declare is flagged ⚠ but still
  installs.
- Files already in your cache are marked as downloaded.

### Model details

Opening a Civitai model shows three tabs. The tab you chose last stays chosen
for the next model you open.

- **Files** — the versions and their files.
- **About** — the description, tags, and each version's notes and trigger
  words.
- **Images** — what people generated with it, one version at a time (newest
  by default): the author's showcase first, then posts from Civitai's search
  ten at a time, re-sortable, **Load More** for more. A tile opens a viewer;
  videos play with controls; a link leads to the post on civitai.com, where
  the prompt is shown when its author shared it. Civitai's image search is
  slow and sometimes overloaded: a failure says why beside a **Retry**.
  A repository that does not open — ModelScope answers "获取模型目录树失败" now and
  then, and its files and pictures come in that one request — says so above
  the tabs, with a **Retry** that asks for that repository again. Trying again
  usually works.

### Installing

- A checkpoint: pick one file; it becomes a model of the family.
- A LoRA: the version's file rows are toggleable; the install button shows the
  count and size, and the browser stays open for the next one
  ([loras.md](loras.md)).
- An upscaler: same version flow, from the Model Cache's Upscalers tab.

## Tokens

Add a HuggingFace token in [settings.md](settings.md) for gated repositories
(their files and their cards) and a Civitai token for creator-restricted or
early-access downloads. The Civitai token is sent to civitai.com only. A
ModelScope token opens private and gated ModelScope repositories; it is sent to
modelscope.cn only, never with the CDN address a download is redirected to.
