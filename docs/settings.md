# Settings

The Settings page holds the two things drift needs before it can run anything: a
runtime to run models with, and the tokens some downloads need.

## Runtimes

A runtime is a build of **sd-cpp** (images and videos) or **llama.cpp** (the
assistant). drift downloads and manages these itself, so you never need to install
either tool by hand. Each build comes in three flavours:

- **ROCm** — the AMD GPU build. It needs a matching ROCm library set, which drift
  fetches from AMD as a "TheRock" build and pairs automatically.
- **Vulkan** — works on most GPUs, no extra libraries. The fallback when ROCm
  misbehaves. Flash attention is unavailable here; drift drops the flag and tells you.
- **CPU** — slow, but runs anywhere.

### Adding a runtime

Click **Add a runtime**. The modal has two tabs:

- **Install a runtime** — pick the tool, a release (newest first) and the build.
  For ROCm, also the GPU target (`gfx1151` by default) and the TheRock build: drift
  preselects the version the release was built against and lists every stable
  version if you want to try another. **Install latest** instead installs a
  runtime that follows the newest release; its **Upgrade** button keeps it current.
- **Adopt an existing install** — point drift at a directory that already holds
  `sd-server` or `llama-server`. drift uses those files but never deletes them.

Installs are downloads like any other: progress stays on the page and in the
downloads panel, and two can run at once.

### The runtime list

Each row shows the tool, the build, the release, the paired ROCm version and
whether the build **validated** (drift ran the executable once with its libraries;
a runtime that fails stays listed with the error and cannot be the default). Per row:

- **Set default** — one default per tool. Launching a configuration uses the
  default unless you pick another runtime in the launch control.
- **Upgrade** / **Check for update** — on a latest-tracking runtime. When the new
  release wants a different ROCm version, the row asks which TheRock build to pair.
- **Change ROCm** — re-pair a ROCm runtime with another TheRock build without
  downloading the tool again.
- **Revalidate**, **Delete**. Files shared with another runtime are kept.

Everything unpacks under `~/.cache/drift/runtimes/`, which is safe to delete: it
only costs a re-download.

### drift's own runner

drift is growing its own inference engine for Strix Halo, the **drift runner**.
It speaks llama.cpp's and sd-cpp's languages, so it shows up as two runtimes,
**drift runner (gfx1151)** for chat and **drift runner, images (gfx1151)** for
images. A configuration runs on it when its **Runner** says so, as long as its
architecture lists the drift runner (see [models.md](models.md)).

It ships inside drift rather than being downloaded. **Add a runtime → Install
the drift runner** installs both runtimes, on the ROCm (TheRock) build you pick
as for any ROCm runtime. It is paired against the ROCm its kernels were built
with, and the build is downloaded if needed. **Change ROCm** moves both, and
deleting one deletes both. Each start of drift brings an installed runner up to
the one drift carries, so updating drift updates the runner.

It runs only on a Strix Halo GPU (gfx1151). Elsewhere its validation fails and
says why, and it never launches.

So far it runs dense Qwen 3 models, Qwen 3.5/3.6/3.8 dense or with experts
(such as Qwen 3.8 27B and Qwen 3.6 35B-A3B) and Qwen 3.8 Flash Next. Qwen 3.6 and 3.8 see images when
the configuration has the model's vision checkpoint (its mmproj GGUF): attach
pictures in the assistant or a text project as with llama.cpp. Videos are
refused for now. For Qwen 3.8 Flash Next,
MTP drafting takes the MTP head's own GGUF (the configuration's MTP
checkpoint); past about 2000 tokens of context the runner attends to every
token where the model would pick a selection, as llama.cpp does with the same
files. A model file without an MTP layer (the ROCmFP4 builds of Qwen 3.8 27B
and Qwen 3.6) drafts with an MTP-only GGUF in the MTP checkpoint too. Its
answers follow the exact maths more closely than llama.cpp's: the model's
inputs are never rounded to 8 bits, except for ROCmFP4 weights, which meet
8-bit inputs as in the fork that made the format.

- **Images: Krea 2, FLUX.2 [klein] and [dev], Qwen Image 2.1, HiDream O1 and
  PiD.** A second runtime, **drift runner, images (gfx1151)**, runs Krea 2,
  FLUX.2 [klein] 9B, FLUX.2 [dev], Qwen Image 2.1, HiDream O1 and PiD 1.5
  configurations in
  place of sd-cpp: pick
  it in the launch control. It tells them apart by the checkpoint, and reads
  the checkpoints the way sd-cpp does: bare or with ComfyUI's
  `model.diffusion_model.` prefix, and FLUX VAEs under their original or
  diffusers' names (ERNIE-Image's `flux2-vae`, for instance). Flags it has no
  use for, such as `--attn-scale`, are dropped at launch with a note saying
  why, so a configuration made for sd-cpp runs unchanged. `--guidance` is
  used by FLUX.2 [dev] and ignored, with a note, by the other models.
  - **Krea 2** makes a 1024² image in about half sd-cpp's time, LoRAs
    included, and does img2img (so redraws run on it). It doesn't do
    reference images.
  - **FLUX.2 [klein]** does text to image (about 40% faster than sd-cpp),
    img2img, reference images and LoRAs (ComfyUI's, BFL's or diffusers'
    names), all faster than sd-cpp: about half its time for text to image,
    a third less with a reference. It ignores
    `--flow-shift`, which Klein doesn't use.
  - **FLUX.2 [dev]** does the same as Klein, with its Mistral Small text
    encoder (GGUF) and the distilled guidance (`--guidance`, or the form's
    distilled guidance). LoRAs in kohya's names work too (the SexGod ones,
    for instance). With the Turbo LoRA at 8 steps it takes about as long as
    sd-cpp with its step cache for text to image and a reference (about 3
    and 6½ minutes), and 40% less for img2img.
  - **Qwen Image 2.1** does text to image with real guidance (`--cfg-scale`
    above 1 runs the negative prompt too), img2img and LoRAs (diffusers' or
    ComfyUI's names), in about 60% of sd-cpp's time with guidance and 80%
    with a 4-step turbo LoRA. Its images are RGBA, as the model makes them. It
    edits with reference images when the configuration has Qwen3-VL's vision
    checkpoint (its mmproj): each reference is scaled to about a megapixel,
    read by the text encoder and placed before the image being made. It
    ignores `--flow-shift`: the shift follows the image's size, as in the
    official pipeline.
  - **HiDream O1** does text to image and img2img, from its one file and its
    `tokenizer.json`. The Dev checkpoint samples as HiDream's own code does
    (its distilled timesteps, fresh noise each step), which is why its
    images come out finished where sd-cpp's look undercooked; a 2048² image
    in 28 steps takes about 2 minutes, 40% less than sd-cpp. The full
    checkpoint runs with guidance (CFG 5) on Euler steps, where HiDream's
    code uses UniPC. LoRAs work (kohya's `lora_A`/`lora_B` over the
    checkpoint's names, as Civitai's O1 LoRAs come). A LoRA trained on the full
    checkpoint (ai-toolkit's, most of Civitai's) belongs on the full one: on Dev
    its images come out washed out, soft and faintly gridded, worse the
    stronger it is. It doesn't do reference images yet.
  - **PiD** upscales 1024 → 4096 in one pass, about 5 minutes, where sd-cpp
    needs nine tiles (its single pass comes out black). All three variants
    run: FLUX.2, FLUX.1 and Qwen Image, each with its own VAE (FLUX.1's takes
    Z-Image's or FLUX.1's `ae`, Qwen Image's the Qwen Image VAE). It is
    distilled for 4 steps and takes no other count.
  - **None** does masks, hires fix or VAE tiling yet; the runner says so
    when a request asks for them.
  - **Images differ from sd-cpp's** for the same seed: the runner follows
    the reference implementation's noise schedule and uses its own random
    numbers.
- **MTP drafting.** With `--spec-type draft-mtp`, a model whose file carries an
  MTP layer drafts `--spec-draft-n-max` tokens per step (2 by default). The
  answer is the same as without drafting, only faster: on Qwen 3.6 35B-A3B
  ROCmFP4, about 110 tokens per second on code and 90 on prose instead of 80
  (2 drafts); on Qwen 3.8
  27B ROCmFP4 with its MTP-only file, 28–34 instead of 13.5 on code and 21 on
  prose (2 to 4 drafts; fewer drafts suit prose).
- **Prefix cache.** A conversation's next message only runs its new tokens,
  since the earlier ones are kept from the previous reply. When the chat
  template rewrites the last reply (Qwen 3.6 with thinking off), it resumes
  from that reply's start, and images already read are not read again.
- **Refused flags.** It refuses a configuration that asks for what it cannot do
  yet, such as LoRAs, and names what it refused.
- **Accepted flags.** Flags that only change speed, like `-fa`, are accepted.

## Authentication

Some sources need an API token:

- **Civitai** — searching, viewing and downloading account-gated or NSFW models,
  and LoRA and upscaler installs from Civitai.
- **HuggingFace** — downloading gated repositories (for example LTX-2.5 once you
  have accepted its licence on the site).
- **ModelScope** — browsing and downloading your private repositories and
  gated ones. Create an access token (SDK token) in your ModelScope account
  settings. Public repositories work without one, and a wrong token does not
  get in their way.

**Add a token** stores one under a label; several can coexist per provider and
**Make active** picks the one drift sends. Tokens are stored as plain text in
`~/.config/drift/auth-tokens/`. Without a saved token drift falls back to the
`CIVITAI_API_TOKEN`, `HF_TOKEN` and `MODELSCOPE_API_TOKEN` environment variables. Listing releases on
GitHub is anonymous unless `GITHUB_TOKEN` is set, which helps against rate limits.

## Prompts

The prompt library: every fixed text drift puts in front of a model, in
variants you can pick where it is used.

- **Assistant system prompts** — chosen in the assistant panel and saved per
  project. drift's helper and the Ideogram 4 caption writer are built in.
- **Redraw restoration prompts** — chosen under Advanced on a redraw. drift
  restoration (the default), skin de-artifacting and women's portrait are
  built in; [gallery.md](gallery.md) says when each fits.
- **Edit prompts** — chosen under Advanced on an edit; your instruction
  follows it. *drift seamless edit* is built in.
- **Compaction prompts** — used when a project's conversation is compacted.

Built-ins are read-only and come back on every start. **New prompt** starts
a blank one of that kind; **Duplicate** copies an existing one. Either can be
**Edit**ed or **Delete**d: the label, the text, and for assistant templates
which of drift's additions follow it and whether a proposal is read from
fenced `prompt` blocks or from the last JSON object in the reply. **Show**
unfolds any template's text. See [assistant.md](assistant.md).

See also: [models.md](models.md) for launching on a runtime,
[model-cache.md](model-cache.md) for downloads, [files-and-folders.md](files-and-folders.md).
