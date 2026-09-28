# 23 — Seeded vendor models

**Status:** done — 80 models fill every checkpoint slot of all 27 built-in
architectures; seven seeded models carry per-model parameters (SenseNova's turbo
merge, HunyuanVideo's 480p cfg-distilled checkpoint, and the full — non-turbo,
non-dev — checkpoints of Boogu Image, ERNIE Image and HiDream O1, whose step
count and CFG differ from their architecture's distilled defaults), the others
take the architecture defaults. Mage-Flow Turbo, Mage-Flow Edit Turbo,
HunyuanVideo 1.5, Boogu Image, ERNIE Image and HiDream O1 are seeded but not
yet run live; Qwen Image 2.1 is seeded ahead of the sd-cpp release that runs it
(2026-09-20), so it cannot be run at all until that binary lands
**Depends on:** 02, 05, 16

A fresh install can run something without hunting for repository paths: a
reference file of models, seeded exactly like the architectures, one per checkpoint
slot of every built-in architecture of both tools.

## What it does

- `backend/resources/reference/models.json` is seeded into
  `~/.config/drift/models/` with `builtIn: true`: protected from deletion,
  rewritten from the reference, removed when the reference drops them. Every
  start compares, never blindly rewrites: a model whose stored bytes already
  match the reference is left alone ([`01`](01-architecture-registry.md)), so
  editing this file is what makes a start write anything.
- A seeded model whose id a user-created model already holds is skipped, so
  someone who registered the same file gets no duplicate row.
- Families shared between architectures are seeded once (`flux2-klein-9b-vae`
  serves Flux.2 dev, Klein 9B and Klein 4B; `qwen3-4B` serves Z-Image and Klein
  4B; `qwen3-vl-4b-instruct` serves Krea 2 and both Mage-Flows;
  `qwen2.5-vl-7b-instruct` serves Qwen Image and HunyuanVideo 1.5;
  `flux2-klein-9b-vae` serves ERNIE Image too — its VAE is the FLUX.2 one, as
  the sd-cpp doc's `flux2_ae.safetensors` and Comfy-Org's `vae/flux2-vae.safetensors`
  are the same file; `qwen3-vl-8b-instruct` serves Ideogram 4, Boogu Image and
  Qwen Image 2.1, and `qwen3-vl-8b-instruct-mmproj` Boogu Image Edit and Qwen
  Image 2.1's editing slot;
  `mage-flow-vae`, `wan-2.1-vae`, `umt5-xxl`, `qwen-image-vae`, `z-image-vae`,
  `flux1-vae`, `ministral-3-3b`, `gemma-2-2b`, `gemma-2-2b-tokenizer`,
  `pid-flux2-diffusion` likewise), so one download serves every taker.
- Nothing is downloaded at seed time; a slot's weights are fetched when a run
  configuration assigns it ([`05`](05-model-cache-and-downloads.md)).

## Shape

- `models.json`: a `List[Model]` (see [`02`](02-model-registry.md)); all sources
  are HuggingFace. Seeded by `StorageService.init` through the same
  `seedFromResource` as `architectures.json`.
- Mix: 33 GGUF, 46 safetensors (VAEs, mmproj files, fp16/bf16 models nobody
  has quantized, one sharded official release, `34`, the fp8 checkpoints), 1
  JSON tokenizer. Sources: Comfy-Org (38), unsloth (30), Lightricks (3),
  QuantStack (2), realrebelai (2), city96, vantagewithai, sensenova,
  ChrisColeTech, leejet (1 each).

## Notes — the rules the seed follows

- **GGUF, bf16/fp16/fp32 or fp8, never int8.** ComfyUI's `int8_tensorwise`
  (convrot or not) loads as `GGML_TYPE_I8`, which has no matmul kernel on HIP or
  Vulkan, so the whole diffusion model computes on the CPU
  (leejet/stable-diffusion.cpp#1929, upstream `docs/int8_convrot.md`). fp8,
  scaled or not, keeps its type and is cast to bf16 per layer on a backend
  without an fp8 kernel: it stays on the GPU (seen with SenseNova's turbo merge
  on ROCm). Ideogram 4 seeds an `int8_convrot` pair beside its fp8 pair: that
  pair is the exception, and generates on the CPU.
- `ChrisColeTech/SenseNova-turbo-FP8` is SenseNova U1.5 with its 8-step LoRA
  merged (16.3 GiB against 32.7 for the official shards), good at 8 steps and
  CFG 1; it carries those two numbers as model parameters, so assigning it is
  enough.
- **Source preference**: unsloth's GGUF where it exists (`Q4_K_M` for diffusion
  models, `UD-Q4_K_XL` for LLM repositories, which are the only ones with the
  dynamic XL line); else a GGUF the model's own sd-cpp doc names, whoever owns
  the repository (`realrebelai/KREA-2_GGUFs`, `QuantStack/Wan2.2-T2V-A14B-GGUF`,
  `city96/umt5-xxl-encoder-gguf`, `vantagewithai/LTX-2.5-GGUF`); else Comfy-Org's
  repackage (VAEs, PiD weights, Mage-Flow); else the vendor. VAEs and mmproj
  files stay as shipped.
- **When the doc's source cannot be seeded**, the nearest file that can:
  Microsoft withdrew the Mage-Flow repositories the sd-cpp doc links, and
  Comfy-Org's `Mage-Flow` carries the same bytes (LFS sha256 identical to every
  community mirror of them). SenseNova U1.5's doc passes the official sharded
  repository as a directory; drift seeds its index instead (sd-cpp loads either,
  `34`), and beside it `realrebelai`'s community `Q5_K_M` GGUF (official tensor
  names) for smaller memory.
- The three image models added on 2026-09-18 follow sd-cpp's own docs
  (`docs/boogu_image.md`, `docs/ernie_image.md`, `docs/hidream_o1_image.md`).
  Boogu Image takes a diffusion model, the FLUX.1 VAE and Qwen3-VL-8B as its
  text encoder — Comfy-Org packages all three in `Comfy-Org/Boogu-Image`, so
  `flux1-vae` and the fp8 Qwen3-VL come from there. Boogu Image Edit is its
  own architecture beside it, as Mage-Flow's is: the same VAE and text encoder
  plus the Qwen3-VL-8B vision projector on `--llm_vision`, since the reference
  image is read by the text encoder. ERNIE Image takes the FLUX.2
  VAE and Ministral 3B, and carries `--diffusion-fa` with **`--attn-scale
  0.0078125`** beside it. Bare `--diffusion-fa` decodes ERNIE Image at
  1024×1024 to a uniformly white image on ROCm (gfx1151), which is what a NaN
  latent clamps to — bisected on 2026-09-18, and still reproducing on sd-cpp
  master-881-17860c0 with the official `ernie-image-turbo-UD-Q4_K_M` weights.
  sd-cpp's `docs/troubleshooting.md` names that overflow and the cure:
  `--attn-scale` rescales flash attention's K and V. Measured on 2026-09-20
  with `sd-cli` outside drift, one prompt and one seed at 1024×1024:
  `--attn-scale` alone fixes it at every value from 0.5 down to 0.00390625,
  `--linear-scale` alone does **not** (still white), and the flag pair halves
  the run — 97.5s without flash attention, 49.3s with. The value is not
  critical: edge energy is flat across the range and the images differ from
  each other by as much as they differ from the no-flash-attention render, so
  0.0078125 is taken from the doc for its margin. The threshold is weight
  dependent, which is why margin matters: the community fp8 checkpoints
  (Big Love, JibMix) never overflow and render the same picture with the scale
  on (RMSE 0.025) as without it, so the flag pair is harmless where it is not
  needed. Every other seeded architecture keeps bare `--diffusion-fa`: the cure
  is one flag away if one of them ever comes out white. HiDream O1 is
  one file loaded with `--model`, VAE and text encoder included, like
  SenseNova's.
- Their distilled checkpoints set the architecture defaults (Boogu Turbo and
  Edit Turbo: 4 steps, CFG 1; ERNIE Turbo: 8 steps, CFG 1; HiDream O1 Dev: 28
  steps, CFG 1, flow shift 1), and the full checkpoints carry their own
  parameters (Boogu Base: 25 steps at CFG 4; Boogu Edit: 25 at CFG 5; ERNIE:
  50 at CFG 5; HiDream O1: 50 at CFG 5, flow shift 3). CFG 1 in sd-cpp is what "guidance 0" is on the vendors' cards: no
  classifier-free guidance.
- Browsing for their LoRAs works on all three sites: Civitai files them under
  `Boogu`, `Ernie` and `HiDream-O1`; HuggingFace under the vendors'
  repositories (`Boogu/Boogu-Image-0.1-*`, `baidu/ERNIE-Image*`,
  `HiDream-ai/HiDream-O1-Image*`); ModelScope under `Boogu/Boogu-Image-0.1-*`,
  `PaddlePaddle/ERNIE-Image` (Baidu publishes there, not under `baidu/`) and
  `HiDream-ai/HiDream-O1-Image*`, each with its `@master` revision.
- **Qwen Image 2.1** (seeded 2026-09-20, ahead of the binary) follows
  `docs/qwen_image_2.1.md`. It is not Qwen Image with a new number: 7B of
  single-stream DiT against 2512's 20B MMDiT, its own VAE, Qwen3-VL-8B-Instruct
  as text encoder, and one checkpoint that both generates and edits — so it is
  one architecture tagged `image` and `edit` rather than the pair Boogu and
  Mage-Flow need, with the vision projector on `--llm_vision` in the only
  **optional** slot of an image architecture: editing reads the references
  through the text encoder, generating from a prompt alone does not. Its VAE is
  its own (`qwen_image_2.1_vae_bf16`); the 2512 one does not fit, which is why
  it gets a family rather than sharing `qwen-image-vae`. The diffusion slot
  seeds leejet's `Q4_K` (4.2 GB, the doc's own GGUF repository — unsloth has no
  2.1 repository) and Comfy-Org's `bf16` (14.2 GB, small enough to be the
  quality reference on a 128 GB machine); the `int8_convrot` pair the doc's
  example uses is skipped, as int8 computes on the CPU. It is also what
  brought `sizeMultiple` in — 32 for it, read off upstream's
  `get_scale_factor` × `get_diffusion_model_down_factor` (16 × 2). The same
  pair gives 32 to SenseNova U1.5 (1 × 32), LTX 2.3 and 2.5 (32 × 1),
  MiniMax-H3 and Wan 2.2 5B (16 × 2, the TI2V VAE), which is seeded with them
  even though only an image architecture tiles today; every other family lands
  on 16 or below, which 16 covers ([`27`](27-redraw.md)).
- **Qwen Image 2.1's numbers are not settled**: the architecture carries the
  sd-cpp doc's `--cfg-scale 6.0 --sampling-method euler` with 25 steps from
  leejet's ComfyUI example workflow (`qwen_image_2_1_t2i_gguf.json`) — which
  itself runs **CFG 1**, while the vendor's diffusers card shows 40 steps at
  2048×2048. Real classifier-free guidance either way: no `--guidance`, and the
  negative prompt bites at CFG 6. One comparison on the first live run settles
  it.
- **No speed LoRA for Qwen Image 2.1 yet** (checked 2026-09-20, hours after the
  release): no adapter on HuggingFace declares `Qwen/Qwen-Image-2.1` as its base
  model, lightx2v has published nothing since MiniMax-H3, and ModelScope holds
  only style LoRAs for it. The existing Qwen-Image-Lightning LoRAs are for the
  20B 2512 DiT and cannot load on this one. The architecture already names the
  base model on both sites, so the browsers list 2.1 adapters as they appear;
  a 4- or 8-step LoRA goes in `loras.json` once one exists ([`33`](33-lora-sources.md)).
- **SenseNova quants are fragile at 4 bits**: a K-quant of the layer-0
  understanding `mlp.down_proj` corrupts the prefix cache and the image with it
  (realrebelai withdrew its `Q4_K_S` and `Q4_K_M` for that). Nothing below
  `Q5_K_M` from that line.
- **HunyuanVideo 1.5**: the sd-cpp doc names Comfy-Org's repackage and no
  GGUF, so its 720p text-to-video fp16 checkpoint is seeded, with the 480p
  cfg-distilled fp8_scaled one (half the size, CFG 1 and 848×480 as model
  parameters) beside it. Image-to-video checkpoints are left out: the doc
  shows text-to-video only. Civitai files 1.5 under the same "Hunyuan Video"
  base model as the first version, whose LoRAs do not fit 1.5.
- **Text encoders** are substituted with a stock GGUF only after checking the
  vendor's really is stock (Klein 4B and Z-Image carry base Qwen3-4B, Klein 9B
  base Qwen3-8B, Krea 2 Qwen3-VL-4B-Instruct); model-specific encoders (LTX's
  `with-proj` Gemma, H3's Qwen3-VL, LTX-2.3's embeddings connectors) come from the
  model's own repository.
- **Tokenizers** are seeded where sd-cpp embeds none (PiD's Gemma 2): the
  `tokenizer.json` from unsloth's open `gemma-2-2b-it` mirror (the same bytes
  as its base `gemma-2-2b`), where `google/gemma-2-2b`, the file sd-cpp's doc
  names, is gated.
- **Every repo/file pair is verified before being written** with a `HEAD` on
  `https://huggingface.co/<repo>/resolve/main/<file>`: a wrong path in seed data
  looks authoritative and fails minutes into a load. `Lightricks/LTX-2.5` is
  gated: its files answer 401 until the licence is accepted and a HuggingFace
  token is saved in Settings; every other seeded repository is open.
- LTX-2.5's video VAE must be the **conv** variant; the default one is a
  diffusion decoder sd-cpp does not implement.
- **Boot the backend on an isolated `DRIFT_CONFIG_DIR` after editing the file**:
  `loadResource` swallows a decode failure into a warning and seeds nothing, so
  bad JSON is otherwise silent.

## Post-v1

- A `sha256` and a size per seeded model, so the cache can verify without a
  lookup and the UI can say how much a slot will download.
- Per-model `parameters` on the other seeded turbo checkpoints (Z-Image, Krea 2,
  Mage-Flow), so a first generation uses the right numbers without relying on
  the architecture defaults.
