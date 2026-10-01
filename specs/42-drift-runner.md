# 42 — drift runner: an inference engine for Strix Halo

**Status:** partial — steps 1 to 9 done: drift can launch the runner on Qwen 3 and Qwen 3.6 35B-A3B, with MTP drafting and a prefix cache; step 11's image vision done (Qwen 3.6, Qwen 3.8 Flash Next, Qwen Image 2.1 editing); Qwen 3.8 27B (dense) chats; HiDream O1 draws; step 14's MiniMax H3, Wan 2.2 A14B and LTX 2.5 make videos (H3 and LTX with their soundtracks), with LoRAs and image inputs (H3 also guides and a ControlNet, its references built but broken live; LTX guides); groom each later step before building it
**Depends on:** 06 (runtimes, TheRock), 07 (launch and supervision), 16 (parameter resolution), 17 and 18 (the chat runtime and its sessions), 41 (text projects)

drift's own runner replaces sd-cpp and llama.cpp for one machine: Strix Halo
(gfx1151, RDNA 3.5, 128 GB of unified memory, about 256 GB/s). It must run
every architecture drift describes (image, edit, video with audio, upscale,
chat) and more chat models. Qwen 3.8 Flash Next (125B, 6B active) with its MTP head is the first new
one. Everything except the GPU kernels is Scala on the JVM. The kernels are
HIP and are compiled for gfx1151 only. The runner does not have to be portable.
It has to be faster or more correct than the general-purpose tools on this one
machine, and it is only worth building if it is.

## What it does

- **A llama.cpp runtime, for now.** The runner takes llama-server's flags
  and speaks its API, so drift adopts it as a llama.cpp runtime (a folder
  whose `llama-server` starts the runner), and a chat configuration picks
  the runner or llama.cpp in its launch control. That needs no drift code.
  A tool of its own (`RuntimeTool.Drift`) comes when drift needs to tell the
  runner apart: its version, diffusion (where it would take sd-server's
  flags), or the parameter layer below. A model the runner cannot run yet
  stays on sd-cpp or llama.cpp; the runner never replaces them all at once.
- **The same wire.** The runner serves the HTTP API drift already speaks:
  sd-server's (`/v1/img_gen`, `/v1/vid_gen`, `/v1/jobs/…`, `/v1/capabilities`,
  `/v1/models`) and llama-server's OpenAI chat (`/v1/chat/completions`,
  streaming, vision). It emits the same progress and log lines. Launching,
  supervision, pause and resume, the status socket and chat sessions are
  unchanged ([`07`](07-launch-and-supervision.md), [`18`](18-assistant-models-and-sessions.md), [`40`](40-pause-and-resume.md)).
- **Its own process.** It is a separate JVM launched and supervised like
  sd-server. A GPU fault kills the runner and not drift, stopping it frees all
  of its memory, and it gets its own JVM flags.
- **Chat.** It covers today's two chat architectures, Qwen 3.6 35B-A3B and
  Gemma 4 26B-A4B (with its separate `mtp` file), and Qwen 3.8 Flash Next. It
  supports MTP speculative decoding, vision through the mmproj encoder, chat
  LoRAs ([`35`](35-assistant-loras.md)), and a prefix cache that keeps a text
  project's conversation ([`41`](41-text-projects.md)) warm between turns.
- **Image, edit, video, upscale.** It covers every sd-cpp architecture in
  [`01`](01-architecture-registry.md), with LoRAs, tiled VAE, masks,
  reference images, hires, ESRGAN and PiD.

## Shape

### How much native code, and where

Nearly all of the compute time goes into a small set of operations. Those
operations are the only native code:

| Kernel group | Used by |
|---|---|
| Dense GEMM, bf16/fp16 (hipBLASLt; own kernels only where fusion pays) | everything, prefill, diffusion |
| Fused dequant GEMV/GEMM: GGUF K-quants, q8_0, ROCmFP4, int8, fp8 (dequantized in registers; gfx1151 has no fp8 math) | LLM decode, quantized diffusion weights |
| Grouped GEMM over routed experts, top-k routing | MoE: Qwen 3.6/3.8, Gemma 4 |
| Flash attention: prefill, decode over a paged KV cache, sliding window | everything |
| Gated linear attention (chunked prefill, recurrent decode) | the hybrid Qwen line |
| conv2d, causal conv3d, group norm, up/downsample | VAEs (image, video, audio), ESRGAN |
| RMS/layer norm, RoPE (1-D, 2-D, 3-D), activations, softmax, modulation, residuals, fused where possible | everything |
| Sampling: top-k/p, temperature, MTP acceptance | LLM |

Estimate: **8–15k lines of HIP** for full coverage. Plain bf16 GEMMs go to
hipBLASLt, and the hand-written work is the fused dequant, attention and
linear-attention kernels. The Scala side is about **25–40k lines**. It holds
the loaders, the tensor layer, the models, the schedulers, the tokenizer,
the chat templates and the server.

### JVM ↔ GPU: FFM, no C++ host code

- **Java FFM** (Panama) calls the HIP runtime and hipBLASLt directly. There
  is no JNI and no `.so` of drift's own. The bindings are written by hand in
  `native/HipRuntime.scala`, one downcall per function used, each repeating
  its C signature. jextract would generate thousands of lines to regenerate
  for every TheRock pairing, for a few dozen functions. Calls go through
  `MethodHandle.invokeExact`, which Scala 3 types exactly, so nothing is
  boxed.
- **Kernels** are `extern "C" __global__` functions compiled by a mill task
  with `hipcc --genco --offload-arch=gfx1151`, one code object per `.hip`
  file, served as the resource `kernels/<name>.hsaco`. `KernelModule` loads
  it with `hipModuleLoadData` and `HipRuntime.launch` runs kernels with
  `hipModuleLaunchKernel`. Adding
  hipRTC later would let Scala generate shape-specialized kernels at model
  load, cached on disk. AOT comes first.
- **JDK 25** for the runner alone. FFM is final from 22 on. mill sets the JVM
  per module, so the backend can stay on 21. The runner JVM runs with
  `--enable-native-access=ALL-UNNAMED`.
- **Off-heap everywhere.** A `Tensor` is a descriptor (dtype, shape, strides,
  quant layout) over a `MemorySegment`. Weights, activations, KV caches and
  linear-attention states never live on the heap. The garbage collector only
  sees descriptors, so the heap stays small.
- **Unified memory.** `FileChannel.map` gives weights as one segment of any
  size. Registering that mapping with the GPU (`hipHostRegister`) gives zero-copy
  weights and a load at page-cache speed (`native/RegisteredFile.scala`,
  `Storage.Registered`). Measured in step 1, a streaming read runs at
  237.6 GB/s from a registered read-only mapping and 238.3 GB/s from
  `hipMalloc` memory, about 93% of the peak. Weights are therefore never
  copied.
- **Launch overhead.** A decode step is hundreds of small kernels. The step is
  recorded once per shape and replayed as a HIP graph.

Rejected integrations:

| | Why not |
|---|---|
| JNI + C++ host library | FFM does the same job without native glue to build and ship |
| WASM (Chicory, GraalWasm) | No GPU and slower SIMD. The whole problem is the GPU |
| TornadoVM, Babylon HAT (kernels written in Java) | Won't get close to hand-tuned attention or dequant GEMM on gfx1151 |
| Vulkan compute | A later option. HIP brings hipBLASLt, WMMA intrinsics and C++, and drift already manages TheRock |

### Module and packages

The runner is one mill module, `runner`. It depends on nothing else in drift:
drift talks to it only over HTTP.

```
runner/
  kernels/          HIP sources; a mill task builds them into one gfx1151 code object (a resource)
  src/drift/runner/
    native/         hand-written FFM bindings (HIP, hipBLASLt), module loading, launches
    tensor/         Tensor, dtypes and quant plug-ins, allocators, the arena planner
    formats/        safetensors, GGUF, config.json, ROCmFP4 identification
    ops/            the Ops interface and its two backends: Cpu (reference) and Hip
    plan/           the recorded plan of a forward pass, its lifetimes, HIP graph replay
    blocks/         attention variants, MLPs and MoE, norms, residual variants, DiT and VAE blocks, ViT
    models/         one file per family, written against blocks
    state/          paged KV cache, indexer cache, recurrent and conv states, snapshots, prefix cache
    text/           tokenizers (byte-level BPE, character BPE with byte fallback), the Jinja interpreter for chat templates
    decode/         samplers, stop conditions, grammars, MTP speculation
    diffusion/      schedulers, sampler loop, guidance, cache modes, tiling
    server/         the sd-server and llama-server APIs, progress, logs
  test/             utest, CPU only, runs everywhere
  gpuTest/          utest, needs gfx1151: every kernel against the Cpu backend
  fixtures/         Python (uv) generator for golden tensors and the ROCmFP4 C harness, run by hand; outputs in test/resources
```

**Everything is written once, against `Ops`.** Blocks, models and pipelines
never call a kernel directly. They call `Ops`, which has two backends:

- **`Cpu`**: plain Scala, fp32 with fp64 accumulation, slow and obviously
  correct. It is the oracle.
- **`Hip`**: the kernels.

The same model code therefore runs on both backends. A tiny model on `Cpu`
checked against the Python reference validates the model code. The same
tiny model on `Hip` checked against `Cpu` validates the kernels. One check
never has to do both jobs.

**A forward pass is a plan.** A model's forward records ops into a plan
(shapes, dtypes, lifetimes). The planner lays out the activations in one
arena. `Hip` replays the plan as a HIP graph, and `Cpu` interprets it. The
plan is also where debug mode hooks in: it checks every op's output for
NaN or Inf and names the first op that produced one, and it records a timing
per op.

### Weight formats

The runner reads two containers, both from a memory-mapped file, and nothing
else:

- **safetensors**, plain or sharded through `model.safetensors.index.json`.
  Supported dtypes are F32, F16, BF16, fp8 E4M3/E5M2 (plain or with scale
  tensors, dequantized in registers), and ComfyUI's int8 tensorwise format,
  which runs on the int8 matrix units instead of falling back to the CPU. A
  bare safetensors file has no hyperparameters. They are inferred from tensor
  shapes and checked against the family's constants, or read from a
  `config.json` next to the weights.
- **GGUF**: F16/BF16, Q8_0, Q4_0/Q4_1/Q5_x, the K-quants (Q2_K to Q6_K), and
  ROCmFP4 (below). The IQ quants and MXFP4 are added when a model needs them.
  A GGUF file carries its hyperparameters and tokenizer.

Types are chosen per tensor, never per file. A ROCmFP4 model keeps f16
embeddings and a Q6_K output head.

**Weight names** (`models/WeightNames`, bug 32). `WeightSource` gives every
tensor the name the loaders read, whoever exported the file, as sd-cpp's
`name_conversion.cpp` does; no loader has a prefix or naming of its own.
ComfyUI's and the single files' prefixes are dropped (`model.diffusion_model.`,
`diffusion_model.`, PiD's `net.`, `first_stage_model.`, `vae.`); a diffusers
`AutoencoderKL` (ERNIE-Image's `flux2-vae`) gets the LDM names `FluxVae`
reads, the decoder's levels counted from the other end. Two stored tensors
coming to one name are refused by name. Not mapped yet, as none of the
configurations has one: diffusers-named transformers (Krea 2's, FLUX.2's)
and Wan's diffusers VAE.

**A quant type is a plug-in**, not a case in a match. A type is its block
layout (bytes per block, weights per block), a dequant-to-registers routine,
a decode GEMV kernel, a prefill GEMM kernel, and optionally a CPU reference
decoder used for tests. Adding a type means adding one of these and changing
nothing else. In the code, a type is a `DType` object (`tensor/DType.scala`,
`GgmlQuants`, `KQuants`, `RocmFp4`) listed in `DType.all`. Its CPU decoder
repeats the arithmetic of the format's own reference operation by
operation, in float, so it matches that reference bit for bit.

The readers (`formats/`) parse a whole-file `MemorySegment`, either a
`MappedFile` or the host side of a `RegisteredFile`. A `StoredTensor` is a
slice of that mapping, never a copy. A tensor whose type the runner cannot
decode (an IQ quant, an I64 position table) is listed in `unsupported` and
fails only when it is asked for. GGUF dimensions are innermost first, so the
reader reverses them once into a `Shape`. The backend's header inspectors
([`24`](24-model-details-in-browsers.md), [`25`](25-model-conversion.md))
read onto the heap and skip GGUF metadata values, so the runner does not
share them. They could move onto these readers later.

**ROCmFP4** is an unofficial 4-bit format for gfx1151, from the rocmfp4-llama
fork of llama.cpp (`charlie12345/ROCmFPX`,
`PlunderStruck/rocmfp4-turboquant`):

- `Q4_0_ROCMFP4`: 32 weights in 18 bytes (4.5 bits per weight). Two
  unsigned E4M3 scale bytes, one per 16-weight half.
- `Q4_0_ROCMFP4_FAST`: 32 weights in 17 bytes (4.25 bits per weight). One
  scale for the whole block.
- Values are 4-bit, E2M1-derived codes into a signed "Codebook10"
  (`0 1 2 3 4 6 8 10`, then the same negated), times half the unsigned E4M3
  scale. Element `j` is the low nibble of byte `j` for `j < 16`, and the high
  nibble of byte `j − 16` after. The 16 code bytes come first, then the
  scale bytes. Scale bytes outside `0x00–0x7e` are rejected.
- GGUF type ids are 100 (dual) and 101 (fast).
- The kernels there decode with `amdgcn_perm` lookups and multiply with
  integer dot products (MMVQ for decode, MMQ for batches). They avoid WMMA,
  which slowed MTP decode.

The format is not upstream, so its GGUF type ids belong to the fork and could
collide with a future official type. The loader recognizes a ROCmFP4 file by
more than the type id alone, and it refuses a file it cannot identify
instead of misreading it. Two checks apply:

- For every tensor of every type, the byte size its type gives must fill the
  room the file leaves before the next tensor, give or take the alignment
  padding. A type read as the wrong type nearly always breaks that.
- ROCmFP4's scale bytes must be finite E4M3 in the first 4096 blocks of
  each tensor. The fork's published numbers are the chat
benchmark to beat, not upstream llama.cpp.

The fork also has **TurboQuant**, a KV-cache quantization with Walsh–Hadamard
rotation (turbo3/turbo4). The KV cache's element type is therefore a
parameter of the cache, like a weight quant type, even though the first
version uses F16 (more mantissa than BF16, and ample range for keys and values).

**Pickle** (`.pth`, `.pt`, `.ckpt`) is never read by the runner. drift
accepts pickle today for ESRGAN upscalers and LoRAs because sd-cpp loads it.
For the runner, such a file is converted to safetensors when it is installed,
with the conversion machinery of [`25`](25-model-conversion.md). That
converter uses a restricted unpickler that rebuilds tensors and refuses any
other global, because arbitrary pickle can execute code.

### Chat requirements that shape the design

- **Two kinds of per-sequence state.** Full attention keeps a KV cache that
  grows with the context, stored in pages. Gated linear attention keeps a
  fixed-size recurrent state per layer. A sequence owns both, and the model
  says which layer uses which.
- **MTP speculation.** The MTP head drafts *k* tokens, one batched forward of
  the main model verifies them, and the accepted prefix is kept. Rolling back
  a rejected suffix is trivial for the KV cache (truncate the pages) but not
  for the recurrent state. That state is not small: on Qwen 3.8 Flash Next it
  is fp32, 48 heads × 128 × 128 per layer across 36 layers, about 113 MB per
  sequence. The verifying forward has the recurrent kernels keep the state
  after every token (a few verified tokens' worth), and a rejection copies
  the accepted one back. MTP weights come either from the model file (Qwen)
  or from a sidecar slot (Gemma's `mtp` checkpoint). The model definition
  declares which.
- **MoE decode is bandwidth-bound.** Tokens per second are roughly 256 GB/s
  divided by the active weight bytes per token. That makes the fused
  dequant GEMV the kernel that matters most for chat.
- **Prefix cache.** Pages of a conversation's KV cache (and a snapshot of its
  recurrent state) are kept after a reply. The next turn only prefills the
  new messages.
- **Templates and tokens.** The runner has its own tokenizer, read from
  `tokenizer.json` or from the GGUF vocabulary. Chat templates are the
  model's own Jinja, rendered by the runner's own interpreter, so thinking
  modes and tool formats come from the model and are not reimplemented.
  There is no native tokenizer and no JVM Jinja library: a GGUF's
  vocabulary is read directly, one implementation serves GGUF and
  `tokenizer.json` (which the diffusion text encoders ship), and
  HuggingFace's own tokenizer (DJL) and transformers serve only as test
  references.

### Qwen 3.8 Flash Next

[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
is `qwen4_exp` in the transformers 5.8 dev branch, and that code is the
correctness reference. It has 125B parameters, 6B of them active per token.
51B of the total are n-gram embeddings, and 4B are the MTP layer. The weights
are released in BF16 as safetensors, and quantized GGUFs exist. What it needs
beyond Qwen 3.6:

| Part | Config | What the runner needs |
|---|---|---|
| 48 layers, 3 Gated DeltaNet : 1 full attention | `layer_types`, `full_attention_interval` 4, conv kernel 4, 16 key / 48 value heads of 128, fp32 state | the gated linear attention kernels, with fp32 state |
| Qwen Sparse Attention on the full layers | indexer: 4 heads, 1 key head of 128, keys compressed 4×, budget 512 blocks (2048 tokens) | an indexer kernel that scores key blocks and picks the top-k, then block-sparse flash attention over the chosen blocks. The indexer's compressed keys are a second cache beside the KV cache, and they are paged, cached and rolled back with it |
| Gated attention output | `output_gate_type` sigmoid | a fused sigmoid gate on the attention output |
| Gated residual | `hc_count` 4 branches, `hc_lowrank` 320 | the residual is 4 streams, not 1: data-dependent read gates and per-branch write gates through a rank-320 bottleneck, fused into the norm and residual kernels |
| N-gram embeddings (51B) | bigrams and trigrams hashed (`ngram_size` 3, `ngram_vocab_size_base` 20M, `heads_per_ngram` 8, `split_ngram_parts` 128), injected at layer 2 through a per-layer embedding with a conv of 4 | only gathered, never multiplied: per token it reads a few rows. The table stays in the memory-mapped file and is never copied into GPU memory. The hashing follows the reference code exactly |
| MoE | 512 experts, 10 routed + 1 shared, intermediate 640 | grouped GEMM with a top-10 router. Experts are small, so the batched GEMV reads many short slices, and that access pattern is benchmarked early |
| Rotary positions | interleaved mRoPE, sections 11/11/10, partial rotary 0.25, θ 10⁷ | 3-D mRoPE on a quarter of each head |
| MTP | 1 full-attention layer (4B), in the same checkpoint | the draft loop of the chat requirements |
| Vision | ViT of 27 layers, patch 16, spatial merge 2, temporal patch 2 (images and video) | the mmproj path, with video frames |

**It only fits quantized.** In BF16 it is 250 GB. At about 4.5 bits per
weight it is roughly 70 GB, which leaves room on the 128 GB machine for the
262k context. Decode reads about 6B × 4.5 bits ≈ 3.4 GB per token, which caps
decode at roughly 75 tok/s without MTP. The runner's target is a large
fraction of that ceiling. drift installs it as a GGUF, or converts the BF16
release with [`25`](25-model-conversion.md). A ROCmFP4 build needs a ROCmFP4
quantizer: the fork's, or one of drift's own.

### Coverage by building block

| Architectures | Needs beyond the shared set |
|---|---|
| qwen3.6-35b-a3b | gated linear attention, MoE, MTP in the file, vision |
| Qwen 3.8 Flash Next | the above, plus sparse attention with an indexer, the gated 4-stream residual, n-gram embeddings, mRoPE, video input |
| gemma-4-26b-a4b | sliding-window attention, MoE, MTP sidecar, vision |
| flux.2-*, z-image-turbo, krea2, qwen-image(-2.1), ideogram-4, ernie-image, boogu-image(-edit), mage-flow(-edit)-turbo | DiT with 2-D RoPE, image VAE, LLM text encoder, reference images for edit |
| hidream-o1, sensenova-u1.5 | single-file unified models: map their tensors onto the blocks |
| wan-2.2-*, hunyuanvideo-1.5, ltx-2.3/2.5, minimax-h3 | 3-D RoPE, causal conv3d video VAE, audio VAE, two-expert switching (Wan 14B) |
| pid-* | the PiD decoder over the DiT blocks, tiling ([`26`](26-tiled-pid.md)) |
| ESRGAN ([`15`](15-post-hoc-resize.md)) | conv net only |

### Variations the design must allow

This list covers what drift runs today and what nearby models use. Each
item is a parameter or a plug-in of a block, never a fork of the code. The
first versions implement only what their models need, but no item may
require restructuring when it arrives.

| Area | Variations |
|---|---|
| Attention | MHA, GQA, MQA, MLA (compressed KV); causal, bidirectional (encoders, ViT, DiT), sliding window, alternating local and global layers; sinks; logit softcap; QK-norm; sigmoid output gate; indexer-selected sparse blocks; cross-attention (Wan); joint text+image attention (Flux double-stream) |
| Positions | RoPE full or partial, NeoX or interleaved layout; YaRN and NTK scaling; mRoPE; 2-D and 3-D axial RoPE for DiTs; learned embeddings (ViT) |
| Norms | RMS (including Gemma's `1 + w`), LayerNorm, GroupNorm, adaLN modulation, pre-, post- and sandwich norms |
| MLP | SwiGLU, GeGLU, GELU (tanh, erf); MoE with shared experts; routing softmax→top-k, top-k→softmax, or sigmoid with bias |
| Residual | plain, layer scale, hyper-connections (n streams, low-rank gates) |
| Embeddings | tied or untied, scaled, per-layer (PLE), hashed n-grams, final logit softcap |
| Sequence state | paged KV, quantized KV (TurboQuant), indexer key cache, gated DeltaNet or Mamba-2 recurrent state (fp32), the short-conv state of linear layers (the last `kernel − 1` inputs, rolled back with the rest) |
| Decoding | temperature, top-k, top-p, min-p, penalties, seeds, stop strings, logprobs, JSON-schema or grammar constraints, reasoning and tool-call parsing, MTP or a draft model, several sequences at once (the assistant and a text project), cancellation |
| Tokenizers | byte-level BPE (Qwen), SentencePiece (Gemma, T5), special and placeholder tokens |
| Vision input | smart resize, patching, video frame sampling, deepstack features |
| Diffusion | flow-matching and epsilon schedules, shift and dynamic shift, Euler, Euler ancestral, DPM++; CFG, distilled guidance, negative prompts; img2img strength, masks, reference latents; step-skipping caches; LoRA, LoKr and LoHa as side paths; unified LLM+diffusion models (HiDream O1, SenseNova) |
| Decoders | 2-D KL VAE, causal 3-D video VAE, LTX's patchified VAE, audio VAE, tiled with overlap blending, PiD, ESRGAN |
| Runtime | memory plan checked before loading (refuse with a reason), progress events, pause at a step boundary, per-op timings, NaN debug mode |

**Seeds do not carry over.** The runner's random noise is not sd-cpp's, so a
gallery seed reproduces an image only on the runner that made it. The runner
is recorded with the image, like the configuration is.

### Testing

The maths cannot be checked by eye, so each layer of the tests checks one
thing against an independent reference. Nothing needs a real model except
the last layer.

1. **Formats and quant types, bit-exact.** Every quant type's CPU decoder is
   compared with `gguf-py`'s numpy dequantizers on blocks the fixture
   script quantizes. ROCmFP4 is compared with the fork's CPU reference. The
   safetensors and GGUF readers are tested on small generated files,
   including sharded ones, a truncated one and an unknown type id.
2. **Ops, `Hip` against `Cpu`** (`gpuTest`). Random inputs, per-dtype
   tolerances, and awkward shapes: a single token, sizes that are not a
   multiple of the block, long sequences, strided views. Each kernel has
   its test before it has a caller.
3. **Invariants**, properties that must hold whatever the weights. They run
   on `Cpu` in `test` and on `Hip` in `gpuTest`:
   - flash attention equals naive attention;
   - paged KV equals contiguous KV;
   - chunked DeltaNet prefill equals the step-by-step recurrence;
   - sparse attention with a budget covering the context equals dense attention;
   - prefill + decode equals one longer prefill;
   - a prefix-cache hit equals a fresh prefill;
   - rollback equals never having drafted;
   - greedy decoding with MTP gives exactly the tokens of greedy without it;
   - a tiled VAE decode equals the untiled one away from the seams;
   - the ops sum to 1 and are invariant where the maths says they should be.
4. **Golden tiny models.** The fixture script instantiates each family from
   its config shrunk down (2 to 4 layers, small widths, few experts, and the
   same layer types), with seeded random weights. It uses transformers for
   the chat models and text encoders and diffusers for the DiTs and VAEs. It
   dumps the weights, the inputs and every layer's output to safetensors.
   The runner loads the same weights through its own loaders and compares
   layer by layer on `Cpu`, then `Hip`. The fixtures weigh kilobytes to a
   few megabytes and are committed, so the tests need no Python. Tokenizers
   and chat templates are checked the same way: token ids and rendered
   strings from `tokenizers` and `apply_chat_template` over a corpus of
   edge cases (unicode, special tokens, tool calls, thinking).
5. **Real models, opt-in.** These are tagged tests run when a step is
   closed, not in every build:
   - the first-token logits against llama.cpp on the same GGUF (top-k
     overlap and KL divergence);
   - WikiText-2 perplexity against llama.cpp's number;
   - an image at a fixed seed against diffusers (PSNR), on test subjects
     generated for the purpose;
   - the benchmarks.

A step is done when its layers 1–4 pass and layer 5 has been run once. The
spec records the numbers from that run.

## Notes

- **Why it can win.** It is specialized for one GPU target, so it has no
  per-backend branching. It has no offloading logic, because 128 GB fits
  every model. Weights load zero-copy. Kernels are fused per family. Its
  attention is its own, so the ROCm flash-attention NaNs and `--attn-scale`
  workarounds go away. Int8 weights no longer fall back to the CPU. If a
  step's benchmark doesn't show a win over the tool it replaces, that
  model stays on the old tool.
- **Correctness reference.** Each model is checked against the reference
  implementation (transformers or diffusers) on fixed inputs, tensor by
  tensor. For image models, a fixed-seed image is compared as well, on test
  subjects generated for the purpose.
- **The new chat blocks are less settled than the rest.** Three parts of Qwen
  3.8 Flash Next exist only in the transformers dev branch: sparse attention,
  the gated residual and the n-gram hashing. The runner ports them from that
  code, not from a paper, and checks them tensor by tensor.
- **The TheRock tree is the whole toolchain.** The trees drift manages
  ([`06`](06-sdcpp-runtime.md)) include `hipcc`, amdclang, the HIP and
  hipBLASLt headers and the libraries. The build's `runner.rocmRoot` takes
  `DRIFT_ROCM_ROOT`, or else the newest stable `gfx1151-*` tree under drift's
  runtimes. The runner reads `DRIFT_ROCM_ROOT` only: drift sets it at launch,
  and the build sets it for the tests. Building the runner therefore needs
  a TheRock tree, even for the CPU-only tests, because the kernels are
  resources of the main module.
- **JDK 25 comes from mill** (`jvmId = "temurin:25"` in the `RunnerJvm`
  trait), with `--enable-native-access=ALL-UNNAMED`. The backend stays on
  21.
- How the runner is shipped follows the undecided distribution questions and
  is not settled here.

- **gufo** ([gufo-org/gufo](https://github.com/gufo-org/gufo), MIT, C++/HIP,
  gfx1151 only) is an engine for Strix Halo that runs Qwen 3.8 Flash Next
  (with MTP and QSA), Qwen 3.8 27B and others. It is a second reference next to
  llama.cpp: a CPU oracle for qwen4exp (`reference.cpp`), and designs worth
  borrowing:
  - **compact DeltaNet rollback:** a full state, then only each draft's
    rank-1 operands;
  - **HIP-graph replay** keyed by batch shape, with positions in a device
    control block, split at the PLE layer;
  - **verify-batch experts grouped by expert**, with slot masks;
  - **fused hyper-connection combine**, next grouped norm and MoE epilogue;
    the router and the shared gate as one product;
  - **the n-gram table read with O_DIRECT** by worker threads.

  Its README reports llama.cpp at 22.2 tok/s decode on Qwen 3.8 Flash Next
  (Q4, empty context) and gufo at 26.0. It measured single-shape GEMV loops
  reading 400–860 GB/s out of the 32 MB MALL: a benchmark must stream the
  whole per-token footprint. Its hnorm reading lost to llama.cpp's here (step
  10.5).

## Remaining

Fifteen steps in four milestones. Each step ends in its tests (see
Testing). Each step that brings a model ends in a benchmark against the tool
it replaces, on the same weights. If the runner doesn't win, the model stays
on the old tool.

**Foundation. No model yet, everything tested.**

1. ~~**Module and the metal.**~~ Done. It has the `runner` module on JDK 25,
   the kernel build task, the FFM bindings, device memory, streams, events
   and a registered mapped file. `Ops` has both backends with `add` and
   `rmsNorm`, and there is the tolerance comparator (`tensor/Comparison`).
   `runner.test` has 9 tests and `runner.gpuTest` has 5. `MemoryBandwidth`
   (a main in `gpuTest`) measures registered memory against device memory.
2. ~~**Formats.**~~ Done. safetensors (sharded, the scale-tensor pairing),
   GGUF v2/v3 with every metadata type, `config.json` (`ModelConfig`), CPU
   decoders for F32, F16, BF16, fp8 E4M3/E5M2, I8, U8, Q4_0, Q4_1, Q5_0,
   Q5_1, Q8_0, Q2_K–Q6_K and both ROCmFP4 layouts, and ROCmFP4
   identification. The golden fixtures (160 KB, `runner/test/resources/fixtures`)
   come from `runner/fixtures/generate.py`: gguf-py for GGML's types, the
   fork's `rocmfp4.c` through a small C harness for ROCmFP4, ml_dtypes for
   fp8. All of it is pinned to the fork's commit `9c37bc7`. The quant blocks
   are random bytes, so every bit pattern is exercised, and all 256 fp8 codes
   are covered. `runner.test` has 40 tests.
   Pure CPU.
3. ~~**Core kernels.**~~ Done, and the gate passed. `Ops` covers `linear`,
   `add`, `mul`, `scale`, `addRow`, `activation` (SiLU, GELU tanh/erf,
   sigmoid), `gated` (SwiGLU, GeGLU), `convert` (F32 ↔ F16/BF16), `rmsNorm`
   (with Gemma's `1 + w`), `layerNorm`, `softmax` and `rope`. `rope` covers
   NeoX and interleaved pairs, partial rotary, and mRoPE both contiguous and
   interleaved as Qwen 3.8 uses. `runner.test` has 48 tests and
   `runner.gpuTest` has 24.

   **Decode (`linear`, M ≤ 8, `kernels/matvec.hip`)**, measured on a
   16384×8192 weight, which is too large for the Infinity Cache. The
   llama.cpp column is the fork's `test-backend-ops perf -o MUL_MAT`, built
   for gfx1151 with cases added at this shape. Its own 4096×14336 case
   (33 MB) mostly stays in the 32 MB cache between runs, so it measures the
   cache and not decode.

   | Type | llama.cpp / fork, M=1 | runner, M=1 | llama.cpp, M=4 / 8 | runner, M=4 / 8 |
   |---|---|---|---|---|
   | Q4_K | 398–413 µs | 364 µs | — / 1346 µs | 469 / 674 µs |
   | Q8_0 | 659 µs | 639 µs | 661 / 893 µs | 645 / 724 µs |
   | Q4_0 | 346 µs | 341 µs | — | 430 / 616 µs |
   | Q6_K | 514–546 µs | 498 µs | 767 / — µs | 589 / 819 µs |
   | ROCmFP4 | 351 µs | 365 µs | 397 / — µs | 449 / 629 µs |
   | ROCmFP4 fast | 332 µs | 352 µs | 398 / — µs | 444 / 627 µs |
   | F16 | 1252 µs | 1280 µs | 1772 / 1801 µs | 1348 / 1793 µs |

   How the kernel works: one wave per row. The wave copies about 4 KB of
   blocks per trip into LDS with coalesced non-temporal dword loads, and the
   lanes read their blocks back as aligned words joined by `v_alignbyte`.
   Each lane decodes its 32 weights once and meets up to 8 vectors with
   them. The integer types (Q4_0, Q8_0, Q4_K, Q6_K, ROCmFP4) take x
   quantized per 32 values (`quantize_x`) and multiply four weights per
   `v_dot4` (sudot4); ROCmFP4 codes come from three `v_perm`s. Dense
   F32/F16/BF16 take x as float. The tests hold the integer path to a
   rigorous bound, `Σ|w|·step/2` of x's rounding.

   **Prefill (`linear`, M > 8)**: the weight is dequantized to F16 (skipped
   for F16 weights) and x converted, then hipBLAS `GemmEx` with F16 output
   and F32 accumulation, converted back to F32. At M = 512 the runner takes
   6.8 ms (F16) and 7.9–8.4 ms (quantized), against llama.cpp's 7.1 ms (F16),
   7.3 ms (Q4_0), 8.2 ms (Q4_K) and 9.6 ms (ROCmFP4). Three findings apply
   to gfx1151:
   - An F32 GEMM output runs untuned kernels at 3–6 TFLOPS against about
     27, so the output is F16. It overflows beyond ±65504.
   - `ROCBLAS_USE_HIPBLASLT=1` is 20% faster, and `HipBlas` sets it itself
     before creating its handle.
   - F16 accumulation (`COMPUTE_16F`, llama.cpp's choice) is no faster than
     F32 accumulation, so the runner keeps F32.

   Left for later, where the benchmarks say so:
   - ROCmFP4 with M = 4 to 8 on this int8-x path was about 13% behind the
     fork. It now takes the direct int8 kernels (step 9).
   - Each prefill GEMM dequantizes the weight again (about 1.2 ms on 268 MB
     of F16). The F16 copies could be cached while prefill lasts.
   - Diffusion-only kernels come with their models: GroupNorm, conv2d/3d,
     2-D and 3-D axial RoPE (steps 12–14).

4. ~~**Attention.**~~ Done. `Ops.attention` and `Ops.cacheWrite` work
   over `state/KvCache`, a paged F16 cache. Pages hold a multiple of 16
   tokens. Keys are stored `[page][P][kv head][D]` and values transposed
   within a page, `[page][kv head][D][P]`, so both matrix operands load as
   32 contiguous bytes per lane. `PageAllocator` and `SequencePages` hand
   out pages, and `SequencePages` truncates, for rollback. The options are
   causal or not, a sliding window, a softcap and sinks, for heads of 64,
   128 or 256 and any GQA group.

   The kernel (`kernels/attention.hip`) runs one wave per tile of 16 query
   rows (token × query head of the group, so a group shares every K and V
   load), per kv head, per key split. It computes `Sᵀ = K·Qᵀ` and
   `Oᵀ += Vᵀ·Pᵀ` with `v_wmma_f32_16x16x16_f16`. On gfx1151, accumulator
   element `v` of lane `l` holds row `2v + l/16`, column `l % 16` (probed
   on the hardware). With the products transposed, each lane owns one
   query row: its online softmax needs one shuffle with lane `l ^ 16`, and
   rescaling O is one factor. No lane may skip a WMMA, so masked rows get
   zero weights instead of branching. Decode splits the keys into splits
   of at least 256 keys, and `attention_combine` merges them (flash
   decoding).

   On Qwen 3.8 Flash Next's shape (24 query heads over 2, heads of 256),
   against the fork's `test-backend-ops perf -o FLASH_ATTN_EXT` with these
   cases added:

   | Case | llama.cpp | runner |
   |---|---|---|
   | decode, 1 query, 4096 keys | 51.5 µs | 75.1 µs |
   | decode, 4 queries, 4096 keys | 101.8 µs | 88.0 µs |
   | decode, 1 query, 16384 keys | 215.9 µs | 196.2 µs |
   | decode, 4 queries, 16384 keys | 394.7 µs | 361.3 µs |
   | prefill, 2048 × 2048 causal | 17.0 ms | 6.4 ms |
   | prefill, 2048 × 2048, heads of 128, 32 over 8 | 5.6 ms | 4.4 ms |

   Measuring these needs a warm JVM. Before the JIT compiles the host side
   (FFM calls, struct packing), a small kernel waits on its submission; one
   unmeasured pass over the cases removes that. Once warm, submitting an
   attention call costs 8 to 14 µs of host time. The HIP-graph replay of
   step 6 removes it from decode altogether.

   The tests (`AttentionTests`, `AttentionReferenceTests`) cover:
   - the kernel against the reference, over head sizes, GQA groups of 1, 4
     and 12, prefill, decode and MTP-sized steps with split keys, encoders,
     windows, softcap and sinks (worst error 0.001);
   - the reference: one visible key gives its value, and a sink of equal
     logit halves it;
   - invariants: shuffled pages equal pages in order, prefill then decode
     equals a longer prefill, and a whole-context window changes nothing.

   Left for later:
   - one query at short contexts, 1.5× behind: too few splits to fill the
     GPU, and a fixed setup per split;
   - prefill tiles of 16 rows re-read K and V per tile (8 TFLOPS); tiles of
     64 rows over four waves sharing LDS would cut that fourfold;
   - heads of 512, if a model brings them, need the O accumulator split
     over two waves.

5. ~~**Text.**~~ Done. The tokenizer lives in `text/`:
   - `Tokenizer`: added tokens matched first (longest first); then NFC or
     `▁` spaces; then the pieces, either a regex split with byte-level
     words (Qwen, gpt-oss, `ignore_merges`) or characters with byte
     fallback (Gemma); then `Bpe` (lowest rank first, leftmost among
     equals, as HuggingFace's `Word::merge_all`). `StreamingDecoder`
     releases text only on whole UTF-8 characters.
   - `TokenizerJson` reads the pipelines the runner's models use and
     refuses any other by name.
   - `GgufTokenizer` maps a GGUF pre-tokenizer name (`qwen2`, `qwen35`,
     `gpt-4o`) to the regex of the model's own `tokenizer.json`. Java's
     engine runs those as they are, with `UNICODE_CHARACTER_CLASS` so that
     `\s` is Unicode, as it is on Oniguruma. `gemma4` is character BPE with
     byte fallback, and a BOS is always added, as llama.cpp does.

   Checked against HuggingFace's own tokenizer (DJL, a test dependency) on
   the official `tokenizer.json` of Qwen3, Qwen 3.6, Qwen 3.8, Gemma 4 and
   gpt-oss:
   - 2028 texts (a corpus of edge cases and seeded random mixtures) give
     the same ids and the same decoded text;
   - the four chat GGUFs on disk tokenize exactly as their
     `tokenizer.json`;
   - about 4 M tokens per second.

   Chat templates (`text/jinja`, `ChatTemplate`) are a Jinja interpreter for
   the subset chat templates use, with Python's semantics (truthiness,
   `str`/`repr`, tuples, string and dict methods, `loop`, `namespace`,
   macros, `trim_blocks`, `lstrip_blocks`, transformers' `tojson`). An
   injectable clock serves `strftime_now`. Checked against transformers'
   `apply_chat_template` (`fixtures/chat_templates.py`):
   - 6 conversations × 3 flag sets render identically on the real templates
     of all five models, errors included (gpt-oss refuses image parts);
   - 13 focused cases match Python's jinja2 (`fixtures/jinja_cases.py`).

   Parsing reasoning and tool calls out of replies moved to step 7, where
   the server turns them into OpenAI deltas. Still to come with their
   models: Tekken (Mistral, a byte-level regex to add and check against
   llama.cpp, because its repositories ship no `tokenizer.json`) and Unigram
   (umT5, Wan's text encoder, step 14).

**Chat.**

6. ~~**A dense model.**~~ Done: Qwen3-4B (Q8_0 GGUF) chats end to end on the
   GPU. Its weights are read in place from the registered GGUF, and it uses
   its own tokenizer and chat template, the paged cache and the sampler.
   `ChatDemo` (a main in `gpuTest`) streams a reply.
   - `models/`: `Qwen3` (its config from GGUF metadata or `config.json`,
     GGUF and transformers weight names), `CausalModel`, and `WeightSource`
     (GGUF, safetensors or a shard index, mapped through `Ops.mapFile`: a
     registration for `Hip`, a plain mapping for `Cpu`).
     `HuggingFaceConfigs` finds `rope_theta` at the top level or in
     `rope_parameters` (transformers 5).
   - `decode/`: `Sampler` (greedy; temperature, top-k, top-p, min-p,
     seeded) and `Generator`. The generator owns one sequence's caches and
     pages, prefills in chunks, decodes and streams text. The sampler draws
     from candidates, not the whole row: `Ops.candidates` returns each
     logits row's largest values (one launch for every row of an MTP
     verification; a radix select, then a sort, the lower id first among
     equals), as many as the top-k (at most 1024), 1 when greedy, 256
     without a top-k. Without a top-k the row comes back for the weight of
     the rest, and a nucleus reaching past the candidates sorts it all
     (primitive keys). Same arithmetic in the same order, so the same draws
     as sorting every row (`SamplerTests` against the old sampler).
   - Additions to `Ops`: `embedding` (a gather that decodes any storage
     type), `linears` (several products of one input share its
     preparation), `release`, `writeInts`, and `Tensor.view` and `rows`.
     Per-step inputs are written into workspace buffers, because a
     `hipFree` waits for the GPU.

   Checks:
   - The golden tiny Qwen 3 (`fixtures/tiny_models.py`, transformers,
     random weights): the logits match to 0.022% of the largest on `Cpu`,
     and to 0.027–0.076% on `Hip`, all at once and as prefill then
     decode.
   - The generator's chunked prefill and decoding give the tokens of a
     naive loop that re-runs the whole sequence each step.
   - Against llama.cpp on the same GGUF (`LlamaComparison`): the same most
     likely token at 24 of 24 steps, and top-10 overlaps of 9–10.

   **Numerics: int8 activations are the error, and they are not the
   default.** `ExactnessCheck` computes a prompt's next-token
   log-probabilities with the reference backend (double sums, x never
   rounded) and compares the GPU's paths. The worst gap over the top 10:

   | Path | Worst gap |
   |---|---|
   | x quantized to int8 per 32 values (llama.cpp's MMVQ and MMQ) | 0.63 |
   | the F16 GEMM | 0.034 |
   | float x, weights decoded in registers | 0.005 |

   `HipOps` takes `MatVecInputs`: `Float` is the default, and `Int8` is
   kept. With one vector the two run at the same decode speed: 36.6 tok/s
   either way. ROCmFP4 in rows of whole 256s takes int8 x under both (step 9,
   "ROCmFP4 over int8 x").

   Speed on Qwen3-4B Q8_0, against llama.cpp b11160 on the same GGUF:
   - **Decode:** 36.6 tok/s against 42.7 (86%). The step is GPU-bound
     (26 ms, of which host submission is 2.7 ms). `rocprofv3` gives the
     matrix-vector products 87%, attention 4% and RMS norm 3.6%. On the
     model's small matrices the products reach 180–218 GB/s.
   - **Prefill:** from 32 tokens a GEMM (1896 tok/s for llama.cpp at 512).
     Below that, passes of 8 rows, because dequantizing the weights to F16
     is a fixed 40 ms: a 19-token prompt runs at 129 tok/s, JIT warm-up
     included.

   Left for later, as the benchmarks point:
   - matrix-vector efficiency on small matrices (the decode gap);
   - float x with 4–8 vectors is slow (MTP verification). F16 x with
     `v_dot2` would round x to 2⁻¹¹ at int8's speed;
   - a fused dequantizing GEMM for prefill (no F16 copy of the weights);
   - HIP-graph replay of a decode step, which pays only once the kernels
     leave host submission on the critical path;
   - RMS norm per head with 256 threads on rows of 128 is wasteful.
   - Runner-agnostic parameters in drift (François, 2026-09-25): higher-level
     parameters and checkpoints with a value per runner, such as "flash
     attention", which is sd-cpp's `--diffusion-fa` and llama.cpp's
     `-fa on`. A runner may still take arguments of its own that others lack.
     Later, when a second runner makes it pay.
7. ~~**Server and drift.**~~ Done: drift can launch a chat configuration on
   the runner.
   - `decode/ChatEngine` is the library core: the model (`models/Models`
     dispatches on the GGUF architecture), its own tokenizer and template,
     one sequence, and replies streamed to a `ChatListener`.
     `ReasoningSplitter` cuts `<think>` blocks into reasoning, whatever the
     pieces' boundaries. `StopStrings` holds text back while it could start
     a stop string. drift could call the engine directly; it is a separate
     process for crash isolation, memory that returns when it stops, and
     drift's existing launch path.
   - `server/` is a thin shell over the engine: the JDK's HTTP server with
     `/health` (503 while loading), `/props`, `/v1/models`,
     `/lora-adapters`, `/tokenize` and `/v1/chat/completions`. Streamed
     replies carry `content` and `reasoning_content` deltas, the finish
     reason, `usage` and llama.cpp's `timings`. A dropped stream (drift's
     Stop) cancels the reply.
   - `ServerOptions` takes llama-server's flags. Flags that change speed but
     not the answer (`-ngl`, `-fa`, the speculative flags until step 9) are
     accepted with a note. Flags it cannot honour (`--lora`,
     `--chat-template`) are refused by name, and so is any unknown flag.
   - **Shipped inside drift** (2026-09-25; it was `./mill
     runner.installRuntime` before). The runner's jar, kernels included, and
     the ROCm version its kernels were built with are the backend resources
     `runner/drift-runner.jar` and `runner/rocm-version`. The universal stage
     carries them, and mill is not needed where drift runs. The runtimes page
     installs it (spec 43, `RunnerFiles`), and each start refreshes an
     installed one. The launcher runs drift's own Java: the backend runs on
     JDK 25 (`temurin:25`) too, so a distribution carries one JDK. The jar has
     no prepended launcher script (`prependShellScript = ""`), since drift
     reads it as a zip. A GGUF loads in 0.4 s: its weights are mapped and
     registered, never copied.
   - Tests: `ReplyParsingTests` and `ServerTests` (CPU), and `ServerSmoke`
     (a real model driven as drift's assistant proxy drives llama-server).

8. ~~**Qwen 3.6 35B-A3B.**~~ Done: `models/Qwen35Moe` (GGUF `qwen35moe`,
   transformers `qwen3_5_moe`) chats end to end. On the same GGUF it gives
   llama.cpp's most likely token at 24 of 24 steps. Against the exact
   reference (`ExactnessCheck`) its worst top-10 gap is 0.0017, where the
   int8-input path, llama.cpp's, is at 0.29. Its gaps against llama.cpp
   are therefore llama.cpp's own error.
   - **Layers.** Three gated-DeltaNet layers out of four:
     - a causal convolution with a carried state;
     - the delta rule, one workgroup per value head with the F32 state in
       registers;
     - a fused gated RMS norm.
     The fourth layer is full attention with a sigmoid output gate
     (`splitHalves` takes each head's query and gate) and partial RoPE (64
     of 256). Every layer has an MoE: GPU top-k routing (`route_topk`),
     SwiGLU experts in one fused kernel (`expertsGatedLinear`, the shared
     expert as expert 0 of one), and `moeCombine`. The combine reduces the
     shared gate logit itself and adds the residual.
   - **Weights.** Both sources are read as stored:
     - llama.cpp's GGUF: norms with Qwen's `+1` folded in, value heads
       tiled (key head `h % keyHeads`), `ssm_a = −exp(A_log)`.
     - transformers' checkpoints: `(1 + w)` norms, grouped value heads,
       `A_log`, one tensor per expert (stacked at load).
   - **`state/Sequence`** holds a sequence's KV pages and caches and its
     recurrent F32 states (convolution inputs and delta-rule matrices),
     reset to zero. `CausalModel.newSequence` creates them.
   - **Tests.** The golden tiny Qwen 3.5 MoE matches transformers to
     0.0011% (`Cpu`) and 0.0012% (`Hip`). transformers computes it with the
     *chunked* delta rule and the runner with the recurrent one, and it also
     holds prefill then decode. The server smoke test passes on the real
     model, with reasoning separated.
   - **Speed.** Decode runs at 52 tok/s (19.4 ms per step). llama.cpp b11160
     on the same GGUF: 47 tok/s (ROCm), 69 tok/s (Vulkan), both without
     MTP. About 2 GB of weights per token make a ceiling near 110 tok/s.
     `rocprofv3` shows the large products at full bandwidth, and many small
     ones (routers, the shared expert, 420-byte expert rows,
     `[2048, 4096]` projections) bound by a per-kernel floor. The step has
     775 launches, busy 84% of the time. Done so far: narrow-trip kernels
     for short rows, split rows for matrices of up to 1024 rows, the fused
     SwiGLU experts, a parallel combine, and the gated norm.
     Since overtaken: see **Decode speed** under step 9.
   - **Prefill** (2026-09-25). From 64 slots on,
     `expertsLinear` and `expertsGatedLinear` group the slots by expert
     (`group_slots`, then `matvec_experts_grouped` and
     `matvec_experts_gated_grouped`). A wave reads an expert's row once for up
     to 8 of its slots, where it read it once per slot. Per slot, a prompt read
     about a decode step's expert weights for every token: on Flash Next the
     MoE is 16 ms of a 43 ms step, so about 10 of the 16 s of a 630-token
     prompt. The shared expert goes the same way, as one expert.
     `PrefillProfile` times a prompt; `LinearTests` checks the grouped path
     against `Cpu`. A 512-token prompt, warm, on the laptop:

     | | per slot | grouped | llama.cpp Vulkan |
     |---|---|---|---|
     | Qwen 3.6 35B-A3B Q4_K | 1.34 s | 1.03 s (498 tok/s) | |
     | Qwen 3.8 Flash Next IQ4_XS | 4.73 s | 2.80 s (183 tok/s) | 519 tok/s |

     The grouped kernels are now about 2.5 of Flash Next's 2.8 s (rocprof).
     They multiply in F32 on the vector units, not the matrix cores.
   - **ROCmFP4's prompt experts on the matrix units** (2026-09-26,
     `experts_matrix`): an expert's batch of up to 16 slots against 16 of
     its rows per wave, as `v_wmma_f32_16x16x16_f16` tiles. x rounds to F16
     as the dense GEMMs round it; the weights decode to F16 exactly (a code
     of at most 10 times a 4-bit scale); the sums stay F32. `FusedOpsTests`
     holds it to 2⁻¹¹ of Σ|w·x| (worst at 6% of that). A prompt's numbers
     move, so the greedy text after a prompt of 8 tokens or more differs from
     before; against the exact reference (`ExactnessCheck`) the ROCmFP4
     model's worst top-10 gap stays that of its int8 dense products (0.37 /
     0.49 / 0.50 before, 0.40 / 0.35 / 0.63 after, GEMM / int8 / float
     runs). Server prompts, tok/s best / median, warm:

     | prompt tokens | 33–43 | 320–491 | 1600–1800 |
     |---|---|---|---|
     | runner ROCmFP4, before | 262 / 258 | 453 / 451 | 451 / 450 |
     | runner ROCmFP4 | 302 / 301 | 1018 / 982 | 1010 / 1007 |
     | runner Q4_K | 262 / 253 | 459 / 455 | 451 / 447 |
     | llama.cpp b11194 Vulkan, Q4_K | 276 / 275 | 929 / 887 | 1013 / 1007 |
   - **Left for later:**
     - the other types' grouped experts on the matrix units (Q4_K and the
       IQ quants still multiply in F32 on the vector units);
     - the chunked delta rule: the recurrent one costs 94 ms of Flash Next's
       512-token prompt (rocprof), so it waits.

9. **MTP, rollback, prefix cache.** Done for Qwen 3.5/3.6 MoE:
   - **The MTP layer** (GGUF `blk.<layers>` with `nextn.eh_proj`, `enorm`,
     `hnorm` and `shared_head_norm`; transformers' `mtp.*` with fused
     experts) runs as vLLM's `qwen3_next_mtp` does. Entry `i` joins token
     `i + 1`'s normed embedding and the target's final-norm hidden state at
     `i` (`joinHalves`, then `eh_proj`). Then comes one full-attention
     decoder layer with its own KV cache (the sequence's last), and the head
     norm and the shared output head. The tiny fixture adds random MTP weights
     and a reference built from transformers' own decoder layer. The runner
     matches it to 0.003% (`Cpu`) and 0.005% (`Hip`).
   - **The loop** (`decode/Generator`, `decode/Drafter`):
     - `MultiTokenDrafter` catches the MTP cache up on the entries that were
       waiting for their token, together with the first greedy draft, then
       chains further drafts on its own output state.
     - `CausalModel.verify` runs `[token, drafts…]` with every logit.
       `causal_conv_silu` and `delta_rule` write each token's state to a
       history. The target's distributions are sampled in order, so the
       output is exactly that of plain decoding, sampling included.
     - A rejection restores the kept state (`Sequence.restore`). The KV
       caches just move their length back.
     - Tests pin the invariant on both tiny models and both backends. One
       uses an oracle drafter that is right, partly right or wrong on
       purpose; one uses the MTP layer itself.
   - **The prefix cache.** A checkpoint is a copy of the recurrent states,
     plus the drafter's waiting entries (`Snapshot`). One is taken at each
     prompt's end and before each token that opens a turn (`<|im_start|>`),
     the latest 8 kept (about 115 MB each on Flash Next). A new prompt resumes
     from all the sequence holds when it starts with the last prompt and its
     reply. Otherwise it resumes from the latest checkpoint within what it
     shares with the sequence (`Prompt.sharedPrefix`: the same tokens and
     pictures, no image cut). So a conversation's next turn runs only its new
     tokens, even when the template rewrites the last reply: Qwen 3.6 with
     thinking off drops the reply's empty reasoning, and the next turn
     resumes from that reply's start, its images not read again. Live on
     Qwen 3.6 (`ServerSmoke`'s follow-up): the second turn reuses 18 of its
     41 prompt tokens, where it reused none. A model without recurrent states or
     drafter resumes from anywhere. `timings` reports `cache_n`, `draft_n`
     and `draft_n_accepted`.
   - **Server.** `--spec-type draft-mtp` drafts `--spec-draft-n-max` tokens
     per step (2 when not given). Other drafting kinds are accepted, and
     nothing drafts. `--spec-draft-model` (llama.cpp's `-md`,
     `--model-draft`) names a GGUF whose MTP layer drafts instead of the
     model's own: a whole model's file, of which only the `nextn` layer is
     copied to the device (`WeightSource.copied`), or an MTP-only file. The
     embedding and output head stay the model's. So the ROCmFP4 Qwen 3.6,
     which has no MTP layer, drafts with the Huihui Q4_K file's. The
     draft-side flags (`--spec-draft-ngl`, `-devd`, `--spec-draft-n-min`,
     `--spec-draft-p-min`) are accepted and ignored.
   - **Verification equals decoding, bit for bit.** A token's logits in an
     `m`-token verification are exactly its decode step's, so drafting never
     changes greedy output (`TinyModelGpuTests`, and the live runs below).
     What makes it so: the float-x products pick their kernel by the matrix,
     never by `m` (ROCmFP4's int8 products run one routine for any `m`), and the direct kernels spell out their fused products
     (contraction differed where they were inlined); routers and split rows
     take every vector in the same kernel; RMS norms use 1024 threads per long
     row at any row count; and attention gives each token its own tile, its
     keys past the token zeroed (the matrix units' sums move by an ulp when a
     masked key's value is there), split by the token's own key count (up to
     32 runs of at least 2 blocks, the rest empty). A lone split is divided
     as the combine divides. Before this the Q4_K text matched by luck.
   - **Speed on Qwen 3.6 35B-A3B Q4_K** (200 greedy tokens of Python code):
     - plain decoding: 50 tok/s;
     - 2 drafts: 70–76 tok/s, 95% accepted;
     - 3 drafts: 72 tok/s, 91% accepted;
     - llama.cpp Vulkan with MTP: about 70 tok/s.
     
     The text is identical with and without drafts. A step (a 3-token verify
     and 2 draft passes) costs about twice a plain one: the verify routes
     more distinct experts.
   - **Decode speed.** Qwen 3.6 35B-A3B Q4_K now decodes at 60.6 tok/s
     plain (16.6 ms a step, from 19.3) and 80.5 tok/s with 2 MTP drafts
     (99 of 106 accepted). llama.cpp Vulkan on the same GGUF does 69 and
     about 70. The text is unchanged. What moved it (`DecodeProfile`, then
     `rocprofv3` from the TheRock tree):
     - **Q4_K read straight into registers** (`q4_k_direct` in
       `matvec.hip`). Q4_K's 144-byte blocks keep 16-byte alignment, so each
       lane loads its block's header and its 32 bytes of nibbles with no LDS
       staging. A wave takes four rows at once, their loads in flight
       together and x loaded once. The staged kernels kept a wave alive
       about 8 µs to move one 1–2 KB row, which capped them near 140 GB/s.
       The direct ones reach 165–206 GB/s. They serve single-vector
       products of more than 1024 rows, several matrices at once, and both
       expert kernels. The down products' short rows (two blocks) go four
       to a pass.
     - **One launch for a layer's input projections** (`linears`: products
       of one type over the same x share a launch, four matrices at most).
     - **The router**, an F32 `[256, 2048]`: a workgroup per row with
       float4 loads, 25.6 → 16.7 µs.
     - **The delta rule**: a head's columns over four workgroups and each
       column over four waves, so a thread holds 32 state values, not 128
       (they spilled). 39 → 27.6 µs.
     - **Reductions** by wave shuffles, and 1024 threads for the residual
       stream's RMS norm: 8.5 → 5.6 µs.
     - **Greedy drafts argmaxed on the GPU** (`Ops.argmax`): one int comes
       back, not the 1 MB of logits.

     Tried and dropped: small weights copied to device memory (no change;
     the registered mapping streams at device speed), 2 KB trips for short
     rows (no change), int8 x (a few percent on the dense products).
     A gap between kernels costs about 2 µs unprofiled, 1.3 ms over a
     step's 685 launches.
   - **Sampled decoding** (2026-09-26). A request without sampling fields
     takes llama.cpp's defaults (temperature 0.8, top-k 40, top-p 0.95,
     min-p 0.05). Every row's logits came back and were sorted whole on the
     CPU: 14 tok/s. Now they are drawn from GPU candidates (step 7). Qwen 3.6
     35B-A3B ROCmFP4, 500 tokens, best of 5, thinking off / on:

     | | default sampling | greedy |
     |---|---|---|
     | plain, before | 13.3 / 13.5 | 67.4 / 67.2 |
     | plain, after | 65.3 / 65.0 | 67.0 / 66.6 |
     | 2 drafts, before | 14.1 / 14.3 (58 / 64% accepted) | 92.0 / 89.4 (64 / 66%) |
     | 2 drafts, after | 82.5–86.9 / 85.2–88.4 (60 / 62%) | 92.7 / 89.7 (64 / 66%) |
     | llama.cpp b11194 Vulkan, Huihui Q4_K, 2 drafts | 75.5 / 75.3 (58 / 61%) | 75.8 / 77.1 (63 / 65%) |

     The greedy text is unchanged. The runner's log prints each request's
     timings as llama-server's does (prompt and eval time, tokens per
     second, draft acceptance).
   - **ROCmFP4 decode** (2026-09-26). Qwen 3.6 35B-A3B ROCmFP4
     (Lord-H4D3ZS, fast layout, no MTP layer) drafts with the Huihui Q4_K
     file's MTP layer. What moved it:
     - **ROCmFP4 read straight into registers** (`RocmFp4Direct`): eight
       blocks as one of 256, lane j loading blocks 2j and 2j + 1 as nine
       words from the word they start in (a dwordx4 needs only word
       alignment; the fast layout's lanes start 0 or 2 bytes in), codes
       found by `v_alignbyte` and `v_perm`, plus 10 so four convert straight
       to float, and 10 Σx taken back per half block. Decoded once per row,
       they meet every vector of a verification. Rows of whole 256s only;
       both layouts. Per step (rocprof): the products 12.1 → 9.3 ms, the
       SSM projections 96 → 69 µs (13.4 MB, 195 GB/s), the output head
       1.2 ms (225 GB/s). Plain step 18.5 → 13.7 ms, 3-token verification
       29.8 → 22.0 ms.
     - **Attention per token** (above), more splits for one token: 66 →
       about 36 µs a layer at 160 keys (with the combine), 37 µs at 2200,
       90 µs at 8000.
     - **Routers over every vector in one launch** (verification: 39 → 20 µs).
     - **ROCmFP4 over int8 x** (for Qwen 3.8 27B, dense). With float x a
       verification's every vector paid the conversions and products (about
       three instructions per weight), so a 5-token verification took 2.2
       plain steps. Now ROCmFP4 in rows of whole 256s meets x quantized per
       32 values (`quantize_x`, once per `linears`), as the fork's MMVQ
       does: four codes per `v_dot4`, the blocks loaded as `RocmFp4Direct`
       loads them, two rows a wave, one routine (`rocmfp4_int8_rows`) for
       any number of vectors, so verification still equals decoding bit for
       bit. The tests hold it to the int8 bound. On the 27B (rocprof, µs a
       launch, one vector / five): gate and up 441 / 1035 → 444 / 575, the
       dual QKV 141 / 325 → 141 / 182, the output head 2955 / 7009 →
       2975 / 3262. Qwen 3.6 35B-A3B ROCmFP4: 2 drafts 94.8 → 101.0 tok/s,
       plain 69.1 → 67.3 (the quantization's launches).
     - Tried and dropped for the float-x verification: codes converted once
       for every vector (slower: the x loads waited in turn), and x staged
       in LDS per workgroup (slower: the barriers left too few waves to hide
       the weights' latency).

   - **Fewer launches** (2026-09-26). A plain step had 794 launches and
     spent 2.6 ms outside the products (rocprof: 12.2 of 14.1 ms busy, the
     products 9.7 ms at 155–220 GB/s); it has 303, and 1.4 ms outside them.
     A token reads 1.64 GB of weights and 0.13 GB of delta-rule state: 7.4 ms
     at the 240 GB/s the GPU streams, the floor. Every fused kernel gives the
     launches it replaces bit for bit (`FusedOpsTests`), so drafting still
     changes no token, and the greedy text is the same apart from the prompt
     (above). What changed:
     - **x quantized where it is written** (`quantize.h`): the RMS norms, the
       gated norm and the attention's output gate write the int8 form beside
       the float one; `linears` takes it when no launch or copy came between
       (`HipOps`' `QuantizedValues`). The separate `quantize_x` launches are
       gone.
     - **The residual stream's add and sum in the next norm**: `addRmsNorm`
       after the attention, and the layers now end with the next one's input
       norm (`moeCombineNorm`, `RowNorm`).
     - **A decode step's mixture of experts in one launch** (`moe_rocmfp4`,
       `mixtureNorm`): the shared expert's gate (its sigmoid), gate, up and
       down (int8 x quantized in LDS), the routed experts' gate, up and down,
       then the sum and norm, phase after phase in the grid. A wave waits
       only for what it reads (a slot's down for that slot's gate and up);
       workgroups take their place by a ticket, so the ones they wait on have
       started. The sum's 32 waves reduce as `moe_combine_norm`'s 1024
       threads do. Single tokens only: a verification's run faster as
       separate launches (the shared expert still inside the routed gate and
       up's).
     - **The router's top-k in the router's launch** (the last workgroup
       routes, `matvec_rows_f32_route`, four rows a workgroup).
     - **The convolution inside the delta rule** (`conv_delta_rule`): each
       reader convolves its channels; a key head's keeper writes their state
       once its other readers have read it. **The attention's split, q/k
       norms, RoPE and cache write in one** (`attention_inputs`).
     - **Int8 products two rows a wave**, rows of fewer than eight blocks of
       256 several to a wave (`matvec_int8_group`).
     - **Sampling candidates** over 4096-value chunks, the last chunk of a row
       to finish selecting among theirs (was one workgroup per row, 0.4 ms).

     Server, prose prompt, 500 tokens, thinking off, tok/s best / median of
     5 after a warm-up, before and after interleaved; default sampling is
     llama.cpp's (acc. = drafts accepted):

     | | default sampling | greedy |
     |---|---|---|
     | plain, before | 65.8 / 65.5 | 67.3 / 67.0 |
     | plain | 78.7 / 78.4 | 79.7 / 79.1 |
     | 2 drafts, before | 84.9 / 78.9 (0.57) | 92.7 / 85.3 (0.64) |
     | 2 drafts | 91.4 / 87.4 (0.59) | 97.3 / 91.0 (0.64) |
     | code prompt, 2 drafts, before | 102.3 / 97.9 (0.86) | 104.8 / 101.7 (0.86) |
     | code prompt, 2 drafts | 113.5 / 112.3 (0.87) | 111.9 / 109.0 (0.85) |
     | llama.cpp b11194 Vulkan, Q4_K, 2 drafts | 72.4 / 69.8 (0.57) | 77.0 / 76.4 (0.64) |

     A plain step is 12.5 ms, 1.7 times the floor: the products run at
     155–220 GB/s (the smaller the matrix, the slower), the MoE launch at
     about 150.
     Tried and dropped: the router inside the MoE launch (its phase took the
     products' low occupancy), a verification's experts read once per
     expert (few of a real verification's slots share one; slower), top-k
     by ranks over a workgroup (slower than a wave's rounds), the next pass
     prefetched in the int8 products (more registers, slower), non-temporal
     weight loads (the router and delta rule slowed), weights copied to
     device memory (4% faster, twice the memory).

     500 greedy tokens of Python code (thinking off), through the server's
     own `timings`, best / median of 5, the same prompt and the laptop as it
     is (power-limited, balanced profile). acc. is the share of drafts
     accepted:

     | | plain | 2 drafts | 3 drafts | 4 drafts |
     |---|---|---|---|---|
     | llama.cpp b11189 Vulkan, Q4_K | 68.1 / 66.9 | 92.2 / 88.8 (0.90) | 91.7 / 89.9 (0.86) | |
     | runner before, Q4_K | 54.4 / 54.2 | 74.8 / 74.7 (0.88) | | |
     | runner before, ROCmFP4 | 57.3 / 53.6 | — | | |
     | runner, Q4_K | 57.2 / 56.9 | 77.9 / 77.4 (0.90) | 77.5 / 76.8 (0.85) | |
     | runner, ROCmFP4 | 68.5 / 68.2 | 95.9 / 95.8 (0.86) | 93.0 / 92.4 (0.75) | 90.3 / 89.9 (0.69) |

     2000 tokens with thinking on: the runner's ROCmFP4 91.7 tok/s with 2
     drafts (0.85 accepted), 65.7 plain; llama.cpp's Q4_K 80.4 with 2 drafts
     (0.77). The runner's text is the same with and without drafts;
     llama.cpp's varies run to run. Two drafts is the knee here. The
     recorded 60.6 / 80.5 for Q4_K did not reproduce: commit 7abe30b itself
     measures 51.9 / 71.8 on this prompt today, below the current code, so
     no regression, a slower laptop state (llama.cpp too, 66.9 against 69).
     Tried and dropped: four rows per wave for verification (registers
     spill), eight for single vectors (slower), 64 key splits (slower past
     2000 keys).
   - **Left:**
     - several sequences at once (one sequence per engine so far);
     - the verify pass's dense products in the other types cost about twice
       a decode step's (float x: every vector pays the conversions and
       products); ROCmFP4's no longer do;
     - the products' bandwidth on matrices of a few MB (155–190 GB/s against
       220 for the output head), the one lever left of size for ROCmFP4;
     - Q6_K read directly (the output head is at 190 GB/s already, the
       other Q6_K products near 145); its 210-byte blocks are only 2-byte
       aligned;
     - the other types' products over x quantized as it is written, and their
       experts in one launch (ROCmFP4 only so far).
10. **Qwen 3.8 Flash Next.** Groomed 2026-09-25 from transformers'
    `qwen4_exp` (the reference) and llama.cpp's `qwen4exp` (b10660 and
    later, for the GGUF's conventions). The file on disk is mradermacher's
    abliterated IQ4_XS GGUF (98 GB) with EasiiX's MTP GGUF (Q8_0, 4.1 GB).
    How the model differs from Qwen 3.6:
    - **Four residual streams** (`hc_*`, 10240 wide). Every block reads
      them through a hyper-connection:
      - `Xn` = a per-stream RMS norm with `(1 + w)` (folded in the GGUF);
      - `u = mean_j(sigmoid(W_up · silu(W_down · Xn / 4))_j ⊙ Xn_j)`;
      - `w = 2 · sigmoid(W_inject · Xn / 4)`.

      Each stream then adds `w_j · block(u)`. These norms are the only ones
      outside the blocks: no `input_layernorm` or post-attention norm, and
      the head reads the streams through the `output_hc_*` mixer, with no
      final norm.
    - **The n-gram embedding (PLE)**, before layer 1 (0-based; HF's
      `ple_layer_ids` is 1-based):
      - Host-side hashing over the token and its two predecessors, with
        context cut at EOS: `(t0·m0) ^ (t1·m1) [^ (t2·m2)] mod P_h + off_h`.
        The multipliers, primes and offsets come from the GGUF.
      - 16 rows of 160 are gathered from the 320M-row IQ4_NL table; the table
        stays in the mapped file.
      - A per-stream key/query sigmoid gate on `W_value · E`, then a
        depthwise causal conv, kernel 4, dilation 3, over the normed gate
        output. It keeps 9 steps of state, and the last two token ids.
    - **The DeltaNet's output gate is a sigmoid**, not a SiLU. There are 48
      value heads over 16 key heads (tiled in the GGUF).
    - **Qwen Sparse Attention** on every fourth layer, as Qwen 3.6's gated
      attention (24 heads over 2, heads of 256, partial RoPE of 64). An
      indexer caches raw 128-wide keys. Each query scores blocks of 4 (keys
      mean-pooled, normed, roped at the block's start), keeps the best 512
      blocks plus the incomplete last one, and attends to those tokens.
      Up to 2051 visible tokens that is every token: exactly dense
      attention. The IQ4_XS GGUF says `compress_ratios` 0, and llama.cpp
      then runs dense attention everywhere. The runner follows the reference.
    - **MoE** with 512 experts, top 10 of a softmax, renormalized, experts of
      640, a sigmoid-gated shared expert.
    - **Weights** in IQ4_XS, IQ4_NL, Q5_K, Q5_1, Q6_K, BF16, F16 and F32.
    - **MTP.** transformers has no MTP code. The only reference is llama.cpp's
      open PR #28243: `eh_proj` over the next token's normed embedding and each
      stream of the target's normed streams (`hnorm`, 10240), then one dense
      attention block with the hyper-connections, the `output_hc` mixer and
      the head. The sidecar carries its own embedding and head, in Q8_0.

    Sub-steps, each ending in its tests:
    1. ~~**Quant types.**~~ Done 2026-09-25.
       - **CPU decoders:** `tensor/IQuants` (IQ4_NL, IQ4_XS), bit-exact
         against gguf-py's fixtures; Q5_K and Q5_1 were already there.
       - **GPU:** Q5_1, Q5_K, IQ4_NL and IQ4_XS as staged block types in
         `matvec.hip`, so every kernel family serves them: int8 and float
         x, experts, split rows, dequantization for the GEMM, and gathers.
         The IQ4 table is looked up four nibbles at a time with `v_perm`.
         `LinearTests` covers them all.
       - A direct-load IQ4_XS kernel (136-byte blocks, 8-byte aligned) waits
         for the benchmark in sub-step 6.
    2. ~~**The tiny golden model.**~~ Done 2026-09-25.
       `fixtures/tiny_models.py qwen4exp` builds it with transformers 5.17 (the
       first release with `qwen4_exp`; the script says how to run it with uv).
       It has random weights, a hyper-connection rank of 32, 4 n-gram heads over
       a vocabulary base of 50, the PLE before layer 1, three DeltaNet layers
       and one sparse-attention layer whose indexer keeps 2 blocks of 4. So
       positions 0–10 see every token, and later ones a selection (8% apart
       from dense). EOS sits twice in the 20 tokens.
    3. ~~**The model**~~ Done 2026-09-25: `models/Qwen4Exp`.
       - **Shared blocks.** Qwen 3.6's blocks moved to `models/HybridBlocks`
         (`HybridShape`, `HybridWeights`, the weight holders,
         `BlockBuffers`, `HybridBlocks`); `Qwen35Moe` now uses them too.
       - **New ops,** on both backends:
         - `groupRmsNorm`: a weight row per stream;
         - `streamsMix` and `streamsCombine`: the hyper-connections' read and
           write;
         - `pleGate`;
         - `ngramRows`: the hash on the GPU in 64-bit arithmetic;
         - `causalConv` with a dilation;
         - `gatedRmsNorm` with a sigmoid gate;
         - `toInts`, and an F32 gather of any width.
       - **The n-gram state.** The last two token ids are one more recurrent
         state, stored as id + 1, so rollback and checkpoints keep them.
         transformers' hash constants are computed for its checkpoints; a GGUF
         stores them.
       - **Checks:**
         - The tiny model matches transformers on its dense positions to
           0.0036% (`Cpu`) and 0.0034% (`Hip`), all at once and prefill then
           decode across an EOS.
         - On the IQ4_XS GGUF it gives llama.cpp b11160 Vulkan's most likely
           token at 24 of 24 steps, with top-10 overlaps of 9–10.
         - It answers in ChatDemo at 13.9 tok/s, untuned.
       - **The dense kernels take whole blocks of 32** and now refuse other
         widths. A rank of 16 in the first fixture read as zero.
    4. **Sparse attention past 2051 tokens**: an indexer kernel that pools,
       norms, ropes and scores blocks, a top-k of 512, and attention over the
       chosen blocks. Rolled back with the rest for MTP. Checked on the tiny
       model with a long prompt.
    5. ~~**MTP**~~ Done 2026-09-25, from the sidecar
       (`--spec-draft-model`, drift's MTP checkpoint flag).
       - **The draft pass** (`Qwen4Exp.draft`), as llama.cpp's PR #28243
         computes it:
         - the target's four streams after its last layer, each RMS-normed
           with `nextn.hnorm`;
         - joined, stream by stream, with the next token's embedding normed
           with `enorm`, through `eh_proj`;
         - one dense full-attention layer with its hyper-connections and
           experts;
         - the head's own mixer (`nextn.hc_head_*`, or `output_hc_*` in heads
           exported alone, as EasiiX's is), then the model's output head.
       - **Reusable parts.** `WorkspaceSlot` moved to `HybridBlocks`.
         `Models.open` takes the draft model; Qwen 3 and 3.5 refuse one.
       - **The per-stream norm is the right reading.** gufo (see Notes) norms
         the whole 4 × 2560 row at once instead. On the real model the
         per-stream norm accepts more drafts: 62% against 53% on prose, 88%
         against 83% on code.
       - **Tests:**
         - The tiny fixture's head (`mtp.safetensors`, GGUF names, norms
           folded) comes from transformers' own decoder layer and mixer. The
           runner matches it to 0.0031% (`Cpu`) and 0.0228% (`Hip`).
         - The invariant tests (oracle drafts, MTP drafts, the prefix cache)
           cover Qwen 3.8 too, with its n-gram states.
       - **Speed, measured on François's laptop** (its power limit is below
         a desktop Strix Halo's, so published figures don't transfer; warm
         page cache, 200 greedy tokens):

         | Qwen 3.8 Flash Next IQ4_XS | decode tok/s |
         |---|---|
         | llama.cpp b11160 Vulkan (`llama-bench` tg128) | 31.2 |
         | llama.cpp b11160 ROCm (`llama-bench` tg128) | 22.3 |
         | runner, plain (43.5 ms a step) | 21.4–22.3 |
         | runner, 2 MTP drafts | 27.6 (code, 88% accepted), 22.1 (prose, 62%) |

         After a first speed round (below): plain 24.8–25.0 tok/s; 2 MTP drafts
         32.1 (code) and 25.7–26.6 (prose).

         llama.cpp b11160 has no MTP for this model (the PR is open). Its
         prefill of 512 tokens runs at 519 (Vulkan) and 389 (ROCm) tok/s.
         Cold, the first run after another tool faulted the 98 GB file back
         in, and a step took 83 ms. rocprof puts the output head at 10 ms,
         but moving it to device memory changes nothing, so the profiler
         inflates it here; per-kernel costs need another measurement.
    6. **Server and benchmark** against llama.cpp on the same GGUF, and
       drift's launch. A first speed round is done (2026-09-25). Each change
       was A/B-measured warm on the laptop; rocprof misled here (it put the
       output head at 10 ms; skipping it saves 3).
       - **Where a plain step went** (43.5 ms, by skipping each part): the
         MoE blocks 16.4 ms, the DeltaNet blocks 14.3, the output head 3.0,
         the rest (attention, hyper-connections, n-grams) about 10.
       - **Kept:**
         - Direct-load kernels for Q5_K and IQ4_XS: the direct core is now a
           template over the block type (`Q4KDirect`, `Q5KDirect`,
           `IQ4XSDirect`). 43.3 → 40.5 ms.
         - Plain loads instead of non-temporal ones (gufo's finding): within
           noise, slightly better. Kept for simplicity.
         - Direct kernels for several vectors, and several matrices over
           several vectors (a verification): a 3-token verify 81.7 → 71.9 ms.
         - The hyper-connections' scales and sigmoids folded into the mix and
           combine kernels: about 580 fewer launches, no measurable gain
           (launch gaps are not the limit here).
       - **Tried and dropped:**
         - Experts grouped by expert during verification. 68–73% of a real
           verify's slots are distinct experts, but the grouped kernel (every
           token's sums in registers) was no faster on real text, and slower
           on prose.
         - The output head copied to device memory: no change.
       - **Left, by measured cost:**
         - The down experts (IQ4_NL, Q5_1, rows of 360 bytes) run at about
           116 GB/s, 3.8 ms a step.
         - The DeltaNet blocks run at half their weights' speed.
         - The hyper-connections' small products (IQ4_NL rows of 180 bytes,
           320-row IQ4_XS) are latency-bound.
         - llama.cpp Vulkan is still at 31.2 plain against 25.

    ROCmFP4 is left for step 15.
11. **Vision.** The ViT with video, chat LoRAs, grammars. Gemma 4 is
    dropped (François, 2026-09-25: its results didn't convince him).
    - **Done 2026-09-25: images, for Qwen 3.6, Qwen 3.8 Flash Next and Qwen
      Image 2.1's editing.** All three use Qwen3-VL's ViT (27 blocks of 1152,
      16 heads of 72, patches of 16, 2 × 2 merge; Qwen 3.5/3.6/3.8 differ only
      by the merger's output width, Qwen3-VL adds deepstack).
      - **The tower** (`models/QwenVision`) reads llama.cpp's mmproj
        (`qwen3vl_merger`) or transformers' `model.visual.`. The Conv3d's two
        frames are summed into one linear layer (an image is its own second
        frame). Heads of 72 are padded with zeros to the attention kernels'
        80 at load (`Attention.paddedHeadWidth`), which changes neither scores
        nor values. The learned position table is sampled bilinearly on the
        host (`fast_pos_embed_interpolate`), the rotary positions are 2-D
        (row, column; NeoX, 18 + 18 pairs), attention is bidirectional over the
        image. The merger joins each window's 4 patches (layer norm, GELU
        MLP); a deepstack tap does the same after its block, norm after the
        join.
      - **Pixels** (`vision/`): Qwen's `smart_resize` to multiples of 32
        between 64 and 4096 tokens' pixels (transformers' cap of 16 384 costs
        minutes on a photo), PIL's antialiased bicubic, alpha over white,
        `x / 127.5 − 1`, patches in merge-window order. Only `data:` URLs;
        ImageIO reads PNG, JPEG, GIF and BMP.
      - **The language model.** The template's `<|image_pad|>` becomes the
        image's tokens (`Prompt.withImages`), whose embeddings the tower gives
        (`GivenRows`). `Sequence` holds each slot's mRoPE position: text counts
        on by one on all three axes, an image's tokens sit on its grid after
        the text before it, the text after resumes past its longer side
        (`get_rope_index`), so a reply's tokens turn at their slot plus the
        prompt's shift. The hybrids now read `rope.dimension_sections` (or
        `mrope_section`) and turn by interleaved mRoPE; text alone is
        unchanged. MTP drafting keeps working; its entries for image tokens
        embed the placeholder (drafts only cost speed).
      - **Prefix cache.** A held prefix matches only with the same pictures
        (SHA-256 of the bytes) and never cuts an image. The tower runs once
        per new image, inside the prompt's time.
      - **The server.** `--mmproj` loads the tower, `/props` says vision,
        `image_url` parts become the template's `{"type": "image"}`.
        `video_url` is refused.
      - **Tests.** Tiny transformers models: Qwen 3.5 MoE seeing
        (`qwen35moe_vision`): the processor's patches exact, a 70 × 100 photo
        resized within 1 pixel level of PIL's, the tower 0.0001%, the logits
        of 22 tokens with the image, prefilled whole or cut inside it,
        0.0016% (`Cpu` and `Hip`). Qwen3-VL with deepstack (`qwen3vl_vision`):
        the last hidden state 0.0068%.
      - **Live**, a purpose-made picture (three coloured shapes and a title):
        Qwen 3.6 35B-A3B Q4_K describes every shape, colour, place and the
        text; 630 tokens with the tower in 2.9 s, MTP accepting 81%. Qwen 3.8
        Flash Next IQ4_XS with mradermacher's mmproj gets it right too; its
        prompt takes 16 s (its prefill, not the tower). A follow-up turn
        reuses 742 cached tokens with the image.
    - **Left:** video frames, chat LoRAs, grammars.

    - **Done 2026-09-25: Qwen 3.8 27B**, the dense Qwen 3.5 layout (GGUF
      `qwen35`, transformers `qwen3_5`): 64 layers of Qwen 3.6's hybrid shape
      (5120 wide, 48 value heads over 16 key heads, 24 query heads over 4 of
      256) with a dense SwiGLU of 17408, and a dense MTP layer.
      - `Qwen35Moe` became `models/Qwen35`, dense or MoE by the GGUF's
        architecture or config (`HybridShape.feedForward`: `RoutedExperts` or
        `DenseFeedForward`; `HybridBlocks.feedForward`).
      - The tiny `qwen35` fixture (transformers 5.7, its MTP head built as the
        MoE one's) matches on both backends; the MTP invariant tests cover it.
      - cygnal's heretic Q4_K_M (with MTP) against llama.cpp b11179 Vulkan: the
        same most likely token at 24 of 24 steps, top-10 overlaps of 10.
        huihui's abliterated Q4_K_L (Q8_0 tensors) answers too.
      - Speed on the laptop (Q4_K_M, 16.8 GB): plain decoding 10.7 tok/s
        (93.5 ms a step, about 180 GB/s) against llama.cpp Vulkan's 11.3;
        with 2 MTP drafts 12.9–14.1 tok/s (91% accepted), 3 drafts 14.2–14.4.
        A 3-token verify takes 145 ms, 1.55 plain steps: the lever for MTP
        here. Prefill of 512 tokens 1.71 s (299 tok/s) against llama.cpp
        Vulkan's 280.
        François recalls llama.cpp Vulkan with MTP at about the same speed on
        this model (not measured here).
      - **ROCmFP4** (2026-09-26): kingjones777's STRIX_LEAN (fast layout, the
        QKV projections dual, 14.6 GB, no MTP layer) drafts with the MTP-only
        `mtp-Qwen3.8-27B-Q4_0.gguf` of the same repository (`blk.64.nextn.*`
        and that block's tensors), the first MTP-only file run. Greedy text
        with 2, 3 or 4 drafts equals plain decoding. The built-in
        architecture `qwen3.8-27b` drafts 3 by default: two is the knee on
        prose, four on code. 500 greedy tokens, thinking off, the server's
        `timings`, best / median of 5 (acc. = share of drafts accepted);
        llama.cpp b11189 Vulkan on cygnal's Q4_K_M with its own MTP layer:

        | | plain | 2 drafts | 3 drafts | 4 drafts |
        |---|---|---|---|---|
        | code, llama.cpp Q4_K_M | 12.2 / 12.2 | 24.5 / 24.5 (0.87) | 25.5 / 25.4 (0.82) | 28.4 / 28.0 (0.75) |
        | code, runner before | 13.5 / 13.4 | 19.6 / 19.5 (0.86) | 20.3 / 20.3 (0.81) | 20.2 / 20.0 (0.77) |
        | code, runner | 13.5 / 13.5 | 28.0 / 28.0 (0.88) | 31.8 / 31.7 (0.83) | 33.8 / 33.7 (0.78) |
        | prose, llama.cpp Q4_K_M | 12.2 / 12.2 | 19.2 / 19.0 (0.56) | 17.5 / 17.0 (0.43) | 16.9 / 16.8 (0.35) |
        | prose, runner | 13.2 / 13.1 | 21.1 / 21.1 (0.54) | 20.5 / 20.5 (0.42) | 19.0 / 18.9 (0.33) |

        The model card's 24.7 (prose) and 42.7 (code) tok/s with MTP were on
        another machine. Plain decoding is a 74 ms step, about 197 GB/s; the
        products run at 209–229 GB/s. What moved drafting is ROCmFP4 over
        int8 x (step 9): a 5-token verification 160 → 95 ms, 3 tokens
        115 → 84 ms.

**Diffusion.**

12. **The first image: Krea 2**, groomed 2026-09-25.
    - **Reference.** diffusers 0.40 (`transformer_krea2.py`,
      `pipeline_krea2.py`, `AutoencoderKLQwenImage`) is the reference; sd-cpp
      (`krea2.hpp`, `wan_vae.hpp`) is the tool to beat.
    - **sd-cpp baseline on the laptop:** drift's configuration, the Lox bf16
      finetune, Wan 2.1 VAE, Qwen3-VL-4B Q8_0, 1024², 4 steps, CFG 1. It
      takes 63.6 s:
      - text encoding 1.3 s;
      - 4 steps at about 13 s each (the first also loads the weights);
      - VAE decode 5.8 s.

      The spectrum cache does nothing at 4 steps (its warm-up is 4). A step is
      about 110 TFLOP at about 8.5 TFLOPS, so the lever is GEMM efficiency,
      not bandwidth.
    - **The pipeline.**
      - **Text.** Qwen3-VL-4B's text model (GGUF `qwen3vl`: dense Qwen 3 with
        q/k norms; the mRoPE axes are equal for text, so plain RoPE). The
        prompt goes in a system/user/assistant template. The residual stream
        after layers 2, 5, …, 35 (12 taps, no final norm) is kept per token,
        without the 34 template-prefix tokens.
      - **Text fusion in the DiT.** 2 blocks attend across the 12 taps of
        each token, a 12 → 1 projector, then 2 blocks across tokens. Then
        RMS norm and an MLP from 2560 to 6144.
      - **DiT.** 12.8B: 28 single-stream blocks of 6144 wide over [text ;
        image] tokens, 48 query heads over 12 KV heads of 128, q/k RMS norm
        with `1 + w`. RoPE is 3-axis interleaved (32/48/48, θ 1000); text
        positions are 0. The attention output is gated by a sigmoid, the MLP
        is a SwiGLU of 16384. The modulation is AdaLN-single: one timestep
        vector of 6 × 6144, plus a learned table per block. The final layer
        is modulated by the timestep MLP's output.
      - **Latents.** Patch 2 over 16 channels (64 per token), 1024² → 4096
        image tokens.
      - **Sampler.** Flow-matching Euler, σ from `linspace(1, 1/N, N)`
        shifted by `e^μ / (e^μ + 1/σ − 1)`, μ = 1.15 at 1024² (Turbo, and
        the dynamic shift there alike), timestep σ × 1000. `--cfg-scale` keeps
        sd-cpp's meaning (1 = the conditional pass only).
        sd-cpp's own `discrete` schedule differs ([1, 0.863, 0.613, 0.003]
        against diffusers' [1, 0.905, 0.760, 0.513] at 4 steps). The runner
        follows diffusers, so its images are not sd-cpp's pixel for pixel.
      - **VAE.** The Wan 2.1 decoder, whose causal 3-D convolutions are
        exactly 2-D ones on a single frame (the last temporal kernel slice).
        It has RMS channel norms, one single-head attention, nearest ×2
        upsampling, and output clamped to [−1, 1]. The latents are
        un-normalized with the Wan 16-channel mean and std first.
    - **Done 2026-09-25: Krea 2 runs from drift.** The runner makes a
      1024² image in **31.1 s** against sd-cpp's 63.6 s:
      - text encoding and VAE together: about 1.4 s;
      - 4 steps of 7.2 s each;
      - loading: 1.1 s (the weights are mapped).

      It follows diffusers' schedule, so its images are not sd-cpp's pixel
      for pixel. Not there yet for Krea 2: reference images and editing,
      hires fix, VAE tiling. Requests asking for them are refused (400) by
      name.
    - **img2img, done 2026-09-26** (bug 32): the init image through the Wan
      VAE's encoder, packed 2 × 2, mixed at step `steps − ⌊steps ×
      strength⌋` as for FLUX.2. Redraws run on it.
    - **LoRAs, done 2026-09-25** (`diffusion/Lora`, `Krea2.useLoras`).
      - **Names.** The published files hold `lora_A`/`lora_B` pairs (also
        `lora_down`/`lora_up`) under `diffusion_model.` or `transformer.`, in
        the original naming or in diffusers', and are read in all of these.
      - **Scale.** `alpha / rank` when there is an `.alpha`, else 1, times the
        multiplier: sd-cpp's and ComfyUI's rule. They ignore metadata such as
        `ss_network_alpha`, and so does the runner.
      - **Linear targets** run at request time: `out += s × (x · downᵀ) ·
        upᵀ`, in scratch that grows as needed. The mapped weights stay
        untouched.
      - **Table targets** (the final modulation, the blocks' modulation
        tables, the text projector) are rebuilt from their loaded values plus
        the deltas whenever the set changes.
      - **BF16.** Updates not stored as BF16 are converted to it.
      - **Unmatched targets** are logged as warnings.
      - **The server** reads `--lora-model-dir`, lists its files in the
        capabilities, resolves a request's paths against it, and keeps opened
        files mapped.
      - **Tests.** A tiny LoRA (both namings, an alpha, two tables) matches
        diffusers with the deltas merged: 0.08% (`Cpu`) and 0.16% (`Hip`),
        the BF16 rounding of the updates. Setting it and removing it again
        gives the base velocity back.
      - **Live.** All of François's Krea 2 LoRAs match fully. The Kimono Robe
        one adds the robe. The rank-256 8-step Turbo distill gives the same
        look as sd-cpp's at 8 steps: 72 s against 129 s.
      - **Cost.** 31 s without a LoRA; 36 s with a rank-32 one and 37 s with
        a rank-256 one, both including the file's first opening.
    - **Sub-steps**, each ending in its tests:
      1. ~~**Text encoder**~~ done: `Qwen3` reads `qwen3vl` GGUFs, and
         `Qwen3.encode(ids, taps, out)` gives the residual stream after the
         chosen layers (tap-major, the layers past the last tap not run). It
         matches transformers' hidden states on the tiny Qwen 3 to 0.017%
         (the F16 KV cache).
      2. ~~**The DiT**~~ done: `models/Krea2`.
         - **Loading.** Reads the checkpoints' original names (bare or under
           ComfyUI's `model.diffusion_model.`, see *Weight names*) and
           derives its shape from the weights.
         - **GEMMs.** BF16 weights go through hipBLAS BF16 GEMMs (about 30
           TFLOPS at these shapes; 19 for the MLP's down projection).
         - **New ops:** `modulate` and `gatedAdd` (AdaLN), `shortAttention`
           (the text fusion: 12 taps per token, then the prompt's tokens), and
           RoPE `Axes` (each axis its own frequencies).
         - **Main attention.** The paged flash attention, non-causal, over a
           single page.
         - **Check.** It matches a tiny diffusers `Krea2Transformer2DModel`
           to 0.0021% (`Cpu`) and 0.0041% (`Hip`).
      3. ~~**Sampler loop and noise**~~ done: `diffusion/FlowSchedule` (equal
         to diffusers' sigmas) and `diffusion/Krea2Pipeline`. The noise comes
         from a seeded `SplittableRandom` through Box–Muller, so seeds don't
         reproduce sd-cpp's images. The pipeline checks that the template
         prefix is 34 tokens.
      4. ~~**The VAE decoder**~~ done: `models/WanVae`, channels-last.
         - **Convolutions.** 3×3 convolutions are im2col in 256 MB chunks
           plus BF16 GEMMs; every weight is uploaded as BF16.
         - **Attention.** The middle attention runs as GEMMs over chunks of
           queries.
         - **Check.** It matches a tiny diffusers `AutoencoderKLQwenImage`
           to 0.31% (`Cpu`) and 0.28% (`Hip`), the BF16 weights' rounding.
         - Tiling is still to do.
      5. ~~**The sd-server shell**~~ done: `server/ImageMain`, `ImageServer`
         and `ImageOptions`. drift also installs `drift-runner-images`
         (tool SdCpp, launcher `sd-server`). The
         capabilities come from the launch flags. The server loads, then
         listens. Jobs run one at a time and can be cancelled while queued.
         What it planned:
         - `/sdcpp/v1/capabilities`, `img_gen`, `jobs/{id}`, `cancel`;
         - sd-server's flags, as drift passes them;
         - progress lines drift parses (`| i/n - Xs/it`, `sampling using`);
         - `sd-server --help`.
      6. **End to end** on the real finetunes, compared with sd-cpp by eye,
         and benchmarked against its 63.6 s.
13. **Image breadth.** Qwen Image and 2.1, the other image families and the
    unified models; masks, step caches, hires, ESRGAN, PiD.
    - **Done 2026-09-25: FLUX.2 [klein] 9B**, txt2img, img2img and reference
      images (`models/Flux2`, `models/FluxVae`, `diffusion/Flux2Pipeline`).
      diffusers 0.40 (`Flux2KleinPipeline`) is the reference.
      - **Text.** Qwen3-8B in the chat template (thinking closed), padded to
        512 with `<|endoftext|>`. `Qwen3.encode` masks the pads as keys: the
        prompt never sees them, each pad sees the prompt alone. The residual
        stream after layers 9, 18 and 27 goes side by side (12288) through
        `txt_in`, once per prompt.
      - **DiT.** 8 double-stream blocks (text and image normed, modulated,
        projected and MLP'd apart, joint attention) and 24 single-stream
        blocks (one fused linear to q, k, v and the SwiGLU, one linear back
        from both side by side). One modulation per block kind, no biases,
        no guidance embedding (`--guidance` ignored). RoPE on 4 axes of 32,
        θ 2000: text `(0, 0, 0, l)`, target `(0, y, x, 0)`, reference `k`
        `(10k, y, x, 0)`. Sequence [text ; target ; references]; the
        references share the timestep and their outputs are dropped. It reads
        ComfyUI's names (`model.diffusion_model.`, norms `.weight`) and BFL's
        (bare, `.scale`), and splits the fused weights into row views.
      - **VAE.** The FLUX.2 VAE both ways from its LDM names, on the layers
        it shares with the Wan decoder (`models/VaeLayers`: the norm is a
        parameter, RMS or 32-group). The latents are packed 2 × 2 into 128
        features and normalized by the checkpoint's batch-norm statistics.
      - **Schedule.** FLUX.2's μ for the target's tokens and the step count
        (`FlowSchedule.flux2Shift`). `--flow-shift` is ignored for FLUX.2.
      - **img2img** follows diffusers' `get_timesteps`: it starts at step
        `steps − ⌊steps × strength⌋`, from `σ × noise + (1 − σ) × init`
        (`FlowSchedule.firstStep`, shared by the three pipelines; until bug
        32 the floor was taken after the subtraction, so 4 steps at 0.4 ran
        two, not one). sd-cpp counts `⌊steps × strength⌋` steps from the end. The server
        stretches the init image to the output's size.
      - **References.** The server stretches them to the output's size
        unless `auto_resize_ref_image` is false, as sd-server does. Then
        sd-cpp's `resize_before_vae` applies: about a megapixel (never more
        than the output), the shape kept, sides multiples of 16.
      - **New ops:** `groupNorm` (three passes: means, centred squares,
        output), `conv3x3` at stride 2 (pad bottom and right only),
        `packPatches`, `concatColumns`, and 4-axis RoPE.
      - **Tests.** Against tiny diffusers models: the transformer with a
        reference to 0.003% (`Cpu`) and 0.019% (`Hip`); the VAE's latents and
        image to 0.73%/0.36% and 1.25%/0.46%, from the BF16 weights; the
        padded encoder equal to transformers' masked hidden states.
      - **On the laptop.** PornMaster v4 Turbo bf16, Qwen3-8B UD-Q4_K_XL,
        1024², 4 steps, with the tiled attention (below):

        | | runner | sd-cpp |
        | --- | --- | --- |
        | txt2img | **22.5 s** (4.6 s/step) | 42.3 s (6.5 s/it) |
        | one 1024² reference | **46.2 s** (10.2 s/step) | 71.9 s (15.6 s/it) |
        | img2img, strength 0.6 (3 steps) | 22.4 s before the tiled attention | |

      - **Tiled attention, done 2026-09-25** (`attention_tiled_d64`/`d128`).
        - **Why.** The flash kernel gives each wave of 16 query rows its own
          pass over the cache. That suits decode, but with thousands of rows
          every wave re-reads every key and value. At 8704 tokens (a 1024²
          reference) that is 400 ms per block, 3.1 TFLOPS: 12.8 s of a
          20.4 s step.
        - **How.** A workgroup of 8 waves (128 rows) loads each block of 64
          keys and values once into LDS, rows padded by 16 halves against
          bank conflicts, and all 8 waves compute from it. The queries stay
          in registers. It runs from 256 query rows (tokens × the GQA group)
          for heads of 64 and 128; decode and heads of 256 keep the old
          kernel. The masks, windows, softcap and sinks are the same, and
          blocks that every row sees whole skip the mask.
        - **Measured** (`AttentionBenchmark`, 32 heads of 128, non-causal):
          4608 tokens 45.1 → 24.9 ms (7.7 → 14.0 TFLOPS); 8704 tokens 400.8
          → 87.7 ms (3.1 → 14.2 TFLOPS).
          - **Variants.** 4 or 16 waves and blocks of 32 keys were slower.
          - **Pipelining.** Fetching the next block during the products took
            registers and occupancy and was slower too.
          - **Registers.** Each head width is compiled for a number of waves
            per SIMD (`amdgpu_waves_per_eu`): 8 for heads of 64 and 80 (192
            registers), 6 for 128 (240). Left free, heads of 64 took 256 and
            spilled 296 bytes a lane inside the loop, and heads of 80 and 128
            fit fewer workgroups; now only a few scalars spill (12–24
            bytes). Outputs are bit-identical. Measured alone (best of 3):

            | heads | tokens | before | after |
            | --- | --- | --- | --- |
            | 24 of 64 | 16459 | 272 ms (6.1 TFLOPS) | 109 ms (15.3) |
            | 24 of 64 | 65836 | 4.66 s (5.7) | 2.13 s (12.5) |
            | 32 of 64, causal | 4096 | 11.3 ms (6.1) | 5.1 ms (13.6) |
            | 16 of 80 | 16459 | 128 ms (10.8) | 104 ms (13.4) |
            | 32 of 128 | 8704 | 128 ms (9.7) | 81 ms (15.3) |
            | 16 of 128, causal | 4096 | 5.5 ms (12.5) | 4.8 ms (14.3) |

          - **What's left.** The remaining gap to hipBLAS's ~30 TFLOPS is
            the softmax sharing the SIMDs with the WMMAs, and each operand
            being read by both half-waves.
        - **Krea 2** (48 query heads over 12) gains too: 31.1 → 26.7 s at
          1024², 4 steps.
        - **Chat prefill** of models with heads of 128 now runs it as well:
          checked against the reference in `AttentionTests`, not yet timed
          live.
      - **LoRAs, done 2026-09-25**, on the machinery Krea 2's became
        (`LoraUpdates`, `LoraFiles`). Each model now maps a file's names
        itself: `Lora.open` only strips the prefix.
        - **Namings.** FLUX.2 reads BFL's and ComfyUI's names (fused weights
          whole: `img_attn.qkv`, `img_mlp.0`, `linear1`) and diffusers' (split
          and fused: `to_q`, `ff.linear_in`, `to_qkv_mlp_proj`). A fused
          update is split by its `up` rows onto the runner's row views.
          diffusers' `norm_out.linear` has its halves swapped: (scale ;
          shift) against BFL's (shift ; scale).
        - **Unmatched targets.** A target is left unapplied and named when
          its shape differs from the weight's, e.g. a Klein 4B LoRA on 9B.
        - **Tests.** A tiny LoRA (both namings, fused and split targets, two
          alphas, the swapped final modulation) matches diffusers with the
          deltas merged: 0.099% (`Cpu`) and 0.13% (`Hip`).
        - **Live.** François's anatomy slider (rank 4, BFL names) matches all
          its targets and costs no measurable time.
      - **Not yet:** masks.
    - **Done 2026-09-26: FLUX.2 [dev]** (32B), on Klein's code: txt2img,
      img2img, reference images and LoRAs. `Flux2Pipeline` tells the two
      apart by the transformer's `guidance_in` and reads the prompt by
      `Flux2Text` (`Klein`, `Dev`).
      - **DiT.** Klein's blocks, wider and deeper (6144, 48 heads, 8 double
        and 48 single blocks, text features 15360), plus the guidance
        embedding: the distilled scale × 1000 through its own sinusoid MLP
        (`guidance_in`), added to the timestep's before the modulations'
        SiLU. Quantized GGUF weights (Q4_K, Q5_K, BF16) go through the
        dequantize-then-hipBLAS GEMMs.
      - **Text.** Mistral Small 3.x's language model (`Mistral`,
        `DenseStyle.Mistral`: no head norms, SwiGLU). Its GGUFs are
        llama.cpp's `llama`, q and k permuted, so `DenseConfig.ropeLayout`
        is interleaved there and NeoX for transformers' weights. The prompt
        after `<s>` in BFL's system prompt and `[INST]…[/INST]`; the residual
        stream after layers 10, 20 and 30 side by side. The 512 rows are the
        prompt's then zeros (sd-cpp's reading; diffusers pads with masked
        tokens). The tokenizer is Tekken (`tekken`: `mistral_common`'s
        pattern, as llama.cpp and sd-cpp; the published `tokenizer.json`
        carries an older one).
      - **Guidance.** `--guidance` and a request's `distilled_guidance`
        reach the model (3.5 by default, as sd-server); a model without a
        guidance embedding notes the flag at launch and ignores it.
      - **LoRAs.** kohya's names too (`lora_unet_double_blocks_0_img_attn_qkv`,
        `lora_down`/`lora_up`), and the guidance MLP as a target (the Turbo
        LoRA's, in BFL's or diffusers' names).
      - **Tests.** A tiny dev transformer at guidance 3.5: 0.0040% (`Cpu`),
        with its LoRA (kohya's names, the guidance MLP) 0.068%. A tiny
        Mistral from transformers' safetensors and from a llama.cpp-style
        GGUF: 0.022% both, on both backends. Tekken against HuggingFace's
        tokenizer (patched pattern) and the Mistral GGUF: 2028/2028 texts.
      - **Live** (Q4_K_M dev, Mistral Small 3.2 i1-Q4_K_M, ERNIE's
        `flux2-vae`, Turbo LoRA, guidance 4, 8 steps, 1024²):

        | | runner | sd-cpp (spectrum cache) |
        | --- | --- | --- |
        | txt2img | **172 s** (20.7 s/step, 8 steps) | 174 s (27.7 s/it, 6 of 8 steps run) |
        | img2img, strength 0.5 | **97 s** (4 steps) | 161 s (5 steps) |
        | one 1024² reference | **392 s** (48.3 s/step) | 399 s (64.6 s/it, 6 of 8) |

        Per step the runner is about a quarter faster; sd-cpp's step cache
        skips two of eight. The images match sd-cpp's closely (the same
        scene, sign text and style). The quantized GEMMs run at about 16
        TFLOPS, half of BF16's.
      - **Through drift:** the `flux-2-dev-redraw` configuration on the
        runner (Turbo and SexGod LoRAs, the latter in kohya's names, all
        targets matched) redraws a 2048² image in four 1152² tiles with the
        reference as context: 992 s, no seams, the picture kept.
    - **Done 2026-09-25: PiD 1.5, 1024 → 4096 in one pass**, its three
      variants: FLUX.2, FLUX.1 and Qwen Image (`models/Pid`,
      `diffusion/PidPipeline`, text by `Gemma2`).
      - **Reference.** NVIDIA's own code (`nv-tlabs/PiD`, commit 2c8814c) is
        the reference: `PidNet` over PixelDiT and `pid_distill_model_infer`.
        sd-cpp's `pid.hpp` was checked against it.
      - **The network.** 14 MMDiT blocks over 16×16-pixel patch tokens (1536
        wide, 24 heads of 64) with a 300-token text stream, joint attention
        with no mask. The latent's features are injected through a
        sigma-aware per-token gate before every other block. 2 pixel blocks
        follow:
        - each pixel's 16 channels are modulated per pixel from its patch
          token (`modulateChunks`);
        - each patch's 256 pixels are compressed into attention over the
          patches (16 heads of 72, padded to 80 at load with the head norms'
          weights rescaled), then expanded back;
        - a GELU MLP per pixel follows, and a final norm and linear to RGB.
        The network predicts the velocity `noise − image`. RoPE is 2-D with
        positions `linspace(0, 16)` and NTK bases against a 2048-px reference
        (`ropeTable`, cos/sin computed on the host), and 1-D over the text.
      - **The latent.** The source's latent is nearest-upsampled to the
        patch grid and run once through a 1024-wide convolution stack
        (replicate padding, GroupNorm of 4). Its heads feed 7 injections and
        the pixel blocks.
      - **Variants.** The three checkpoints differ only in the stack's input
        width, which tells the latent (as sd-cpp infers it):
        | Stack input | Variant | Latent | VAE (`--vae`) |
        |---|---|---|---|
        | 32 | FLUX.2 | 128 packed features at 1/16, BN-normalized, unpacked 2×2 | FLUX.2's (`FluxVae`) |
        | 16 | FLUX.1 | 16 channels at 1/8, `0.3611 (z − 0.1159)` | FLUX.1's or Z-Image's `ae` (`FluxVae`, no quant convolutions) |
        | 16 | Qwen Image | 16 channels at 1/8, Wan 2.1's `(z − mean) / std` | Qwen Image's (`WanVae.encode`) |
        The VAE's names pick the encoder (Wan's `encoder.conv1` or LDM's
        `conv_in`); a VAE whose latent is not the network's is refused.
        Output sizes are multiples of 4 × the latent's factor (64 or 32).
      - **Text.** Gemma 2 2B (a `DenseDecoder` style beside Qwen 3: `1 + w`
        norms around both sublayers, GeGLU, embeddings × √hidden, softcap
        50, sliding window on even layers). Its last hidden state is taken
        over NVIDIA's instruction prompt followed by the caption,
        right-padded (masked as keys), keeping BOS and the last 299 rows.
        sd-cpp drops the instruction prompt from what it encodes, and with
        it a short caption.
      - **Sampling.** NVIDIA's distilled schedule σ = 0.999, 0.866, 0.634,
        0.342, with no shift. Each step takes `x0 = x − σ v` and re-noises
        to the next σ with fresh noise. Only 4 steps are accepted.
      - **Precision.** PiD's activations reach 1e8 (the official net runs in
        BF16). Attention values beyond 8192 are scaled by a power of two
        around the F16 cache (`maxAbs`); without it the pixel blocks' values
        overflowed to NaN and the whole image came out black.
      - **New ops:** `ropeTable`, `pixelsToPatches`/`patchesToPixels`,
        `modulateChunks`, `gatedAddChunk`, `rowGatedAdd`, `maxAbs`,
        replicate padding in `conv3x3`, attention for heads of 80, and RMS
        norm on narrow rows (a thread per row: millions of 16-channel rows
        pass the one-block-per-row kernel's grid limit).
      - **Tests.**
        - Gemma 2 against transformers: 0.033% (`Cpu`), 0.035% (`Hip`).
        - A tiny official `PidNet` (`fixtures/tiny_pid.py`, patches of 8,
          BF16 weights): 0.0054% (`Cpu`) and 0.93% (`Hip`, BF16 GEMM
          inputs), at degrade σ 0 and 0.3; the 16-channel one 0.011% and
          1.2%.
        - The official tokenizers, tiny: FLUX.1's `AutoEncoder` both ways
          (latent 0.42% / 1.1%, image 0.19% / 0.37%) and Qwen Image's
          `WanVAE2d_` encoding, saved in the 3-D checkpoint's shapes (0.79% /
          0.90%).
        - The real checkpoint at 256² against the official net on the same
          inputs: worst 0.9% of the largest velocity, mean 0.45%.
      - **On the laptop**, 1024² → 4096², 4 steps: **166 s** (about 40 s a
        step, 304 s before the attention's register fix; 2048²: 17.0 s, from
        26.5), a clean, sharp decode with no seams. sd-cpp's single pass
        took 457 s and came out black; drift tiled it into 9 passes instead.
        512 → 2048 through the image server, with drift's flags: about 27 s.
      - **drift.** One pass on the runner (`specs/26`); `--tokenizer` is read
        and `--vae-format` accepted.
    - **Done 2026-09-25: Qwen Image 2.1**, txt2img with real CFG, img2img
      and LoRAs (`models/QwenImage21`, `models/QwenImage21Vae`,
      `diffusion/QwenImage21Pipeline`).
      - **Reference.** diffusers main (`QwenImage21Pipeline`, commit
        bdc2bea; 0.40 predates it) and the official repo's configs; sd-cpp's
        `qwen_image_2_1.hpp` is the tool to beat.
      - **Text.** Qwen3-VL-8B's text model over the template
        `system: Comprehend and analyze the provided prompt.` + user turn +
        open assistant turn; the residual stream after the last layer (no
        final norm), the system turn's tokens dropped. An empty prompt is
        encoded as a space.
      - **DiT.** 7B: 32 single-stream blocks of 4096 (32 heads of 128),
        layer norms without affine, a scale and a `tanh` gate per sublayer
        (no shift), q/k RMS norms, a SwiGLU of 12288, no biases. One
        modulation for all blocks, from the timestep: the image takes the
        step's, the text `t = 0`'s. The text goes through `txt_in` (a `1 + w`
        RMS norm, a tanh-GELU MLP). Latents are 64 channels at 1/16, one
        token per latent pixel.
      - **Attention.** Block-causal over [text ; image]: the text causal,
        the image seeing everything. RoPE is 3-axis interleaved (16/56/56,
        θ 10000): text `(p, p, p)`, the image's frame the text's length, rows
        and columns centred on zero.
      - **The prefix.** The text never depends on the image or the step, so
        it runs through the blocks once per prompt (`QwenImage21.prefix`),
        and each step runs the image's tokens alone against its kept keys
        and values, as diffusers' own KV cache does. Pages of 16 tokens: each
        block owns the prompt's pages, and the image's pages are shared by
        all blocks, each rewriting them before it attends.
      - **Schedule.** μ on the line through (256 tokens, 0.5) and (8192,
        0.9), exponential shift, then diffusers' `shift_terminal`: the levels
        stretched so the last is 0.02 (`FlowSchedule.stretched`).
        `--flow-shift` is ignored.
      - **VAE.** Wan 2.2's residual design on RGBA: every resampling level
        adds a shortcut of its input, channels averaged down (`averageDown`)
        in the encoder, duplicated up (`duplicateUp`) in the decoder; the
        temporal levels shuffle over two frames of which the image is the
        last. Original Wan names, nested (`upsamples.N.upsamples.M`), as
        ComfyUI's file holds them. Wan's name loaders are shared with the Wan
        2.1 decoder (`WanLayers`). Output is RGBA PNG.
      - **LoRAs.** diffusers' names and ComfyUI's; the fused `gate_up` of
        ComfyUI's checkpoint is two row views, and an update of `gate_up`
        is split by its `up` rows.
      - **Tests.** Against tiny diffusers-main models: the transformer
        (prefix, then a velocity) 0.0028% (`Cpu`) and 0.0034% (`Hip`); with
        a LoRA 0.084%/0.13%; the VAE's latents and image 0.16%/0.21% and
        0.41%/0.73% (BF16 weights); the schedule equal to diffusers'.
      - **On the laptop** (1024², the same seed and prompt on both; Qwen3-VL-8B
        Q4_K_M, bf16 VAE):

        | | runner | sd-cpp |
        | --- | --- | --- |
        | Q4_K, turbo LoRA (r64), 4 steps, CFG 1 | **27.8 s** (≈6.0 s/step) | 34.9 s |
        | Q4_K, 25 steps, CFG 6 | **222.8 s** (≈8.7 s/step) | 379.4 s |
        | bf16, 25 steps, CFG 6 | 224.2 s | |

        The Q4_K weights go through F16 GEMMs after dequantization, the bf16
        ones through BF16 GEMMs: the same speed. Text renders correctly. At 8
        steps with CFG the base model is undercooked on both engines (grids,
        wrong colours); it wants its 25–40.
      - **drift.** Its configuration launches the runner as is.
      - **Editing, done 2026-09-25** with `--llm_vision` (Qwen3-VL-8B's
        mmproj), as diffusers' pipeline: each reference scaled to about 1024²
        (sides multiples of 32), `<imageN><|vision_start|><|image_pad|>…` in
        the user turn, the text encoder given the tower's tokens and the
        deepstack taps (added after its first three layers, `SeenImages`) with
        mRoPE on the grid. In the transformer's prefix each reference's
        slots (one per 2 × 2 latents) become its VAE latents through
        `img_in`, modulated from `t = 0`, attending bidirectionally within
        itself (`Segment`s); its frame is the position reached and the text
        after resumes past its longer side, and the image made takes the
        frame after the prompt. Tiny diffusers model with two references
        among the text: 0.0028% (`Cpu`), 0.0044% (`Hip`). Live: "replace the
        red circle with a yellow star" on the shapes picture, 25 steps at
        768² with CFG 5, 5.4 s a step, the rest kept.
      - **Fixed with it:** the text encoder read the residual stream after
        the second-to-last layer; diffusers and sd-cpp read it after the
        last.
    - **Done 2026-09-29: HiDream O1 Image** (Dev and full), txt2img and
      img2img (`models/HiDreamO1`, `diffusion/HiDreamO1Pipeline`).
      - **Reference.** HiDream-ai's own code (`HiDream-O1-Image`, main:
        `models/pipeline.py`, `flash_scheduler.py`,
        `qwen3_vl_transformers.py`); sd-cpp's `hidream_o1.hpp` is the tool to
        beat.
      - **Model.** Qwen3-VL-8B's text model is the transformer (a
        `DenseDecoder` under `model.language_model.`, mRoPE interleaved
        24/20/20, θ 5·10⁶). The prompt is Qwen's user turn, then
        `<|boi_token|><|tms_token|>` after the open assistant turn; the
        timestep token's embedding is the timestep's (`1000 t`'s sinusoid
        through an MLP, t = 1 − σ). The image is 32 × 32 pixel patches, each
        channel-major (3072 values): in through `x_embedder` (3072 → 1024 →
        4096), out after the final norm through `final_layer2` as x̂, the
        clean image. No VAE, no separate text encoder; the tokenizer is
        HiDream's `tokenizer.json` (`--tokenizer`).
      - **Attention.** The prompt causal among itself; the timestep token and
        the patches see everything, both ways. The prompt never depends on
        the step, so it runs once into the sequence (`DenseDecoder.prefill`),
        and each step runs the generated tokens alone over it
        (`attendAll`). Patches sit at (4096, 4096 + row, 4096 + column).
      - **Sampling.** Unguided (CFG 1, the Dev checkpoint) as the official
        flash scheduler: the 28 distilled timesteps (other step counts
        resample their curve), then `z = (1 − σ_next) x̂ + σ_next · 7.5 ·
        noise`, fresh noise each step clipped to 2.5 of its deviation;
        starting noise of deviation 7.5. Guided (the full checkpoint): Euler
        on `(x̂ − z) / σ` over sd-cpp's flow-shifted schedule, CFG on x̂,
        noise 8, the unconditional prompt a space unless a negative prompt is
        given. The official full pipeline runs UniPC there, not yet here.
        sd-cpp runs Euler for both, with noise × 8.
      - **Weights.** ComfyUI's scaled fp8: every fp8 E4M3 safetensors weight
        is dequantized to BF16 × its `weight_scale` on first load
        (`WeightSource`, any model), then runs through BF16 GEMMs.
      - **Tests.** The distilled schedule and the patch layout against the
        official pipeline; no tiny golden model yet (the official model class
        needs flash-attn and CUDA).
      - **On the laptop** (Dev fp8, 28 steps, the same seed and prompt, a fox
        holding a "drift runner" sign):

        | | runner | sd-cpp |
        | --- | --- | --- |
        | 2048² | **113.5 s** (≈4.0 s/step) | 182.7 s (6.3 s/step) |
        | 1024², 8 steps | 8.3 s (≈1 s/step) | |

        The runner's picture is finished: the prompt followed, the sign's
        text right, fur and snow sharp at full size. sd-cpp's, with the
        same file and tokenizer, missed the prompt (a man on a bench) under a
        speckle of noise.
      - **drift.** Migration 008 reseeds the architecture: the drift runner
        listed, HiDream's `tokenizer.json` a required checkpoint (sd-cpp takes
        it too), 2048² by default (the sizes it was trained on are all about
        2048²), sides of 32. The runner's image server takes `--model` for a
        whole model in one file, `--vae` and `--llm` then absent.
      - **LoRAs** at run time on every linear (`LoraUpdates`): the decoder's
        seven per layer (`DenseDecoder.loraSites`, the prompt's prefill
        included), the timestep MLP, the bottleneck and the pixel head. A
        file's targets are the checkpoint's names less `model.` (Civitai's
        O1 LoRAs: `diffusion_model.language_model.layers.N.…`, rank 32).
        Live with Civitai's "Excellent Full Nude" (ai-toolkit, trained on the
        full checkpoint at 1024²; ai-toolkit's noise ×8, t = 1 − σ and x₀
        target match ours): on the full model at CFG 5 the image is clean
        with the LoRA's effect; on Dev it is washed out, soft and faintly
        gridded, less so at 0.5 — the LoRA does not transfer to the distilled
        checkpoint.
      - **Left:** reference images (the vision tower and the deepstack
        mergers, which ComfyUI's file does not carry), UniPC.
14. **Video and audio.** Wan 2.2 (cross-attention, two experts), LTX,
    HunyuanVideo, MiniMax; the causal 3-D VAE and the audio VAE.
    - **Done 2026-09-30: MiniMax H3, text to video** (`models/MiniMaxH3`,
      `models/MiniMaxH3Vae`, `diffusion/MiniMaxH3Pipeline`,
      `server/VideoFiles`). diffusers 0.40 (`MiniMaxH3Transformer3DModel`,
      `AutoencoderKLMiniMaxH3`, the `t2va` modular blocks, `MiniMaxH3Scheduler`)
      is the reference; sd-cpp (`minimax_h3.hpp`) for the pruned files' AdaLN
      curve table, which diffusers lacks.
      - **Text.** The prompt tokenized bare (Qwen3's `tokenizer.json`, passed
        as `--tokenizer`: the text encoder's GGUF has no metadata and no
        vocabulary), Qwen3-VL-32B's residual stream after layer 50. The GGUF
        holds transformers' names and is cut after layer 50 without a final
        norm; `Qwen3Config.fromWeights` reads its shape (θ 5 000 000).
      - **DiT.** 50 single-stream blocks of 5376 over one packed sequence
        `[text | audio | video]`, 56 heads of 128 (wider than the stream),
        q/k RMS norms, a rotate-half RoPE on each head's first 96 values over
        (t, h, w) with fractional positions (`MiniMaxH3Layout`: latent frames
        span 5/3 × (1, 4, 4, 4, 4), the spatial axes centred and scaled to the
        aspect), SwiGLU of 14336 (`fc1` = [gate; value]). Each row is
        modulated by its (timestep, modality) row: text and video at the
        video's t, audio at its own. The pruned files replace the time MLP by
        a table of 1025 points of 8 (`adaln_t_table`, interpolated, no SiLU),
        projected on the host; the full files' time MLP runs on the GPU.
        Q4_K weights go through BF16 GEMMs (`Ops.wideProducts`: dequantized to
        BF16, not F16; sd-cpp prescales the MLP by 1/128 against F16's range).
      - **New ops:** `ropeTable(halves = true)` (transformers' `rotate_half`
        from an angle table), `dequantize_bf16_*`.
      - **Sampler.** t = 1 − σ, the velocity points to the data, one pass per
        step (guidance-distilled). `--steps N` runs N evaluations over
        `linspace(1, 0, N + 1)` shifted by 12 (video) and 3 (audio); sd-cpp's
        default is 20. CFG above 1 runs the negative prompt as for images.
      - **VAE.** The ViT decoder (36 blocks of 2048, 4 registers and a zero
        token, layer-scaled residuals), in the released tiling: chunks of 7
        latent frames (5 new, 5 frames cross-faded), 256-pixel tiles
        overlapping by at least 64, blended. The latent statistics,
        `post_quant_conv` and the embedding fold into one affine at load
        (inputs padded to 32 channels); the per-head qkv is regrouped at load.
        F16 products, as the released recipe.
      - **Output.** `vid_gen` on the image server (`ImageServer` takes an
        image or a video pipeline), one webm encoded by **ffmpeg** (VP8; a
        dependency of the runner's videos, to weigh with the distribution
        questions). Frames round up to `17n + 5`, sides to multiples of 32;
        24 fps whatever is asked.
      - **Checks.** A tiny diffusers transformer over the pipeline's own
        layout matches to 0.0047% (`Cpu`) and 0.18% (`Hip`, BF16 products);
        the layout's positions to F32 rounding; a tiny VAE over two chunks of
        four tiles to 6·10⁻⁵ (`Cpu`) and 8·10⁻⁴ (`Hip`) of a pixel.
      - **Live** (864 × 480, 56 frames, seed 42, the pruned Q4_K and the
        Q4_K_M text encoder): a clean, coherent clip. **470 s against
        sd-cpp's 500 s**, sd-cpp with drift's default spectrum cache (it
        computed 11 of 20 steps at 38.2 s each, 420 s), the runner all 20 at
        19.2 s rising to 23.2 s as the laptop throttles (≈ 19 TFLOPS: 285
        TFLOP of GEMMs and 78 of attention a step). Text 11.4 s → about 1 s;
        video decode 66.4 s → 41.7 s; loading 1.9 s.
      - **Left:** a step cache.
    - **Done 2026-09-30: Wan 2.2 A14B, text to video and I2V**
      (`text/Unigram`, `models/Umt5`, `models/Wan`, `models/WanVideoVae`,
      `diffusion/WanPipeline`). diffusers 0.40 (`WanTransformer3DModel`,
      `AutoencoderKLWan`, `WanPipeline`, `WanImageToVideoPipeline`) and
      transformers (`UMT5EncoderModel`) are the references; sd-cpp's
      `vid_gen` for the two experts' schedule.
      - **Text.** UMT5-XXL (transformers' names; ComfyUI's fp8 as BF16):
        pre-norm blocks, no score scaling, each block's own relative-position
        bias (32 buckets, bidirectional, 128 apart), gated GELU. Its tokenizer
        is the new `Unigram` (Metaspace, Viterbi over the pieces, unknown runs
        fused, `</s>` appended), passed as `--tokenizer` (UMT5's
        `tokenizer.json`); it matches HuggingFace's on the whole corpus. The
        prompt's rows, then zeros up to 512, as diffusers and Wan's own code
        pad them. `shortAttention` takes an additive bias for it.
      - **DiT.** 40 blocks of 5120, heads of 128 (the files record no head
        count), self-attention with RMS norms across the heads and an
        interleaved 3-axis RoPE (22/21/21 pairs), cross-attention to the text
        (keys and values made once per prompt and expert, `WanText`), GELU
        MLP, AdaLN from the timestep's six vectors plus each block's table.
        Latents live as 2 × 2 patch rows, channel-major (`packPatches`); the
        I2V checkpoints' 20 conditioning channels follow the noise's in each
        row; the patch embedding is padded to 160 inputs and the head's rows
        reordered at load. BF16 products (`wideProducts`).
      - **Experts and schedule** as sd-cpp: one Euler schedule of
        `--high-noise-steps` + `--steps`, but `linspace(1, 0, N + 1)` through
        the flow shift, where sd-cpp's discrete one puts N points on 999 → 0
        and appends 0 (its last step a no-op); so videos are not sd-cpp's
        frame for frame, the high-noise expert first at `--high-noise-cfg-scale`,
        or while σ ≥ `--moe-boundary` (0.875) when its steps are not given.
      - **I2V conditioning.** A mask of 4 (the first latent frame's set with an
        init image) and the VAE's latents of the init image then zeros, or of
        zeros alone; without an image it is made once per size and kept.
      - **VAE.** The Wan 2.1 VAE on video: its causal 3-D convolutions as
        streams of 3×3 ones over the frame and the two before it (zeros before
        the first), the decoder's time convolutions doubling frames after the
        first, the encoder's halving them after the first. Frame by frame,
        untiled, as diffusers' default.
      - **Checks.** UMT5 to 0.0001% (`Cpu` and `Hip`); a tiny I2V transformer to
        0.0073% (`Cpu`) and 0.46% (`Hip`, BF16 products); the video VAE,
        encoding 9 frames and decoding 3 latent frames through the streams, to
        0.27% and 0.34% (`Cpu`), 0.25% and 0.55% (`Hip`).
      - **Live** (François's configuration: the Civitai FASTMOVE V2 Q8 experts
        (I2V), the uncensored UMT5 fp8, the Wan 2.1 VAE; 832 × 480, 33 frames,
        4 + 4 steps, seed 42): **302 s against sd-cpp's 603 s**, with the
        high-noise CFG at 1 on both. Steps 31–36 s (sd-cpp 63), the grey
        conditioning 17 s once (sd-cpp 29 s every time), decode 35 s (60).
      - **sd-cpp skips the high-noise CFG** when the low-noise one is 1: it only
        encodes the negative prompt for the low-noise scale (`video.cpp`,
        `use_uncond`), so drift's `--high-noise-cfg-scale 3.5` did nothing
        there. The runner applies it: with it, the step-distilled FASTMOVE
        finetune comes out overcooked (and each high-noise step costs two
        passes, 67 s). Upstream's to fix; set the high-noise CFG to 1 for
        step-distilled experts.
      - **Left:** Wan 2.2 5B (TI2V, the Wan 2.2 VAE), VAE tiling, the GGUF UMT5
        (llama.cpp's names).
    - **Done 2026-09-30: LTX 2.5, text to video** (`models/Gemma4Text`,
      `models/LtxText`, `models/Ltx2`, `models/LtxVideoVae`,
      `diffusion/LtxPipeline`). diffusers 0.40 (`LTX2VideoTransformer3DModel`,
      `LTX2TextConnectors`, the `LTX25AutoBlocks`) and transformers
      (`Gemma4UnifiedTextModel`) are the references; sd-cpp crashes on these
      files (bugs/35), so there is no tool to beat.
      - **Text.** Gemma 4 12B from the text encoder's file (transformers'
        names under `model.`; its tokenizer is the file's own `tokenizer_json`
        tensor, read raw): sliding layers of 8 key heads of 256 and global
        ones of one 512-wide head that is also the value, the proportional
        RoPE on a quarter of the global heads, plain-weight RMS norms, value
        norms, layer scalars, embeddings × √hidden rounded to BF16. Every
        hidden state (the last one normed), each RMS-normed per token, side
        by side, scaled and projected per stream (`LtxTextFeatures`, its
        weight's columns regrouped at load); then the transformer file's
        connectors: the prompt's rows then learnable registers up to 1024,
        8 blocks of gated attention with the split 1-D RoPE. The attention
        runs through `shortAttention`, now up to 512-wide heads, its causal
        mask an additive bias.
      - **DiT.** 48 blocks of two streams (video 32 × 128, audio 32 × 64):
        self-attention, prompt cross-attention (the query modulated and gated,
        the rows by the prompt AdaLN), audio-to-video and video-to-audio
        attention (temporal RoPE both sides), GELU MLPs; every attention gated
        per head by 2σ(logits); nine AdaLN-singles. The split RoPE gives each
        head its own frequencies: `ropeTable` takes `[tokens, heads, pairs]`
        tables. Positions are pixel-span middles, time in seconds from the
        frame rate, so `--fps` is a model input (a value option now, no longer
        ignored). BF16 weights through BF16 GEMMs.
      - **Schedule.** The distilled σ list for 8 steps; other counts
        `linspace(1, 1/N, N)` through the resolution's exponential shift.
        `--flow-shift` is not read. CFG above 1 runs the negative prompt
        (plain CFG, not diffusers' rescaled guider).
      - **VAE.** The conv decoder, its structure from the weights (residual
        stages, depth-to-space upsamplers ×2 in time, space or both, their
        strides from their channels), non-causal (edge frames repeated), pixel
        norms; its channels regrouped at load so the rearrangements are
        `splitHalves` and `unpackPatches`.
      - **Checks.** Gemma 4 to 0.0002%; the features and connectors to
        0.013% (`Cpu`) and 0.017% (`Hip`); the transformer to 0.011% and
        0.013%; the conv VAE (diffusers' own blocks in the file's order) to
        0.21% and 0.38%.
      - **Live** (François's configuration: the official distilled BF16
        files; 512 × 512, 64 frames → 57, 8 steps, 16 fps): a clean,
        coherent clip in **71 s**: steps of 5.3–8 s, decode 21.7 s, loading
        19 s.
      - **Left:** the diffusion decoder (LTX 2.5's default in diffusers), the
        rescaled guider.
    - **Done 2026-09-30: the soundtracks of MiniMax H3 and LTX 2.5**
      (`models/BigVgan`, `models/MiniMaxH3Audio`, `models/LtxAudio`), from
      the configuration's `--audio-vae`; without it the videos stay silent
      and the runner says so. diffusers 0.40 (`AutoencoderKLMiniMaxH3Audio`,
      `AutoencoderKLLTX2Audio`, `LTX2VocoderWithBWE`) is the reference.
      - **New ops:** `conv1d` (im2col then a GEMM with F32 output; BF16 or
        F32 operands), `overlapAdd` (a transposed convolution's second half:
        the input times the weight regrouped `[out × taps, in]`, then summed
        into place), `antiAliasedSnake` (×2 upsampling by the stored
        Kaiser-sinc filter, SnakeBeta, filtered back down, in one kernel).
        `HipBlas.gemm` takes an output type apart from the inputs'.
      - **BigVGAN**, shared: both files use the original names (H3's
        `ups.i.0` and `activations`, LTX's `ups.i` and `acts1`/`acts2`), their
        weight norms folded; the upsampling rates are the files' configs,
        not tensors (H3 5, 5, 2 × 5; LTX 5, 2 × 5, its extension 6, 5, 2, 2,
        2). F32 weights and patches: a vocoder moves its waveform by 10% for
        a 0.2% change of its input, so it adds no rounding of its own, and
        F32 costs nothing measurable here.
      - **H3:** the audio rows channel-major, each channel denormalized by the
        file's `latents_mean`/`latents_std`, `dec_in_proj`, BigVGAN ×800 to
        32 kHz, clamped; the model is mono, so stereo is two decodes.
      - **LTX:** the packed rows denormalized by the per-channel statistics,
        unpacked to `[L, 16 bins, 8 channels]`; the mel decoder causal in time
        (a zero row before, the usual 3×3 padding, upsamplers dropping their
        first row), pixel norms, to `[4L − 3, 64, 2]`; the vocoder to 16 kHz;
        the bandwidth extension: each channel's causal STFT (the file's basis,
        512 taps, hop 80) and log-mel on the host, the second BigVGAN ×240,
        plus the ×3 Hann resampler (computed, not stored), clamped: 48 kHz.
      - **Checks.** Tiny decoders (`fixtures/tiny_diffusion.py`
        `minimax_h3_audio`, `ltx_audio`) to 0.43% and 0.31% (`Cpu`, BF16
        weights then), 0.52% and 0.60% (`Hip`). On the released files with
        random latents (`gpuTest` `SoundtrackCheck` against diffusers on the
        CPU): H3 to 0.0016% RMS; LTX's mel to 0.18%, its vocoders on the same
        mel to 0.0016%; the whole LTX chain differs by 9% RMS, which is
        diffusers' own spread for a 0.2% change of the mel (waveform phase).
      - **Live.** MiniMax H3 (640 × 352, 56 frames, 20 steps, the pruned Q4_K):
        2.3 s of 32 kHz sound decoded in 0.3 s, a clean song and guitar.
        LTX 2.5 (512², 57 frames at 16 fps, 8 steps): 3.5 s of 48 kHz sound
        in 0.8 s; François heard the birds and the outdoor sound of the
        prompt, but with "robotic" parts and noise H3's lacks.
      - **LTX's robotic sound was the prompt.** Every stage matches diffusers
        on the released files: the text features and connectors to 0.2%
        (`gpuTest` `LtxTextCheck`), one step of the transformer to 0.11% on
        the audio against diffusers in F32 (`LtxStepCheck`; the video 5.9%,
        diffusers' own BF16 run 21.6%), the decoder above. LTX 2.5 is trained
        on single-paragraph captions of about 150–220 words that describe
        the sound (ComfyUI's official workflow always rewrites the prompt
        with its Gemma enhancer, `TextGenerateLTX2Prompt`). A one-line prompt
        naming sounds in passing ("barks … birds chirping …") left the audio
        near silence (−48 to −53 dB RMS at 16 or 24 fps; Euler ancestral −116
        dB at 16 fps), which the vocoder turns robotic; the same scene as a
        caption gave −18 dB and the sounds it names, which François found as
        good as what Civitai shows for LTX 2.5. A short prompt with a quoted
        line of speech does speak (−23 to −25 dB, the caption −15 to −17),
        and the frame rate (16 or 24 fps) changes neither: what matters is
        how clearly the prompt describes the sound.
      - **No clamp: the track is fitted.** Long captions make LTX's audio
        hotter than full scale (182 samples at the clamp in 4 s of speech),
        and the references' per-sample clamp to [−1, 1] saturates it, which
        François heard. `Soundtrack.fitted` scales the whole stereo track
        down when its peak passes 0.97 (room for Vorbis' overshoot), for both
        models; the decoders no longer clamp.
      - **Also added while looking:** request knobs `audio_cfg_scale`,
        `modality_scale` and `audio_modality_scale` (the modality guidance,
        a pass with the streams isolated, as diffusers and ComfyUI) and
        `sample_params.sample_method` `euler_a` (ComfyUI's Euler ancestral for
        flows); drift sends none of them. The modality guidance made the
        short prompt's sound louder but more robotic.
      - **Left:** an LTX caption writer in the prompt library (ComfyUI's LTX
        2.4+ system prompt), so a short idea reaches the model as a caption.
    - **Done 2026-10-01: the video requests' inputs.** `VideoRequest` carries
      an end image, LoRAs (`VideoLora`, with sd-server's `is_high_noise`),
      reference media, guides (`VideoGuide`: media held at a frame, negative
      from the end) and a control video (`VideoControl`: strength, a start and
      end fraction of the steps, a mask whose white regenerates over a source
      video). Uploaded media (`Media`: a still, a clip with its soundtrack, a
      sound) are decoded by `VideoFiles.decode` (the JDK for images, else
      ffprobe and ffmpeg; clips to a 1080 short edge). Each pipeline says what
      it takes (`takesEndImage`, `takesLoras`, `takesReferences`,
      `takesGuides`, `takesControl`), the capabilities advertise it, and a
      request asking for more is refused. `--control-net` is a value option,
      refused for every family but MiniMax H3. A mask alone (with its source)
      inpaints without a control video.
      - **Shared fixes found on the way:** safetensors `I64` is read (a Wan
        Lightning pair's `.alpha` 8 at rank 64 had been dropped, running it at
        8×); F16 LoRAs no longer crash (converted by way of F32, every model);
        `HipOps`' hipBLAS path converts dense F32 weights element by element
        (the dequantizers walk rows in blocks of 32, so rows of 196 or 68 came
        out wrong without an error; `LinearTests` covers them now).
    - **Done 2026-10-01: Wan 2.2 A14B LoRAs and end images** (`Wan.useLoras`,
      `WanPipeline.condition`).
      - **LoRAs** at run time on every linear (self- and cross-attention q, k,
        v, o; `ffn.0`/`ffn.2`; the text and time embeddings;
        `time_projection.1`; the head, its up rows permuted like its weight),
        in the original, diffusers and kohya (`lora_unet_`, `lycoris_`) names.
        A high-noise LoRA goes on the high-noise expert, the rest (and all of
        them with one expert) on the low-noise one, as sd-cpp routes them. The
        cross-attention keys and values are made after the LoRAs are set. All
        11 of François's files map fully (400 pairs on 400 sites each,
        `WanLoraCheck`). `diff`/`diff_b` deltas are not read.
      - **End image**: diffusers' `last_image`, the VAE latents of [first,
        zeros…, last] with the mask on the first latent frame and the last
        channel of the last one; an end image alone is taken, as sd-cpp takes
        it (diffusers refuses it).
      - **Checks.** Two tiny LoRAs (both namings, F32 and BF16, float and I64
        alphas) against diffusers' merged weights to 0.0098% (`Cpu`) and 0.50%
        (`Hip`); the conditioning with first, first and last, and last alone to
        0.33% and 0.56%.
      - **Live** (the FASTMOVE V2 Q8 experts, 832 × 480, 33 frames, 4 + 4
        steps, CFG 1, a first and a last frame made for the purpose): 365 s
        with the 80s-fantasy high/low pair (steps 37 s against 35 s without),
        both keyframes held exactly; the pair changes the clip.
    - **Done 2026-10-01: LTX 2.5 LoRAs, image to video, keyframes and guides**
      (`Ltx2`, `LtxVideoVae`, `LtxPipeline`). diffusers 0.40's condition blocks
      (`LTX2ConditionPrepareLatentsStep`, `LTX2ConditionLoopBeforeDenoiser`)
      are the reference; ComfyUI's `LTXVAddGuide` for pixel-frame placement.
      - **LoRAs** at run time on every linear, the text connectors' too (they
        now load with the transformer): original names and diffusers'
        (`proj_in`, `time_embed`, `av_cross_attn_*`, `connectors.*`), a target
        applied when its shape matches. François's BEANFLK (rank 16, 1152
        pairs) maps fully; his seven LTX 2.3 LoRAs match LTX 2.5's names and
        shapes too (not offered).
      - **Conditions** at strength 1: one at frame 0 (a still, or a clip the
        video continues, 8n + 1 frames) replaces the first latent frames; the
        others are appended as keyframe tokens with their own positions (a
        still at [i, i + 1), the end image at the last pixel frame as ComfyUI's
        −1, where diffusers' latent −1 lands 8 frames earlier; a clip from the
        multiple of 8 plus 1 at or before its index, encoded after a throwaway
        frame). Their tokens take timestep 0 in the video's own AdaLN (blocks
        and head); the prompt's, cross-modal and audio ones keep σ; clean
        latents are restored after each step; appended tokens are dropped
        before decoding. Stills go through H.264 at CRF 18 (ffmpeg), as LTX 2.5
        learned them. A guide at frame 0 overrides the init image.
      - **VAE encoder**: causal, space-to-depth downsamplers with grouped-mean
        residuals (rearranged on the host), the mode normalized by the
        per-channel statistics.
      - **Checks.** A conditioned step (first frame held, a still and a clip
        appended) to 0.011% (`Cpu`) and 0.060% (`Hip`); a LoRA on the
        transformer and connectors, both namings, 0.013% and 0.085%; the
        encoder (9 → 2 and 1 → 1 frames) 0.39% and 0.54%.
      - **Live** (832 × 480, 57 frames at 16 fps, 8 steps): first and last
        frame in 80 s (steps 8.2 s, decode 10 s), both held exactly, the middle
        a dissolve between the two shots; reruns identical; BEANFLK adds 1.1 s
        a step. Extending the Wan clip (a guide at frame 0, 89 frames): 150 s, a
        coherent continuation.
      - **Left:** strengths below 1, a guide clip's frame rate and soundtrack
        (continuing the audio needs the audio VAE's encoder), the keyframe
        marker embedding (diffusers' forward pass does not add it), a 100-frame
        extend clip is slow (the encoder's downsamplers run on the host).
    - **Done 2026-10-01: MiniMax H3 LoRAs, keyframes (fl2va) and guides**
      (`models/MiniMaxH3VideoEncoder`, `models/MiniMaxH3AudioEncoder`,
      `diffusion/TorchRandom`, `diffusion/MiniMaxH3Conditions`,
      `diffusion/SoundResampling`). diffusers 0.40 (the `t2va`/`fl2va` blocks,
      `AutoencoderKLMiniMaxH3*`) and ComfyUI (`MiniMaxH3AddGuide`,
      `PackedLayout`) are the references.
      - **LoRAs** at run time on every linear (q, k, v split from `qkv_proj`;
        gate and value from `fc1`; the refiners, projections, both heads, every
        block's `adaln_proj`, the final AdaLN, the time MLP), in the original,
        musubi (`lora_unet_`) and diffusers names. The pruned files' AdaLN
        updates run on the host with their projections. A target without a
        weight (the time MLP on a pruned file) is reported, not fatal. All 10
        of François's Civitai files map fully (104 to 258 pairs).
      - **Noise.** torch's CPU generator (MT19937, both `randn` paths) in
        diffusers' draw order, so a seed gives diffusers' noise; t2va videos
        changed for a given seed.
      - **Video encoder.** A causal 3-D CNN: reflect-padded 3×3×3 convolutions,
        time halved at levels 1 and 2, 256-pixel tiles blended in latent space,
        chunks of 17 frames with the last 3 latents dropped. F16 weights and
        patches with F32 sums: 0.24% of the released weights' reference (BF16
        was 2.5%, and 16% on a clip). New: `conv3x3` reflect padding and F16
        weights.
      - **Audio encoder.** The DAC encoder with Snake (a new op), `pre_block`'s
        head mean, pool and `proj` folded into one linear; 0.04% on the
        released file.
      - **Presentation.** `"<Picture i>: "` and a vision block per keyframe,
        then the prompt, no chat template; the vision tower read from the text
        encoder's `visual.` (16 heads, deepstack 8/16/24; `QwenVision` already
        covered Qwen3-VL), interleaved mRoPE 24/20/20; the vision rows
        modulated as video.
      - **fl2va.** Keyframes as a seed-42 draw rounded to F16, mixed at 0.999,
        held at max(t, 0.999); stretched onto the canvas, the end image
        cover-cropped when both are given.
      - **Guides** as ComfyUI: frames centre-cropped (17k + 5) and encoded as the
        posterior mean, mixed at 0.999 from the request seed restarted per
        guide; sound resampled to 32 kHz as torchaudio does, cut to the
        soundtrack, held clean.
      - **Partitions.** fl2va and ref2va share every name and shape and carry
        no metadata; told apart by the file name, else by
        `final_layer.norm.weight`'s first values.
      - **Checks.** LoRAs 0.11% and 0.14% (`Cpu`), 0.41% (`Hip`); the encoder on
        the released weights 0.24% (a frame) and 0.39% (a clip); the real
        vision tower against transformers 0.048%; an fl2va step 0.0026% and
        0.44%; a guided step 0.0029% and 0.53%; `randn` to 1.3·10⁻⁶; the
        resampler 0.0024%.
      - **Live** (the fl2va Q4_K, 832 × 480, 56 frames, the turbo v4 LoRA, 8
        steps): first and last frame in 341 s, a real drive between the two
        keyframes, held exactly. A step costs 22.6 s bare, 27 s with the turbo
        LoRA, 35 s with two keyframes too. Guides (the Wan clip at 0, a still
        at −1, a sound at 0): 586 s; the clip continued, the still reached, the
        soundtrack's envelope correlates 0.92 with the guiding sound.
      - **Open:** keyframes and guides together have no reference; the
        negative pass keeps the condition rows (ComfyUI's has none).
    - **Done 2026-10-01: the MiniMax H3 Fun ControlNet union; references
      (ref2va) built but broken live** (`diffusion/MiniMaxH3References`,
      `MiniMaxH3.ControlNet`).
      - **References** (diffusers' `ref2va` blocks, ComfyUI's
        `MiniMaxH3ReferenceToVideo` the second reading), on the ref2va
        checkpoint only, which takes no keyframes or guides: up to 9 images, 3
        clips and 3 sounds (12 in all, never sounds alone), in the request's
        order. Images scaled down to a 2048 short edge (never up, PIL's
        LANCZOS); clips to 24 fps on the canvas their own aspect resolves to
        (768 short edge, at most 768 × 1344); soundtracks cut to the video's
        length, stereo, 32 kHz. `<Picture i>: ` with a vision block; `<Audio
        j>: ` first for anything with sound; `<Video k>: ` then a `<t
        seconds>` label and a two-frame vision block per pair read at 2 fps
        (the tower's patch embedding unfolded for the pairs). Packed `[text |
        conditions | references | audio | video]`, each reference on its own
        grid; the references advance a shared rotary clock (an image 1, a sound
        its latents, a clip the longer of its two spans) and the generated rows
        start where it ends. One pass per step (guidance-distilled; another
        CFG is ignored with a warning).
      - **ControlNet** (ComfyUI's `MiniMaxH3FunControl*`): the 2.0 file, 10
        blocks at 0, 5, …, 45 with the pruned files' 8-wide AdaLN (the older
        union file, 5 blocks with the full AdaLN, is refused against the pruned
        transformer). The control is encoded as the posterior mean; with a
        mask, its visibility (trilinear) and the masked source too, 49
        channels. Before block 0 the video rows become the projected control,
        pass `before_proj` and join the stream; each control block runs beside
        its base block with the same spans and angles, and `after_proj` adds
        its output at the strength, zero on audio rows; both CFG passes, any
        conditioning, between the start and end fractions of the shifted
        schedule.
      - **Checks.** Token ids identical to diffusers'; the presentation's hidden
        state 0.011% (`Cpu`) and 0.17% (`Hip`); positions 1.3·10⁻⁶; a ref2va
        step 0.0036% and 0.21–0.32%; a controlled step (with and without a mask,
        ComfyUI's patch) 0.0036% and 0.6%.
      - **Live, ControlNet** (the fl2va Q4_K, the union 2.0 BF16, the edges of
        the Wan clip, a blue pickup in a desert for prompt, the turbo LoRA, 8
        steps, 56 frames at 832 × 480): 421 s (28.5 s a step); the new subject
        follows the control's whole motion (side view, turn, driving away). A
        ControlNet loaded but not asked for leaves the video bit-identical.
      - **Live, references: broken, cause open** (the ref2va Q4_K, 832 × 480).
        The reference's subject comes through (the red convertible), but the
        generated rows are corrupted: an image alone at 20 steps gives a foggy,
        near-static clip at 56 frames (530 s) and at 124 frames (1366 s, so
        not the 5–15 s duration), and on a sunny street prompt a dark frame of
        vertical streaks around the car. An image and a 1 s clip at 12 steps:
        1641 s (65 s a step), smeared. diffusers' `transformer` and
        `transformer_ref` configs are identical, and the shared H3 path is
        unchanged (a t2va rerun is bit-identical), so the fault is in the
        references' inputs at real sizes, beyond the tiny fixtures. Next: one
        step on the released file against diffusers (or ComfyUI's GGUF loader)
        with the same image, tensor by tensor. drift does not offer the ref2va
        checkpoint in a starter configuration.
      - **Left:** ComfyUI's `"match"` sizing of references (diffusers' canvas
        sizing makes a clip reference cost more rows than the video itself);
        the 5–15 s duration check.

**Tooling.**

15. **A quantizer.** K-quants and ROCmFP4 written by the runner, used by
    drift's conversion ([`25`](25-model-conversion.md)) so a BF16 release
    becomes a runnable file on this machine. It can come any time after
    step 3, and should come before step 10 if no ROCmFP4 build of Qwen 3.8
    Flash Next is published.
