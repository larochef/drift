# 25 — Model conversion

**Status:** partial — steps 1 (convert from the Model Cache) and 2 (int8 and
scaled-fp8 dequantization) done and verified on real files; step 3 (importance
matrices and rules presets) parked, see Remaining
**Depends on:** 05, 06, 13

Turn weights drift cannot run well into weights it can, on the user's machine
and from inside drift: GGUF quants that run on the GPU, sized to the memory at
hand, registered as models in the same family as their source. The motive is
INT8 tensorwise checkpoints (ComfyUI's format, convrot or not), which sd-cpp
runs on the CPU on Vulkan and ROCm; an fp8 or full-precision release runs on
the GPU but is often larger than it needs to be.

## What it does

- Model Cache → On disk: every non-LoRA `.safetensors`/`.gguf` row has
  **Convert…**, opening a modal that reads the file's header (format, tensor
  and parameter counts, bytes per dtype) and offers a target type — `q8_0`
  (default), `q6_K`, `q5_K`, `q4_K`, `iq4_xs`, `q3_K`, `f16`, `bf16` — with an
  estimated output size; the family (fixed when a model references the file,
  a text input otherwise, like Assign); an output name
  (`<source base>-<type>.gguf`, refused if it exists); and, folded under
  Advanced, raw `--tensor-type-rules`, a thread count, and "keep the
  intermediate" for dequantized sources. Short disk refuses the job.
- An int8 source (ComfyUI `int8_tensorwise`, with or without `convrot`) or an
  fp8 source with scale tensors is dequantized by drift to an F16 safetensors
  first, then converted; the modal explains the pre-step and its temporary
  disk. Plain bf16/fp8/GGUF sources go straight to sd-cli.
- Jobs run one at a time on a worker, show a tensor bar and a log tail, appear
  in the downloads panel ("n / m tensors"), and can be cancelled (partial
  output removed). The Model Cache page lists them above its tabs.
- The result lands in `models/<familyId>/converted/<name>.gguf`, is listed in
  the Local on-disk tab under the family, and is registered as
  `Model(id = "<source model id or base>-<type>", label = "<source label> ·
  <TYPE>", source = Local(path), format = "gguf", parameters = source model's
  parameters)` — assignable in every run configuration naming the family.

## Shape

- `shared/.../Conversion.scala`: `ModelFileInfo(format, tensorCount,
  parameterCount, bytesByType, comfyQuant: Option[ComfyQuantInfo(format,
  convrot, groupSize)], scaledFp8)`, `ConversionTypes` (bits per weight for
  the estimate), `ConversionRequest(path, familyId, targetType, rules,
  outputName, threads, keepIntermediate)`, `ConversionJob(…, state: Queued |
  Dequantizing | Converting | Registering | Completed | Failed | Cancelled,
  progress, detail, logTail, modelId)`.
- Endpoints: `GET /api/cache/files/inspect?path=`, `POST /api/conversions`,
  `GET /api/conversions`, `POST /api/conversions/{id}/cancel`. Jobs ride the
  status socket as topic `conversions`; no persistence, logs stay in
  `~/.cache/drift/logs/conversion-<id>.log`.
- Backend `backend/.../conversion/`: `ConversionManager` (refusals, disk
  check, registration, start-up sweep of `converted/*.part`) over
  `ConversionJobs` (registry, one worker, pre-step, spawn, bar parsing) and
  `SdCppConvert` (argv), plus `Dequantizer`; `cache/ModelFileInspector` and
  `cache/SafetensorsHeader` read headers; `process/ProcessOutput` is the
  shared stdout reader (split on `\r` and `\n`); `RuntimeManager.sdCliOf`
  finds sd-cli on the default sd-cpp runtime.
- The command: `sd-cli -M convert -m <src> -o <out>.gguf.part --type <t>
  [--tensor-type-rules r] [-t n]`, no `-v`. Success = exit 0, no `[ERROR`
  line, and the output parses as a GGUF with tensors.
- Frontend: `pages/ConvertModelModal.scala`, `pages/ConversionJobList.scala`,
  `services/ConversionService`.

## Notes

- sd-cpp's converter is the right tool: it uses the runtime's own
  `ModelLoader`, so names, accepted types and the protected-tensor list
  (biases, scales, embeddings, in/out projections, rows not a multiple of the
  block) are those of the build that loads the file. `--type` takes any ggml
  type name, K types spelled `q4_K`; `--tensor-type-rules "regex=type,…"`
  first match wins; `--imat-in` repeatable; streams with a 1 GB budget; the
  progress bar is the loading bar `LogProgress.parse` already reads and
  prints regardless of verbosity. GGUF in, GGUF out works (requant).
- `tensor_should_be_converted` returns false for every int8 tensorwise
  tensor, and the converter never applies fp8 scales: those sources need
  drift's dequantization or they come out unchanged.
- The `convrot` rotation is copied from `ggml_regular_hadamard_group_f32` in
  leejet/ggml: scale by 1/√group, then radix-4 butterflies with the block
  `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`. The matrix is symmetric and
  orthonormal, so the same routine undoes it; one scale per output row. Any
  power of four dividing the width is a valid group (ComfyUI's `_build_hadamard`
  is `kron(h4,…)/√size`); real files carry 256 and 1024. Never guess the
  convention — the e2e fixtures use an independent closed-form formula.
- ComfyUI markers vary: `{"format": "int8_tensorwise", "convrot": true,
  "convrot_groupsize": N}` or the same keys with no `format`. An empty format
  is taken as int8 (the I8 dtype and the scale tensor decide); only a *named*
  other format is refused. fp8 scales come as Ideogram's `<m>.weight_scale`
  or ComfyUI's `scaled_fp8` marker with `<m>.scale_weight`; e4m3 is the `fn`
  variant.
- The intermediate is F16 (10-bit mantissa; a quantized source cannot exceed
  its range, proven from the scales, with F32 per tensor when it could), named
  `<output base>-dequantized.safetensors`; kept, it lists as a local orphan of
  the family to Assign and run for a GPU-vs-CPU comparison. Dropped tensors:
  `.weight_scale`, `.scale_weight`, `.comfy_quant`, `.scale_input`, the
  `scaled_fp8` marker.
- Cancel marks the job before killing the process; the worker deletes the
  `.part` after `waitFor`. A queued job is skipped when its turn comes.
- `Model.parameters` are argv flags, so no back-reference to the source lives
  there.
- Model id/label conventions for the converted entry are as built, pending a
  real-file objection.

## Remaining

- **Step 3 — importance matrices and presets. Parked 2026-09-18.** A **Collect** action on an
  sd-cpp run configuration runs `sd-cli` img_gen with the configuration's
  resolved argv over a batch of representative prompts (the project's prompt
  versions from a workspace, else a bundled neutral set) with
  `--imat-out <driftRoot>/imatrix/<familyId>/<date>.dat`, refining an existing
  file with `--imat-in`. The convert modal lists the family's imatrix files
  and passes `--imat-in` for each ticked one; IQ types are offered only with
  an imatrix (without one ggml quantizes them against a flat matrix).
  Presets for the rules field come last and only from measurements (same
  prompt and seed before and after, via the gallery's "same task on another
  configuration"): candidates are attention output projections and first/last
  blocks one type finer than the body; a preset is a per-architecture rules
  string in `architectures.json`, add-only. Upstream's Ideogram 4 recipe
  (no `--type`, rules quantizing only the big matrices to `q8_0`) is the
  safest first shape.
  `sd-cli` is the tool either way: upstream's `docs/imatrix.md` trains the
  matrix alongside ordinary generation (`--imat-out`, refined by passing
  `--imat-in` as well) and takes repeatable `--imat-in` when quantizing, so
  whether sd-server accepts the flag never comes up.
  Parked because the matrix only pays below 4 bits — q8_0 and q6_K, where
  conversions driven by disk space rather than VRAM land, do not notice one —
  and because nothing answers how few prompts still calibrate an image model
  honestly: "representative" spans subject, resolution, sampler, CFG and steps
  at once. Unparks when a model has to go below 4 bits. A source too broken to
  run can still be calibrated, by converting it to q8_0 first and collecting on
  that (upstream: training on an already-quantized model works fine), then
  requantizing from the F16 intermediate or the q8_0 itself.
  There is nothing to download instead: HuggingFace's imatrix files are
  language models; the one image repository, `Eviation/flux-imatrix`, publishes
  the quants and not its `imatrix_caesar.dat` — and a matrix made by another
  project's tooling is keyed by that project's tensor names. Its per-tensor
  recipe, measured by L2 loss, is the best starting evidence for presets.
- Untested on real files: scaled fp8, GGUF→GGUF requant, rules, Q4_K quality,
  cancel mid-dequantize, the F16-intermediate vs Q8_0 image comparison.
- The logs root is not overridable: conversion logs hit the real
  `~/.cache/drift/logs` even under isolated roots.
- Remembering the rules string per family (`settings/conversion.json`) waits
  for step 3.

## Post-v1

- Convert from the Configured tab and from the browsers ("download, then
  convert" as one queued pair); estimated time from the last measured MB/s;
  delete the source after a verified conversion (asked, never automatic).
