# Bug 35 — sd-cpp crashes on LTX 2.5 (the official distilled BF16 files)

**Status:** open (upstream, sd-cpp; found 2026-09-30 while benchmarking the
drift runner, `specs/42` step 14)
**Severity:** high for sd-cpp users of LTX 2.5 (no video at all); the drift
runner runs the same configuration
**Files:** sd-cpp `ggml/src/ggml-cpu/binary-ops.cpp:135`; drift's seed
`backend/resources/reference/architectures.json` (`ltx-2.5`)

## Symptom

`sd-cli -M vid_gen` with drift's LTX 2.5 configuration (the official
`ltx-2.5-22b-distilled-transformer-bf16`, `ltx-2.5-video-vae-conv-bf16`,
`ltx-2.5-audio-vae-bf16`, `gemma4-12b-with-proj-ltx-2.5-bf16`; 512 × 512, 64
frames, 8 steps) encodes the prompt (14 s), loads the transformer, then
aborts on the first step:

```
ggml/src/ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types: dst: f32, src0: f32, src1: bf16
```

and hangs in its crash handler (ptrace denied) until killed. The same with and
without `--mmap`, on the installed builds master-929-3f8527a and
master-841-6b3edaa (ROCm). No LTX 2.5 run appears in drift's logs, so it may
never have worked here.

## What is known

An addition or multiplication whose second operand is a BF16 weight runs on
the CPU backend, which has no F32 + BF16 kernel. Which tensor it is was not
chased (not drift's code).

## Suggested fix

Report upstream with the command line. Meanwhile the drift runner draws LTX
2.5 from the same files (`specs/42` step 14).

## Verification

The command above finishes and writes a video.
