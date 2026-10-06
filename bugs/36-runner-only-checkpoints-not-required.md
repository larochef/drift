# Bug 36 — A slot the drift runner needs is optional for every runner

**Status:** fixed in code 2026-10-06 — `CheckpointRef.runners` (empty = every
runner); blockers, the argv and the configuration form read only the slots of
the configuration's runner; the H3 and Wan 2.2 A14B tokenizers are the drift
runner's and required there (migration 12 reseeds both); the architecture
editor sets it per slot. Unit-tested (`SlotsPerRunnerTests`), not seen in a
browser. Left as they were: the tokenizer slots of PiD and HiDream O1, required
on every runner
**Severity:** medium (a configuration that passes every launch check dies at
startup; the reverse case passes a flag sd-cpp may reject)
**Files:** `shared/src/drift/shared/Api.scala` (`CheckpointRef`),
`shared/src/drift/shared/CommandLine.scala` (`blockers`, the argument list),
`backend/resources/reference/architectures.json` (`minimax-h3`,
`wan-2.2-14B`), `runner/src/drift/runner/diffusion/VideoPipeline.scala`
(`open`, `needed`)

## Symptom

A MiniMax H3 configuration set to the drift runner, its `tokenizer` slot
empty, shows no blocker, launches, and the runner exits:

```
java.lang.IllegalArgumentException: minimax_h3_fl2va_pruned-Q4_K.gguf needs Qwen3's tokenizer.json: pass --tokenizer <file>
    at drift.runner.diffusion.VideoPipeline$.open(VideoPipeline.scala:175)
    at drift.runner.server.ImageMain$.main(ImageMain.scala:70)
```

Wan 2.2 A14B does the same without its `umt5-tokenizer`.

## Root cause

The architectures declare the tokenizer slots `"required": false`, which is
right for sd-cpp (it reads no tokenizer file) and wrong for the drift runner
(H3's text encoder GGUF carries no vocabulary, UMT5's file neither).
`CheckpointRef` has one `required` flag for all runners, and
`CommandLine.blockers` checks it without looking at the configuration's
runner. `CommandLine` builds the arguments the same way for every runner too,
so a tokenizer assigned to a configuration that runs on sd-cpp is passed to
sd-server as `--tokenizer`.

## Suggested fix

Let a slot say which runners it applies to and which require it (e.g. per
`RuntimeEngine`: the tokenizers required by `DriftRunner`, not passed to
`SdCpp`). `CommandLine.blockers` then reports an empty required slot for the
configuration's runner (and the launch→download swap of `specs/46` fetches
the seeded tokenizer), and the argument list leaves out the slots that do not
apply. An entity field: a migration, every architecture record declaring it,
the architecture editor showing it.

## Verification

- H3 on the drift runner with the tokenizer slot empty: a blocker naming the
  tokenizer (or its download), no launch.
- The same configuration switched to sd-cpp: no blocker, and the preview's
  command line has no `--tokenizer` even when one is assigned.
- Wan 2.2 A14B: the same two checks with `umt5-tokenizer`.
