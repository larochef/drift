# 24 — The generation detail is rebuilt on every status tick

**Status:** fixed in code 2026-09-14 with `specs/29` step 1 — awaiting
François's check with a session or a job running. The host builds the detail
through `split`, keyed on the generation and output index; everything that
changes comes in as `distinct` signals (the offer sees only the live
configuration's id), and the post-processing section is built once and hidden
for a video instead of rebuilt per output.
**Reported by:** François (2026-09-14): the detail view "gets very laggy when
the computer is busy, sometimes where there is an original / new version, it
feels like it want to go from one to another one, so we have an issue with
repaints I think"

## What happens

With a job or a session running, the gallery's full-size view — the image, the
compare toggle, the parameters, the upscale rows — lags, and an entry with an
original flickers between the result and the original.

## Cause

`GenerationDetailHost` renders the detail as

```scala
child <-- openGeneration
  .combineWith(
    labels,
    sessionService.sessions,
    runConfigurationService.runConfigurations,
    runConfigurationService.architectures,
    pool
  )
  .map { ... GenerationDetail(...).element ... }
```

so **every emission of any of those signals builds a new `GenerationDetail`**
and replaces the old one in the DOM. `sessionService.sessions` carries each
session's progress and activity line, pushed by the status socket several
times a second while a model loads or samples (`specs/13-log-streaming.md`);
`pool` changes whenever the history adopts a finished job. A busy machine
therefore rebuilds the whole view — decoding and laying out the images again —
many times a second.

Everything `GenerationDetail` keeps in a `Var` is reset by each rebuild:

- `showOriginal` falls back to `false`, so a view switched to *Original* jumps
  back to *Result* — the flicker François sees;
- `selectedIndex` returns to the first output of a batch;
- the upscale rows lose what is being typed: redraw instructions, strength,
  tile and reference size, the Advanced toggles, the PiD fields.

## Notes for the fix

Build the detail once per opened generation (keyed on the generation id and
output index, e.g. `split`/`distinct` on `openGeneration`), and hand it the
changing data as signals — sessions, configurations, derivatives, jobs —
instead of values captured at construction. The reuse offer, the live-session
notice and the chain are the parts that need them. The same audit applies to
any `child <--` that builds a large component from a frequently changing
signal (`specs/29-split-oversized-files.md`).
