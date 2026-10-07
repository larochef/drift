# Bug 50 — Flux.2 Klein with a 2048 × 2048 reference fails on the drift runner

**Status:** open (found 2026-10-07, the edit study; not looked into)
**Severity:** low to medium (an edit pass larger than 1536 is not something drift asks for today; it would let a
4096² window be edited at half size instead of a quarter)
**Files:** `runner/src/drift/runner/diffusion/Flux2Pipeline.scala` and what it calls

## What happened

Klein 9B (SNOFS) asked for a 2048 × 2048 picture with one 2048 × 2048 reference image, 4 steps:

```
job failed: {'code': 'generation_failed', 'message': 'requirement failed: 83968 ints at once'}
```

The same call at 1536 × 1536 runs in 100 s, at 1024 × 1024 in 55 s. 83968 is not the token count of one
picture (16384 at 2048²): a buffer of ints — positions, by the look of it — outgrows a limit somewhere.

## Why it matters

A change that runs across a whole 4096² picture is edited at 1024² and carried up ×4 (`EditRequest.passOf`):
measured the same night, a face redrawn at that size and brought back ×4 by SeedVR2 comes out harsher than
one edited at 1536 and carried ×2. A 2048 pass would keep such a picture at ×2.
