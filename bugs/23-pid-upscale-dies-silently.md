# 23 — PiD upscale dies silently, apparently under VRAM pressure

**Status:** closed 2026-09-14, not a drift bug — François: the process died
for an external reason; no need to worry. Evidence kept below. Since
2026-09-13 a PiD job never decodes more than one 1280² tile per sd-cli run
(`specs/26-tiled-pid.md`), so the 14.9 GB allocation below is no longer made;
the runs that "saved" here were black frames anyway (flash attention,
`specs/15-post-hoc-resize.md`). The silent death itself is still unexplained.
**Reported by:** François (2026-09-10): "upscaling often fails"

## What happens

A post-hoc PiD upscale (`specs/15-post-hoc-resize.md`) ends with no output
image, no error in the job, and nothing in the log after the allocation line.
The job surfaces as `Failed` with sd-cli's tail attached, and the tail's last
line is the compute buffer being reserved.

## The evidence

Three PiD runs on disk under `~/.cache/drift/logs/postprocess-*.log`, all
`--steps 4` on the same PixelDiT decoder:

| job | target | compute buffer | outcome |
|---|---|---|---|
| `g1788644262241-1002` | 4096×4096 | 14856.78 MB | **no image, log stops here** |
| `g1788644972209-1001` | 4096×4096 | 14856.78 MB | saved |
| `g1788646391638-1001` | 4032×4032 | 13776.07 MB | saved |

The failing run and the second one are **the same size with the same buffer**,
so this is not a threshold that 4096² crosses — it is intermittent at a size
that also works. The failing log's last line is:

```
[DEBUG] ggml_extend.hpp:2241 - PiD compute buffer size: 14856.78 MB(VRAM)
```

and then nothing: no `save result image`, no `images saved`, no error. The
process is gone. That is the same shape as
[`17`](17-krea2-launch-segfault-no-output.md) — a native abort that prints
nothing.

## Leading hypothesis

**Memory pressure from a session that is still loaded.** A PiD upscale asks for
~14.9 GB of compute buffer plus ~2.6 GB of parameters, and drift runs it as a
separate `sd-cli` process while the generation session's `sd-server` may still
be holding its own model — the flux2 sessions in the same log directory report
`total params memory size = 33099.65MB (VRAM)`. `PostProcessManager` touches
`SessionManager` only to borrow `resolveArguments`; nothing checks what is
resident or waits for it, so whether an upscale fits depends on what happened
to be loaded when it ran. That fits an intermittent failure at a fixed size
better than anything in the parameters does.

## What would confirm it

- The wall-clock of a failing run against the session log for the same minutes:
  was a model loaded then, and not on the runs that succeeded?
- Re-running the failing job with no session live. If it succeeds reliably, the
  hypothesis holds.
- Whether the machine reports the allocation failing — the run prints nothing,
  so this may need `dmesg` or sd-cli's own stderr, which drift already captures
  since `specs/13-log-streaming.md`.

## Notes for the fix

Do not guess at a VRAM budget. If the cause is a resident session, the honest
options are to stop it for the duration, to refuse the upscale with a message
naming what is loaded, or to queue it — and which one is right is François's
call, since stopping a warm 33 GB load to upscale one image is a real cost.

`13` also means a failure like this now surfaces its last log line ahead of the
tail, so the next occurrence should say more than it used to.
