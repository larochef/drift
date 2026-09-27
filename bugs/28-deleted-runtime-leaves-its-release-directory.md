# 28 — Deleting a runtime can leave its release directory on disk

**Status:** open. Observed 2026-09-20 on the machine, not yet reproduced
deliberately
**Severity:** low — nothing breaks, but a release nothing references keeps
its gigabytes (1.3 GB in the observed case)

## What happens

Install an sd-cpp runtime on a specific (older) release, then delete it from
Settings → Runtimes. The entity goes
(`~/.config/drift/runtimes/<id>.json` is removed) but the unpacked release
stays under `~/.cache/drift/runtimes/sd-cpp/<tag>-<backend>/`.

Observed state on 2026-09-20, after an older sd-cpp release was installed to
check the version chip and then deleted:

```
~/.config/drift/runtimes/   latest-rocm, latest-vulkan,
                            llama-latest-rocm, llama-latest-vulkan
~/.cache/drift/runtimes/sd-cpp/
    master-881-17860c0-rocm     <- latest-rocm
    master-881-17860c0-vulkan   <- latest-vulkan
    master-841-6b3edaa-rocm     <- referenced by nothing, 1.3 GB
```

## Where to look

`RuntimeCleanup` (`backend/src/drift/backend/runtime/`) is what decides whether
a directory may go: it deletes one only when no other runtime references it,
which is right — a pinned and a `latest` runtime can share a release directory,
and two ROCm runtimes routinely share one TheRock build. The question is
whether the delete path calls it at all, and with what, against
`RuntimeManager.delete` / the `deleteRuntime` endpoint.

Worth checking too: the directory may instead be a leftover of an *upgrade*
(`cleanupSuperseded` runs only when the replacement validates, by design), in
which case the entry to fix is "an upgrade whose replacement failed validation
leaves the old directory behind for good".

## Verification

Install a runtime on a release no other runtime uses, delete it, and assert the
directory under `~/.cache/drift/runtimes/<tool>/` is gone. Then repeat with two
runtimes sharing one release directory and assert deleting one keeps it.
