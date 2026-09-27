# 06 — Runtime management (sd-cpp; generalised to llama.cpp by 17)

**Status:** done
**Depends on:** 05 (downloads)

drift owns its runtimes: it fetches a chosen sd-cpp release, pairs a ROCm build
with a matching TheRock distribution, keeps several side by side, validates each
by running it, and picks a default. Owning the runtime is what lets a new model
family be tested the day sd-cpp supports it.

## A runtime is three things

A binary, the libraries it links, and the environment that lets it find them. The
ROCm build's `libggml-hip.so` needs HIP from a ROCm tree that is **not** in the
release archive; TheRock provides it as a portable per-GPU tarball. Launching
therefore composes `LD_LIBRARY_PATH` from TheRock's `lib` (ROCm only), the
executable's own directory and a sibling `lib` when one exists.

## What it does

Settings → **Runtimes** lists installed runtimes (tool tag, backend, release,
ROCm pairing, validity and reported version) and offers, per row, **Set default**
(one default per tool), **Upgrade** / **Check for update** on a latest-tracking
runtime, **Change ROCm**, **Revalidate** and delete. **Add a runtime** opens a
modal with two tabs:

- **Install a runtime**: tool, release (newest first, from GitHub) and build
  (ROCm / Vulkan / CPU); for ROCm a GPU target (default `gfx1151`) and a TheRock
  build select preselecting the version the asset declares, every tagged stable
  version offered. **Install latest** installs a runtime that tracks the newest
  release under a stable id.
- **Adopt an existing install**: register a directory already on disk; drift
  never deletes adopted files.

Install progress stays on the page (a job outlives the modal) and shows in the
global downloads panel. Two installs may run at once.

## Shape

`Runtime` in `shared/src/drift/shared/Runtimes.scala`:

```
id, label, tool, backend: Rocm | Vulkan | Cpu, releaseTag, rocmVersion?, gfxTarget?,
theRockVersion?, installedAt, theRockPath?, adopted, tracksLatest, valid,
validationError?, reportedVersion?, createdAt
```

`RuntimeSelection(defaultRuntimeId, defaultAssistantRuntimeId)`.

- Endpoints under `/api`: `runtimes` CRUD, `runtimes/{id}/validate`,
  `runtimes/{id}/upgrade` (`{theRockVersion?}`), `runtimes/{id}/therock`
  (`{theRockVersion}`), `runtime-selection`, `runtime-releases?tool=`,
  `therock/targets`, `therock/resolve?gfx=&rocm=`, `runtime-installs` (list,
  install, `latest`, `{runtimeId}/cancel`).
- Layout: `~/.cache/drift/runtimes/<sd-cpp|llama-cpp>/<tag>-<backend>/` and
  `…/runtimes/therock/<gfx>-<version>/`; archives under `runtimes/downloads/`.
  Ids: `<tag>-<backend>`, `latest-<backend>` (llama.cpp prefixes `llama-`).
- Executable discovery: `build/bin/`, then `bin/`, then the root (the sd-cpp ROCm
  zip unpacks to `build/bin/`; Vulkan and CPU archives unpack flat).
- Backend `backend/.../runtime/`: `RuntimeManager` facade over `RuntimeCatalog`
  (releases, TheRock indexes), `RuntimeInstalls`, `RuntimeArchives`,
  `TheRockPairing`, `RuntimeValidation`, `RuntimeSelections`, `RuntimeCleanup`.
  `resolveForLaunch(tool, pinnedId)` answers `LaunchRuntime(runtime, executable,
  environment)` or a named refusal; `sdCliOf` finds `sd-cli` beside `sd-server`.
- UI: `frontend/.../pages/settings/`.

## Notes

- **Sources**: sd-cpp releases are per-commit GitHub releases
  (`sd-<tag>-bin-Linux-Ubuntu-24.04-x86_64[-rocm-<v>|-vulkan].zip`), verified by
  the per-asset sha256 digest through the same `Downloader` as the weights.
  TheRock builds come from AMD's tagged tarball indexes
  (`repo.amd.com/rocm/tarball/`, `…/tarball-multi-arch/`,
  `stable.repo.amd.com/rocm/core/tarball/`; `DRIFT_THEROCK_STABLE_BASES`
  overrides the list), which publish no usable hash. The nightly bucket code
  remains but resolution does not draw on it.
- **Pairing rule**: the asset name carries the ROCm version it was built against;
  `TheRockPairing.resolve` proposes the tagged stable release matching it, then an
  on-disk match, else nothing, and the user may pick any stable version. A
  two-component declaration (`10.0`) matches the newest `10.0.x`
  (`RocmVersions.satisfies`). Whether a pairing is "declared" or "chosen" is
  derived, not stored.
- **Validation** runs the executable with its composed environment (`sd-server
  --help`; llama.cpp `--version` then `--help`) and keeps the banner's useful
  part (`commit 6b3edaa`, since release builds print "version unknown"). The
  banner is looked for anywhere in the output and behind a log prefix, not on
  the first line alone: llama.cpp b11060 logs `llama_server: initializing ...`
  in front of it, and the chip showed that log line (François, 2026-09-20). A
  line drift cannot parse is kept as printed, so a build older than the banner
  it knows reads as it always did. Valid
  means "loads its libraries", not "sees the GPU". An invalid runtime is kept and
  marked, and cannot be the default.
- **Shared directories**: a pinned and a latest runtime can point at one release
  directory, and two ROCm runtimes routinely share one TheRock build, so
  `RuntimeCleanup` deletes a directory only when no other runtime references it,
  and `RuntimeInstalls` holds a lock per destination directory across the
  "already unpacked?" check and the fetch.
- **Upgrade** of a latest runtime keeps the current pairing when the newest
  release declares the same ROCm version; otherwise the row shows the picker
  before upgrading. Re-pairing reuses the install job: the release download is
  skipped when the executable exists.
- Unpacking shells out to `unzip` / `tar` because `java.util.zip` drops the
  executable bit. Installs run on a pool sized `Downloader.MaxTransfersPerHost`
  and share the downloader's per-host permits.
- drift never builds sd-cpp; prebuilt releases cover the cadence.
