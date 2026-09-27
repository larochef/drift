# 17 — Assistant runtime (llama.cpp)

**Status:** done
**Depends on:** 05 (downloads), 06 (the runtime manager this parameterises)

drift owns the LLM runner the way it owns sd-cpp: it never depends on an installed
`llama-server`, Ollama or another tool's schedule. A llama.cpp runtime is the same
three things an sd-cpp one is — binary, libraries, environment — and its ROCm build
has exactly sd-cpp's dependency on a TheRock tree, so this is not a second runtime
manager: it is [`06`](06-sdcpp-runtime.md) with the tool as a parameter.

## What it does

- Settings → Runtimes lists runtimes of both tools in one list, each row tagged
  with its tool; installing, adopting, validating, upgrading, re-pairing ROCm and
  deleting work identically for llama.cpp.
- One default per tool: the assistant default switches independently of the
  sd-cpp default. A launch of a chat architecture refuses a runtime of the other
  tool by name.
- Assistant sessions ([`18`](18-assistant-models-and-sessions.md)) launch
  `llama-server` with the same `LaunchRuntime(runtime, executable, environment)`
  the sd-cpp launcher uses.

## Shape

- `RuntimeTool { SdCpp, LlamaCpp }` on `Runtime`, on architectures, on release
  and install requests, and as `?tool=` on `GET /api/runtime-releases`. Per-tool
  knowledge lives in `RuntimeTool` (executable name, id prefix, display name) and
  `RuntimeCatalog` (repository, asset classifier):

| | sd-cpp | llama.cpp |
|---|---|---|
| repository | `leejet/stable-diffusion.cpp` | `ggml-org/llama.cpp` |
| Linux asset | `sd-<tag>-bin-Linux-Ubuntu-24.04-x86_64[-rocm-<v>\|-vulkan].zip` | `llama-b<n>-bin-ubuntu[-rocm-<v>\|-vulkan]-x64.tar.gz` |
| executable | `sd-server` | `llama-server` |
| validation | `--help` | `--version`, then `--help` |
| latest id | `latest-<backend>` | `llama-latest-<backend>` |
| unpack dir | `runtimes/sd-cpp/` | `runtimes/llama-cpp/` |

- `RuntimeSelection.defaultAssistantRuntimeId`; `RuntimeManager.resolveForLaunch(tool,
  pinnedId)`.
- Frontend: releases load for both tools at once; the install and adopt forms
  carry a tool select; the launch select on configuration cards filters by the
  architecture's tool.

## Notes

- llama.cpp declares its ROCm version with two components (`10.0`); the matcher
  takes that as "newest stable `10.0.x`", which is why `stable.repo.amd.com` is
  among the TheRock indexes (it carries `10.0.0`). sd-cpp declares three.
- The Vulkan and CPU tarballs unpack flat (`llama-server` beside `libllama.so`);
  the shared `build/bin` → `bin` → root probe finds it and the executable's own
  directory goes on `LD_LIBRARY_PATH`.
- A TheRock build is keyed by `<gfx>-<version>` and shared by both tools; the
  no-other-reference deletion rule covers runtimes of either tool.
- sttp's `uri"…"` interpolator percent-encodes a slash inside one interpolated
  value: interpolate `owner` and `name` as two segments.
- Validation never touches HIP, so a valid ROCm runtime may still fail at launch
  on an untested TheRock pairing; the Vulkan build is one install away.

## Post-v1

- A remote-endpoint pseudo-runtime (base URL + model name) with the same chat
  contract and no process.
