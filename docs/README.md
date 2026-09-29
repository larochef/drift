# drift — user guide

drift runs image and video diffusion models on your own machine and keeps the
work organised. It wraps two runners, [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
for the models and [llama.cpp](https://github.com/ggml-org/llama.cpp) for a
prompt assistant, installs them for your GPU, fetches the weights, and gives
you a browser UI on `http://localhost:4321`.

What it is for:

- **Projects** — one prompt worked over many attempts; every generation is a
  numbered version you can go back to, with an assistant that helps rewrite
  the prompt.
- **A gallery** of everything you ever made, with the exact recipe of each
  image, reusable on any model.
- **Post-processing** — upscale, diffusion upscale and redraw on any result,
  tiled so they fit the GPU.
- **Models done right** — architectures, models, run configurations and
  runtimes are separate things, so the same checkpoint runs on ROCm, Vulkan
  or CPU with the right flags without editing anything by hand.

No database: everything is JSON files and folders under your home directory,
see [Files and folders](files-and-folders.md).

## Pages

The sidebar, in the order the work happens:

| Page | What you do there | Guide |
|------|-------------------|-------|
| Projects | Work a prompt to a result, with versions and the assistant | [projects.md](projects.md), [assistant.md](assistant.md) |
| Sandbox | Try a model — image, video or text — with nothing kept unless saved | [generating.md](generating.md), [assistant.md](assistant.md) |
| Gallery | Browse, reuse, delete and post-process what drift made | [gallery.md](gallery.md) |
| Models | Architectures, models, run configurations; launch a session | [models.md](models.md), [generating.md](generating.md), [loras.md](loras.md) |
| Model Cache | Weights on disk, downloads, conversion, upscaler weights | [model-cache.md](model-cache.md), [browsers.md](browsers.md) |
| Settings | Runtimes (sd-cpp, llama.cpp, ROCm/Vulkan/CPU) and API tokens | [settings.md](settings.md) |

## Start here

1. [Getting started](getting-started.md) — from a fresh machine to a first image.
2. [Files and folders](files-and-folders.md) — where drift puts things and how to move them.
