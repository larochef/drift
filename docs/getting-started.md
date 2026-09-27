# Getting started

From a fresh machine to a first image.

## Run drift

You need a JDK, [Mill](https://mill-build.org) and Node.js. From the repository:

```sh
./mill backend.run
```

Open `http://localhost:4321`. Only one drift can run at a time: the port is fixed.

If you leave a tab open while you restart drift on a new build, a notice at
the top of the page offers to reload it — a page older than the backend cannot
read what it sends, and nothing on it updates by itself until you do.

## 1. Install a runtime

Settings → **Add a runtime**. Pick the tool (sd-cpp for images and videos,
llama.cpp for the assistant) and a backend:

- **ROCm** for AMD GPUs — drift downloads the release and the matching ROCm
  (TheRock) build and pairs them.
- **Vulkan** for NVIDIA and other GPUs.
- **CPU** as a fallback.

The install shows its progress on the page. Once it is valid, make it the
default runtime for that tool. The **Adopt** tab registers a build you
compiled yourself instead. Details in [settings.md](settings.md).

## 2. Get a model

Models → the architectures come seeded, each with reference models filling its
checkpoint slots. Two ways to get weights:

- Model Cache → **Not downloaded** lists the seeded models; download one
  (Flux.2 Klein 4B is a good, fast first choice).
- Or search HuggingFace or Civitai from an architecture card and install a
  model from there, see [browsers.md](browsers.md).

Some HuggingFace repositories are gated and need a token: Settings →
Authentication. Progress shows in the downloads panel at the bottom of the
sidebar.

## 3. Make a run configuration

Models → **New run configuration**: pick an architecture and a model for each slot,
keep the defaults, save. The launch preview shows the exact command line drift
will run and notes anything the runtime vetoed. See [models.md](models.md).

## 4. Generate

Two ways:

- **Free play**: on the Models page, launch the configuration and generate
  right there. Results are scratch: gone on restart unless you **Keep** them,
  into the gallery or straight into a project.
- **A project**: Projects → new project → open it, launch the configuration
  from the header, write a prompt, generate. Every generation is a version of
  the project. This is the intended way to work; see [projects.md](projects.md).

The form and its options are in [generating.md](generating.md).

## 5. From there

- Everything generated is in the [Gallery](gallery.md), with upscale, diffusion
  upscale and redraw one click away.
- Install a [turbo LoRA](loras.md) to make a slow model usable.
- Run a [prompt assistant](assistant.md) next to your project.
- If a model runs on the CPU although you have a GPU (an int8 checkpoint, for
  instance), [convert it](model-cache.md).
