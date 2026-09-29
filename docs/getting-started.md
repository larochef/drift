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

## 1. Install a runtime when asked

Where a model cannot start because nothing runs it yet, drift says *No image
runtime is installed.* (or *video*, or *text* for chat; *Choose the video
runtime for …* when one is installed but not one that model runs on) with a
choice of build
and an **Install** button. drift's own runner comes first where it can run the
model; sd-cpp or llama.cpp builds (ROCm if you have an AMD GPU, Vulkan, CPU)
follow. Picking a different engine than a configuration's switches that
configuration to it. The one you install is used from then on. Other
releases, or a build you compiled yourself, are in Settings → **Add a
runtime**; see [settings.md](settings.md).

## 2. Pick a model and download it

drift adds ready-made run configurations on first start: **Flux.2 Klein 9B**, **Krea 2
Turbo** and **Qwen Image 2.1** for images, **MiniMax H3** for video, **PiD 1.5**
to upscale, and **Qwen 3.6 35B-A3B** as a chat assistant. On Models → **Run
configurations**, press **Download the weights** on the one you want; the
button becomes **Launch** once the files are on disk. Progress shows in the
downloads panel at the bottom of the sidebar; its ✕ cancels a download
(it resumes where it stopped if you start it again).

Building your own configurations — other architectures, other files, LoRAs,
parameters — comes later; see [models.md](models.md) and
[browsers.md](browsers.md). Some HuggingFace repositories are gated and need a
token: Settings → Authentication.

## 3. Generate

Two ways:

- **The Sandbox**: pick Image, Video or Text, pick the model and generate.
  Nothing is kept unless you **Save** a result, into the gallery or straight
  into a project; the rest goes when you leave the page.
- **A project**: Projects → new project → open it, launch the configuration
  from the header, write a prompt, generate. Every generation is a version of
  the project. This is the intended way to work; see [projects.md](projects.md).

The form and its options are in [generating.md](generating.md).

## 4. From there

- Everything generated is in the [Gallery](gallery.md), with upscale, diffusion
  upscale and redraw one click away.
- Install a [turbo LoRA](loras.md) to make a slow model usable.
- Run a [prompt assistant](assistant.md) next to your project.
- If a model runs on the CPU although you have a GPU (an int8 checkpoint, for
  instance), [convert it](model-cache.md).
