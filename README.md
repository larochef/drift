# drift

drift runs image and video diffusion models on your own machine and keeps the
work organised: projects with prompt versions, a gallery with every recipe,
tiled post-processing, and a local prompt assistant. It wraps
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and
[llama.cpp](https://github.com/ggml-org/llama.cpp), installs them for your GPU
(ROCm, Vulkan or CPU), fetches the weights, and serves a browser UI.

The user guide is in [`docs/`](docs/README.md); start with
[getting started](docs/getting-started.md). The features are described one per
file in [`specs/`](specs/README.md).

## Prerequisites

- A JDK, 25 or later (Ox runs on virtual threads)
- [Mill](https://mill-build.org) — the version is pinned in `.mill-version`; the
  `./mill` wrapper fetches it
- Node.js and npm, for the frontend build

## Quick start

```sh
./mill backend.run
```

This compiles the backend, builds the frontend and serves everything on
`http://localhost:4321`. The port is fixed, so only one drift runs at a time.

## Frontend development

For a faster loop, run the backend as above and, in two other terminals:

```sh
./mill -w frontend.fastLinkJS   # recompile Scala.js on change
npm run dev                     # Vite dev server with live reload
```

Vite serves the frontend on `http://localhost:5173` and proxies `/api` to the
backend on 4321, the status WebSocket included (`ws: true` — without it the app
loads but nothing updates by itself). The Scala.js output is resolved through the `scalajs:` alias in
`vite.config.js`.

## Building a distribution

```sh
./mill backend.universalStage
```

Runs the full Scala.js linker and produces a self-contained distribution under
`out/backend/universalStage.dest/`.

## Project structure

```
backend/     JVM server: tapir routes, sd-server/llama-server supervision,
             downloads, post-processing, conversion; reference seed data
             under backend/resources/reference/
frontend/    Scala.js SPA (Laminar): pages, components, services
shared/      Entities, JSON codecs and tapir endpoint definitions, cross-
             compiled for JVM and JS
docs/        User guide
specs/       One file per feature, describing it as built
bugs/        Bug tracker
build.mill   Mill build definition
```

## Stack

- **Scala 3**, built with **Mill**
- **Tapir** — HTTP API definitions shared between frontend and backend
- **jsoniter-scala** — JSON codecs, on the wire and on disk
- **Netty** (tapir's synchronous server) with **Ox** for structured concurrency
- **Laminar** and **frontroute** — the Scala.js frontend
- **Vite** — frontend dev server and bundler
- **commonmark** and **jsoup** — rendering model cards from the browsers

## Licence

Copyright 2026 François Laroche. Licensed under the
[Apache License, Version 2.0](LICENSE).
