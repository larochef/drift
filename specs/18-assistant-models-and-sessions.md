# 18 — Assistant models and sessions

**Status:** done
**Depends on:** 01–05 (architectures, models, downloads), 07 (sessions), 11 (status push), 17 (llama.cpp runtime)

A chat model is described the way an image model is: an architecture with
checkpoint slots, models filling them, a run configuration choosing them, a
session running `llama-server` on a drift-owned runtime. Nothing here knows
what a prompt is; [`20`](20-prompt-assistant-conversation.md) does.

## What it does

- An architecture carries a **tool** — sd-cpp or llama.cpp — chosen in the
  architecture form. It decides the executable, the listen flags, the
  readiness probe, the runtime default and which session cap counts.
- Two chat families are seeded: `qwen3.6-35b-a3b` (slots `model` `-m`,
  `mmproj` `--mmproj`; defaults `-c 131072 -ngl 99 --spec-type draft-mtp
  --spec-draft-n-max 2`) and `gemma-4-26b-a4b` (slots `model`, `mmproj`,
  `mtp` `--spec-draft-model`; defaults `-c 131072 -ngl 99`, no speculative
  flags). Models are registered from the HuggingFace browser like any GGUF.
- A chat configuration launches `llama-server` and goes ready when `/health`
  answers 200. It is stopped, reaped on crash and killed on shutdown exactly
  like an sd-server session.
- One assistant session may run beside one generation session.
- The frontend talks to the model through the backend only: properties (the
  applied context size, whether vision is on), a streamed chat, uploads.

## Shape

- `Architecture.tool: RuntimeTool` (`SdCpp | LlamaCpp`, required);
  `CheckpointRef.flag` names either tool's flag; `Session.tool`.
- `SessionSettings.maximumConcurrentAssistantSessions` (default 1) beside
  `maximumConcurrentSessions`; a launch counts only running sessions of its
  own tool. The runtime is resolved with the tool, so a chat configuration
  never launches on sd-server.
- argv: the checkpoint flags and parameters, `--host 127.0.0.1 --port N`, no
  `--lora-model-dir` / `--hires-upscalers-dir` (llama-server refuses them).
  Ports come from the shared range.
- `shared/Assistant.scala`: `ChatMessage(role, text, attachments)`,
  `ChatAttachment = OutputAttachment(date, fileName) | UploadAttachment(id)`
  (lowercase `type` discriminator), `ChatRequest(messages, maxTokens,
  temperature)`, `ChatEvent(content, reasoning, done, promptTokens,
  completionTokens, timings)`, `AssistantProperties(contextSize, vision,
  modelPath)`, `UploadedAttachment(id, name, mimeType, sizeBytes)`.
- Endpoints:

```
GET  /api/assistant/{sessionId}/properties  → AssistantProperties | 400 reason
POST /api/assistant/{sessionId}/chat        ChatRequest → one ChatEvent JSON per line
POST /api/assistant/uploads?name=&type=     bytes → UploadedAttachment
GET  /api/assistant/uploads/{id}            the bytes back
```

- `backend/assistant/AssistantProxy`: reads `/props` (context size,
  `modalities.vision`), forwards to `/v1/chat/completions` with `stream: true`
  and `stream_options.include_usage`, parses the SSE stream into events; the
  `done` event carries usage plus prefill and generation durations and rates
  from llama-server's final-chunk `timings`. Aborting the request cancels the
  generation.
- `backend/assistant/AssistantMedia`: resolves attachment references (outputs
  by date and file name, uploads under
  `~/.cache/drift/assistant-uploads/<id>-<name>`), scales images to at most
  1024 px and re-encodes as JPEG, passes videos as `video_url` parts. A missing
  file refuses the whole request by name before anything reaches the model.
- `routes/AssistantRoutes`: the chat is a `streamTextBody(OxStreams)`
  response over an ox `Flow`.
- Frontend: the architecture form's Runner select; the run configurations
  page filters cards and runtime selects by tool; the command-line preview
  names the tool's executable; the gallery's reuse offers only generation
  sessions.

## Notes

- **Attachments go by reference, chat over HTTP.** Netty caps a WebSocket
  frame at 64 KB and videos of tens of megabytes are attachments too, so no
  media bytes ever travel in a message; the backend reads the files.
- The video part convention (`video_url`) is unverified against a llama.cpp
  build with video input.
- jsoniter omits a field equal to its default on write, so the request shape
  sent to llama-server carries no defaults for `stream` / `stream_options`
  (a `stream: Boolean = true` would vanish).
- tapir needs explicit `Schema.derived` givens for the sealed `ChatAttachment`
  and every shape holding it.
- Qwen's MTP build does not support `--mmproj`: a vision configuration
  assigns the projector and overrides `--spec-type none`. Gemma 4 ships its
  MTP head as a separate GGUF, so a configuration wanting it assigns the `mtp`
  slot and adds the spec flags itself.
- Verbose reasoning is a launch-time lever: `--reasoning-budget N` or
  `--reasoning off` as a configuration parameter.
- llama-server flags in use: `--host/--port`, `-c`, `-ngl`, `--mmproj`,
  `--spec-type`, `--spec-draft-n-max`, `--spec-draft-model`; `--jinja` is on
  by default.

## Post-v1

- "Unload while generating" for machines where both models do not fit.
- A GGUF filter in the HuggingFace browser (repositories with an `mmproj`).
