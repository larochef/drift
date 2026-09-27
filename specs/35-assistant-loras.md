# 35 — LoRAs on chat models

**Status:** done in code, not yet run against a real llama-server
**Depends on:** 09, 18, 28, 33

A chat architecture had a LoRA section and nothing behind it: its LoRAs were
installed and never reached llama-server. llama-server takes GGUF adapters on
its command line, lists them at `/lora-adapters`, and applies them per request
(`lora: [{id, scale}]` on a chat completion). drift now launches a chat model
with its architecture's LoRAs loaded but unapplied, and applies the run
configuration's default LoRAs to every assistant reply.

## What it does

- A llama.cpp launch gets `--lora <file>` for every GGUF file of the
  architecture's installed LoRAs that is on disk, then
  `--lora-init-without-apply`, so nothing applies unless a request asks. The
  command-line preview shows the same flags.
- The run configuration form and edit card offer "Default LoRAs" for chat
  architectures too (the assistant-prompt preference stays on image and video
  configurations).
- Every assistant request of a session applies its configuration's default
  LoRAs at their strengths, matched to the server's adapters by path. A
  default LoRA the server did not load — installed after the launch, or with
  no GGUF file — refuses the request by name ("restart the session"), as a
  tiled job refuses a missing default LoRA.
- Installing a LoRA for a chat architecture accepts `.gguf` files only, from
  every source; a safetensors adapter is refused with the conversion to use
  (llama.cpp's `convert_lora_to_gguf.py`). A Civitai version keeps only its
  GGUF files.
- On the card, a chat LoRA's file chip names the file; there is no stage to
  cycle.
- Civitai does not appear for chat architectures: no **+ Civitai** LoRA button
  (not even greyed out), no Civitai base models on the card or in the
  architecture form (saved empty for a llama.cpp architecture), and the backend
  refuses a Civitai LoRA install for one.

## Shape

- `LoraAdapters` (`shared/.../Loras.scala`): `isAdapterFile`,
  `relativePaths(architectureId, loras)`. `CommandLine.argv`/`resolve` take
  `loraAdapters`; `LaunchArguments` passes the absolute, existing files,
  `RunConfigurationsPage` the `~`-forms.
- `AssistantProxy` (now given the storage and the LoRA root): `adaptersFor`
  reads the session's configuration and `GET /lora-adapters`, and the request
  body carries `lora`.
- `LoraManager.refusalOf` gates installs by the architecture's tool.

## Notes

- All installed adapters load, not only the defaults: changing a
  configuration's defaults then applies from the next reply without a restart,
  like sd-cpp's scanned LoRA folder. The cost is the adapters' memory; a chat
  LoRA of a large MoE model can weigh gigabytes.
- llama-server does not batch requests with different LoRA sets together.
- Most chat LoRAs on HuggingFace are PEFT safetensors; they need converting
  against their base model before drift can use them.

## Post-v1

- A LoRA choice in the assistant panel, per conversation, beside the
  configuration's defaults.
- Converting a PEFT adapter from inside drift.
