# 21 — Assistant panel and scratch chat

**Status:** done
**Depends on:** 18 (a live assistant and its chat proxy), 08 (the generation form), 12 (the gallery)

The chat panel: the component that streams a reply, folds the reasoning,
stages images, shows a proposal card and a context meter. It serves free play
on the Models page and the project workspace ([`19`](19-projects-and-prompt-versions.md),
[`20`](20-prompt-assistant-conversation.md)).

## What it does

- **Where it runs.** The Models page's *Run configurations* tab lists sd-cpp
  and llama.cpp configurations together (the former `/inference` and
  `/assistant` pages, which still route there). Launching a chat
  configuration swaps the page to the assistant panel; Stop returns to the
  list. The tabs stay reachable while a session is live.
- **Free play** starts with **no** system prompt — the raw model answers —
  and its turns live in memory for the page's life; Clear chat empties them.
  A workspace's panel carries the project's kept conversation instead.
- **Header**: the configuration, the session status, and what the server
  applied once ready — the context size and `vision` or `text only` — with
  Stop session and a fold for the system prompt (an editable textarea; the
  default asks for fenced `prompt` and `negative` blocks) and the
  conversation actions (Compact, Restart in a workspace; Clear chat in free
  play). In a workspace there is no header: the model bar carries the
  picker, status, Stop session and the applied facts
  (`AssistantSessionFacts`) beside the assistant, as it does the image
  model's controls.
- **Turns**, composer on top, newest first: the reply being written appears
  right under the input. User turns show their images and text; assistant
  turns stream, reasoning folded into "thinking… (n characters)". Under a
  finished reply: prefill and generation as "n tokens in s (tok/s)".
- **Proposal card** on a reply ending with the fenced blocks: **Apply to
  form** in a workspace, **Apply to generation form** in free play (hands the
  prompts to the generation service; the next generation panel to seed takes
  them over reuse and defaults; with no image model live they wait for the
  next launch).
- **Context meter**: prompt tokens over the applied context size, orange
  from 75 %, red from 95 %.
- **Composer**: a textarea (Ctrl+Enter sends), staged images each with a
  remove button, a file input for any other image or video, Send or Stop
  (aborts the request, which cancels the generation server-side).
- **Sending an image**: **Ask the assistant** on a completed image in the
  generation panel and in the gallery detail stages a reference to the
  output plus a text description (configuration, size, steps, sampler, seed,
  prompt, negative). On the gallery it then navigates to the assistant; in a
  workspace it fills the column beside it. The file input uploads once over
  HTTP and stages the returned id; the preview is the backend serving it
  back. Staged files go with the next message only.

## Shape

- `frontend/pages/assistant/AssistantPanel` (constructor takes
  `onApplyProposal`; absent in free play), `frontend/services/AssistantService`
  (turns, staged attachments with preview and description, properties, one
  streamed `POST` per reply read with `fetch` and a `ReadableStream` reader
  through a small `TextDecoder` facade; `stop` aborts), `services/ChatStreaming`,
  `services/AssistantPrompts` (`parseProposal`, `describe`).
- `pages/models/ModelsPage` hosts `pages/inference/RunConfigurationsPage`
  with `tools = List(SdCpp, LlamaCpp)` and `pages/architectures/ArchitecturesPage`
  as tabs, both built once and kept.
- The gallery detail takes an `onAskAssistant` callback beside its reuse
  actions.
- No backend of its own: 18's proxy, properties and upload endpoints carry
  it all.

## Notes

- **The composer is on top and turns run newest first**; a long chat never
  needs scrolling to the bottom. Keep it that way in any conversation view.
- **Bulma's `.content pre` out-specifies a bare class** (`white-space: pre;
  word-wrap: normal`), stretching the page to the longest line and pushing
  the header's controls off screen. The rule is `.content pre.assistant-text`
  with `overflow-wrap: anywhere`. `.panel-header` wraps its two groups
  (Bulma's `.level` does not) and is `position: sticky`.
- The system prompt is set on mount per panel: empty in free play, the
  default in a workspace.
- A full page load empties free play's in-memory chat; in-app navigation is
  pushState and keeps it.

## Post-v1

- Markdown in replies (turn text is a `pre`); the reasoning text streaming
  inside its fold; a video's first frame as the attachment.
