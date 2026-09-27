# 41 — Text projects

**Status:** done in code, not yet run live
**Depends on:** 18 (chat sessions), 19 (projects), 20 (the kept conversation, compaction, restart), 21 (the chat panel), 31 (project kinds)

A text project is a kept conversation with a chat model, and nothing else:
no image model, no versions, no prompt assistant. It is a third project
kind beside images and videos. The model gets the conversation as written,
with nothing added by drift.

## What it does

- The new project form's "Makes" choice offers Texts. The workspace header's
  kind select offers Text project. The tile shows 💬, "a conversation" in
  place of a cover, and no version count.
- **Workspace**: the header has one picker, **Chat model**, over the
  llama.cpp configurations. The page is the chat panel alone, in one centred
  column: no versions column, no generation panel, no results.
- **Raw model**: no system prompt. There is no template, brief, prompting
  notes or rules. The system prompt fold says so instead of offering a
  picker, and no template switch is offered. After a compaction, the summary
  is the only system message.
- **No proposals**: no reply is read as a prompt proposal, so there are no
  cards and no Apply.
- Everything else is the project conversation of
  [`20`](20-prompt-assistant-conversation.md): saved after every change,
  Compact, Restart, Retry, the context meter, and image or video uploads
  when the session has vision. There is no Ask from results, because a text
  project has none. Ask from the gallery still stages into whatever
  conversation is bound.
- Changing a project's kind to or from Text keeps its conversation and
  versions. Only the layout and what the model is told change.

## Shape

- `ProjectKind.Text` in `shared/Projects.scala`; `accepts` takes an
  architecture tagged `llm`. The chat picker still lists every llama.cpp
  configuration, like the assistant picker.
- `AssistantService.rawModel: Var[Boolean]`, set by the workspace from the
  kind and reset on unbind. While it is set, `systemMessages` gives only
  `AssistantPrompts.summaryMessage` (nothing before a compaction), and
  `proposalFormat` is `None`, which `stream` reads as "parse no proposal".
  Compaction never parses one either.
- `ProjectWorkspacePage` switches the columns on the kind
  (`.workspace-columns.is-text`); `WorkspaceHeader` switches the pickers;
  `AssistantPanel` hides the template picker and switch offer.
- No backend change: the conversation is stored and served as in `20`, and
  the cover endpoint finds no `text/` output and answers 404.

## Notes

- Enum values are stored by name, so adding `Text` needs no migration. An
  older build reading a text project fails to decode it and skips it with a
  warning.

## Post-v1

- Markdown in replies (turns are `pre`).
- Several conversations per project.
- A per-project system prompt, if raw ever falls short.
- The newest reply's first line as the tile's cover text.
