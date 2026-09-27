# 20 — Prompt assistant conversation

**Status:** done in code, not yet run live with an assistant (compaction in
particular)
**Depends on:** 18 (a running assistant), 19 (a project to talk about), 21 (the chat components)

Inside a project, the user talks to the assistant about what to generate,
optionally shows it a result, and turns its suggestion into the form's
prompts with one click. The conversation is kept per project, compacted when
the context fills, and restartable. The loop:

```
version vN → generate → look → ask the assistant → proposal → apply → generate → vN+1 …
```

## What it does

- **One conversation per project**, loaded when the workspace opens and saved
  whole after every change. Free play's scratch chat is set aside meanwhile.
- **What the model is told**, per request: the editable system prompt, the
  project brief, the target architecture's prompting notes (a generic note
  when absent), the form's CFG (at CFG 1 the negative prompt has no effect),
  the **working prompt** as it stands in the form right now, then drift's
  rules — always last, so an edited system prompt cannot lose them: the
  user's prompt is the intent, an image only how one model rendered it;
  **edit by default**, rewrite only when asked and then keep every detail;
  never replace the prompt with a description of the image. Then the summary
  turn if any and every message after it.
- **Attaching**: Ask on any result stages a reference with a description
  split into *what I asked for* (prompt, negative) and *what came out*
  (parameters). Images are sent only when the session reports vision; the
  composer says so otherwise. Only the newest message carries images; earlier
  turns travel as text.
- **Proposals**: a finished reply whose last fenced `prompt` / `negative`
  blocks parse shows a card with the proposal as a word diff against the
  prompt the question was asked about; when fewer than half the words survive
  it opens on the plain text and lists the words not carried over. **Apply
  to form** fills the two prompts and nothing more; Generate stays the form's
  button.
- **Compare** in the versions column diffs any two versions' prompts.
- **Context meter**: the newest reply's prompt plus completion tokens over the
  applied context size. Past 75 % it says what to do and Compact is
  highlighted; at 100 % Send is refused.
- **Compact** streams a summary of the transcript into a `summary` turn; only
  a summary that comes back whole replaces anything: the transcript moves to
  the archive, and the summary rides in the system message of every later
  request. "Show archived transcript" expands what it stands for.
- **Restart** archives everything behind a confirmation naming the count.
  Versions and generations stay.
- **Retry** on the newest reply when it failed or was stopped; a request that
  fails mid-stream keeps the partial text.
- With no assistant live the column says to pick one in the header.

## Shape

- `shared/Projects.scala`: `Conversation(projectId, messages, archive,
  compactedAt)`; `ConversationMessage(id, role, text, reasoning, previews,
  proposal, promptBase, promptTokens, completionTokens, error, createdAt)`;
  `PromptProposal(prompt, negativePrompt)`.
- Stored as `~/.config/drift/conversations/<projectId>.json`, never beside
  the project: every file in `projects/` is decoded as a `Project`.
  `GET /api/projects/{id}/conversation` answers an empty one; `PUT` replaces
  it whole; deleting the project deletes it.
- `Architecture.promptingNotes: Option[String]`, seeded for the built-in
  image, video and upscale families in `architectures.json`.
- Frontend: `services/AssistantService` (`workingPrompt`, `send`, `retry`,
  `compact`, `restart`, `stop`; the bound project), `services/ChatStreaming`
  (one streaming path for message, retry and compaction),
  `services/ConversationStore` (load, serialized saves),
  `services/AssistantPrompts` (`DefaultSystemPrompt`, `PromptRules`,
  `GenericPromptingNote`, `CompactionPrompt`, the fence parser),
  `components/PromptDiff` (LCS over words; rewrite view; "not carried over"
  ignoring case, punctuation and filler), `pages/projects/VersionsColumn`
  (the compare modal). `GenerationPanel` publishes the working prompt, the
  target and the CFG and clears them on unmount.

## Notes

- **Saves are serialized and skipped after a failed load**, so an older
  snapshot never lands after a newer one and an empty transcript never
  overwrites a stored one. A reply still streaming for a transcript that has
  been swapped out is dropped.
- **Consecutive messages of one role are merged** before sending: Gemma's and
  Mistral's chat templates refuse two user turns in a row, which a failed
  reply or a compaction request would produce.
- Each assistant turn records `promptBase` at send time so its card diffs
  against what was asked about, not the live form (which after Apply would
  show no change).
- The fence match is forgiving: three or more backticks or tildes, the tag in
  any case, `positive` as a synonym, anything else on the fence line, and a
  block may run to the end of a cut-off reply. It runs only on finished turns.
- Versions stay out of the assistant: a version is made by a generation, and
  there is no "use as v(N+1)" that would create a version that never ran.

## Post-v1

- A second optional fenced block proposing steps, guidance or size.
- Several attachments in one message; automatic compaction at a threshold.
