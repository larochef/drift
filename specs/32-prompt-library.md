# 32 — Prompt library

**Status:** done in code, not yet run live
**Depends on:** 20 (the assistant conversation and its system message), 21 (the
assistant panel), 27 (redraw), 19 (projects, for remembering a choice)

drift puts fixed text in front of models: the assistant's system prompt, the
rules appended to it, the redraw restoration prompt, the compaction prompt.
A user should be able to keep several variants of each, pick one where the
prompt is about to be used, and never lose drift's own version. The first
use is Ideogram 4: its open magic-prompt system prompt turns a plain idea into
the JSON caption the model needs, and a variant of the assistant's system
prompt is enough to get most of that help without anything Ideogram-specific
in drift.

## What it does

- A **prompt template** has a kind, a label and a text. Kinds:
  `assistant-system`, `redraw-restoration`, `edit` (39), `compaction`. Built-in templates
  come from drift's reference data, are re-seeded on every start and cannot be
  edited or deleted; **Duplicate** makes a user copy that can.
- A **Prompts** section on the Settings page lists templates by kind with
  edit, duplicate and delete. Editing is a textarea, nothing more.
- **Where a template is used, it is chosen**:
  - The assistant panel's system-prompt editor becomes a select of the
    `assistant-system` templates with the chosen text shown, read-only,
    under it. A project starts with what its image model suggests (the run
    configuration's override, else the architecture's default — the caption
    writer on Ideogram 4 — else drift's helper) and keeps its own pick once
    made; when a model starts whose suggestion differs from the project's
    pick, the panel offers **Switch** or **Keep** (a kept suggestion is not
    asked again until it changes). Free play follows the suggestion whenever
    the model changes and keeps a pick made after that.
  - Redraw's Advanced section gets a select of `redraw-restoration`
    templates beside the instructions field; the last choice is remembered.
  - Compaction uses the one `compaction` template marked as the project's
    choice, defaulting to the built-in; no picker in version 1.
- An `assistant-system` template says which of drift's additions follow it
  (all on for the built-in): the project brief, the target's prompting notes,
  the prompt being worked on, the editing rules, the CFG note. The Ideogram
  variant turns the editing rules off, since its job is to convert an idea,
  not to edit a prompt.
- A template says how a proposal is read from a reply: `fenced` (the
  `prompt` and `negative` blocks of today) or `json` (the last JSON object in
  the reply, fenced or bare, minified into the prompt; no negative). With
  `json`, a top-level `aspect_ratio` key is removed from the prompt and, when
  it parses as `W:H`, sets the form's width and height to the largest
  multiple-of-16 pair of that ratio within the current size.
- Two built-in `assistant-system` templates ship: drift's own, and "Ideogram
  4 caption writer" carrying Ideogram's `v1` magic prompt with a short
  preamble that supplies the aspect ratio from the form and asks for no
  reasoning. Three built-in restoration templates — `redraw-restoration`
  (the default), `redraw-skin-de-artifacting`, `redraw-women-portrait` (27
  says why), one edit template, `edit-seamless` (39), and one compaction
  template.

## Shape

- `shared/.../Prompts.scala`: `PromptTemplate(id, kind: PromptKind, label,
  text, appends: AssistantAppends, proposalFormat: ProposalFormat, builtIn)`;
  `AssistantAppends(brief, promptingNotes, workingPrompt, rules, cfgNote)`
  all `Boolean = true`; generic CRUD endpoints under `/api/prompt-templates`
  like the other entities (`RouteHelpers`): a create cannot claim `builtIn`,
  an update of a built-in answers with the stored copy, a delete of one is
  refused.
- Reference data `backend/resources/reference/prompt-templates.json`, seeded
  by `StorageService.init` through `seedFromResource`, stored under
  `~/.config/<app>/prompt-templates/`.
- `Architecture.assistantTemplateId: Option[String]`, edited on the
  architecture form ("Default assistant prompt") and seeded for Ideogram 4;
  `RunConfiguration.assistantTemplateId: Option[String]` overrides it
  ("Assistant prompt" on the configuration form and edit card).
- `AssistantService.suggestedTemplateId`: what the live image model suggests
  — the configuration's override, else its architecture's default, else
  drift's helper; nothing while no image model is up. `GenerationPanel`
  feeds `configurationTemplateId` beside `targetArchitecture`.
- `Project.assistantTemplateId: Option[String]` and
  `Project.compactionTemplateId: Option[String]`; a missing or deleted id
  falls back to the built-in. The workspace resolves the project's pick,
  else the suggestion, else drift's helper, and saves a pick in the panel
  back to the project only when it differs from that; free play may choose
  none, the raw model.
- `ConversationMessage.proposal`: the proposal is stored with the reply, so a
  reload shows the same card whatever template is chosen now; a transcript
  from before that is read as fenced blocks.
- `RedrawRequest.templateId: Option[String]`; the backend resolves the text
  and refuses a missing id loudly. `Redraw.RestorationPrompt` moves into the
  reference data.
- `AssistantPrompts.systemMessage` takes the template and honours its
  `appends`; `parseProposal(text, format)` dispatches on the format the
  reply was asked under. `AssistantService(library)` holds `templateId`,
  `compactionTemplateId` and `workingSize` (the form's size, for the aspect
  ratio); `AssistantAppends.aspectRatio` appends
  `TARGET IMAGE ASPECT RATIO: W:H` from it.
- `PromptProposal.aspectRatio`: `GenerationPanel.applyProposal` sets the
  form's size to it — the longer side stays, the other follows the ratio on
  multiples of 16.
- `frontend/.../services/PromptTemplateService.scala` (loaded with the app
  shell), `frontend/.../pages/settings/PromptsSection.scala` (show, duplicate,
  edit, delete; the appends and the proposal format are editable on assistant
  templates), `components/PromptTemplatePicker.scala` shared by the assistant
  panel and the redraw row, which reads the library off the assistant
  service.

## Notes

- The Ideogram magic prompt (`ideogram-oss/ideogram4`,
  `src/ideogram4/magic_prompt_system_prompts/v1.txt`, Apache 2.0) is about
  28 KB, seven thousand tokens. llama-server caches it across turns of one
  conversation; the context meter starts high and compaction must keep the
  system message, which it already does (the summary rides in it).
- The prompt was tuned on Claude Opus and Sonnet. A local model may drop
  parts of the contract or emit invalid JSON; the goal is most of the help,
  not all of it. The `json` reader takes the last `{ … }` that parses and
  otherwise treats the reply as discussion. Bounding boxes come through as
  the model wrote them; drift does not verify or strip them.
- The prompt expects "thinking off". drift's Qwen configuration folds
  reasoning already; the preamble asks for none, and a template cannot
  change launch flags.
- The user prompt the assistant receives is what it is today: the user's
  message, plus the attached image's description when there is one. The
  aspect ratio the Ideogram prompt wants rides in the preamble, from the
  form's width and height, not in the user's message.

## Remaining

- A live run with the Ideogram caption writer on a local model: how much of
  the contract a 35B mixture honours, and whether the preamble's "no thinking"
  is enough.
- Open: whether the description and the attached-image templates join the
  library (kinds `description`, `attachment`).

## Post-v1

- A picker for compaction.
- Import a template from a file or URL.
- Validate an Ideogram caption against the schema and show warnings on the
  proposal card, the way upstream's caption verifier does.
