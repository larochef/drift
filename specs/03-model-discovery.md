# 03 — Model discovery

**Status:** done
**Depends on:** 02

Find the file to register without leaving the app: one browser, opened by
**Add Model**, where the source is a row of buttons in its head — and what is
picked there *is* the model's source, never typed.

## What it does

- **Add Model** opens the browser at once (François, 2026-09-17: one button,
  not one per site). Its head — where a browser says what it is set to, beside
  **Play videos** — switches between Civitai, HuggingFace, ModelScope and this
  machine, each button carrying the site's mark; each site keeps its own
  search, filters and results. The search starts on the architecture's name
  where the site has no base-model filter for it.
- What comes back is a `ModelSource` with the fields that site needs, so the
  form shows the source read-only — site, repository or model, file, and the
  weight format read off the name — with **Change model** to pick another. The
  id and the label are suggested from the file and stay the user's to change;
  an id already typed survives a change of file. Parameters stay editable.
  Nothing types a repository or a file name any more: a repository the search
  cannot surface is out of reach, which is the trade taken.

- **HuggingFace** — search with a sort (most downloaded, most likes, recently
  updated, recently created) and **Load More** paging, then the chosen repository
  with **Files** and **Model card** tabs.
- **Civitai** — search filtered by the architecture's Civitai base models and a
  sort, cursor-paged with **Load More**, an **Include NSFW** checkbox (default on),
  cards with image and video previews; the chosen model shows **Files**, **About**
  and **Images** tabs, files grouped by version. The same browser installs LoRAs
  and upscalers in other modes ([`09`](09-lora-management.md), [`10`](10-generation-time-upscaling.md)).
- **Local** — server-side directory navigation with breadcrumbs, a live name
  filter and a models-only / all-files toggle.
- A failed request shows an error banner and the search stays usable.
- Details of an opened model (model card, description, posted images) are
  described in [`24`](24-model-details-in-browsers.md).

## Shape

- Endpoints, all under `/api` and proxied by the backend so tokens and CORS stay a
  server concern: `hf-search`, `hf-model/{owner}/{name}`, `hf-card/{owner}/{name}`,
  `civitai-search`, `civitai-model/{id}`, `civitai-images`, `files/home`,
  `files/list`, `cache/civitai-files`.
- Components: `components/SourceBrowser.scala` holds the source row and builds
  the site's browser under it (`sourceSwitch` is a mod the four browsers take
  and `ResultBrowser` renders above the search form), normalising what is
  picked into `(ModelSource, suggested label)` — plus `onCivitaiVersion` for
  LoRAs, whose unit is a version (`09`). `pages/architectures/ModelForm.scala`
  opens it, then shows the panel; `SourceTypeSelector` and its typed-in fields
  are gone. `ModelSource.kind`/`lines`/`fileName` (`shared/.../Api.scala`) are
  what the read-only panel and the model list show.
- What drift already has is marked in the browser, for a model as for a LoRA
  (`33`): a result tile whose Civitai model, HuggingFace or ModelScope
  repository is already registered carries a **✓ registered** chip naming it,
  as does the exact file row. `components/Installed.scala` holds the lookup for
  all three — `Installed.loras` (one architecture's, since another's is a
  separate entity), `Installed.models` (every registered model, whatever family
  it was registered in: the file is the same one) and `Installed.upscalers`
  (only the Civitai model is kept on an upscaler, so only its tile is marked) —
  and the word follows: a LoRA is *installed*, a model *registered*.
- `components/ProviderIcon.scala`: the site's initials in its colour ("C",
  "HF", "MS", a disk for this machine), beside a model's or a LoRA's name
  wherever one is listed — architecture card, missing models, the LoRA section
  and the picker — and larger on the source row. The sites' logos are images
  drift does not carry; the tooltip names the origin in full.
- The site browsers `{CivitaiBrowser,HuggingFaceBrowser,ModelScopeBrowser,
  FileBrowser}.scala` over `ResultBrowser`/`BrowserModal` (`24`); services
  `HuggingFaceService`, `CivitaiService`, `ModelScopeService`, `FileService`;
  backend `routes/{HuggingFaceRoutes,CivitaiRoutes,ModelScopeRoutes,FileRoutes}.scala`.
- Civitai calls carry the active Civitai token (`AuthTokens`), which is what makes
  account-gated models visible.

## Notes

- Civitai ignores `offset`; pages chain by `metadata.nextCursor`.
- Civitai's search parameter is `query`, and `nsfw` must always be sent
  explicitly: the API's absent-parameter default hides models the site shows,
  some of them SFW-flagged.
- A null stat in one Civitai result must not fail the page (bugs/20).
- Open low-severity bugs: 08 (label truncation on names with `/`), 12
  (HuggingFace query not URL-encoded), 13 (the browser does not re-search when
  the architecture's base models change: it receives a snapshot list, not the
  signal).
