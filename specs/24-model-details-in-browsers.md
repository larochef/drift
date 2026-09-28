# 24 — Model details in the browsers (model card, description, posted images)

**Status:** done
**Depends on:** 03

Know what a model is before installing it, without leaving drift: the
HuggingFace and Civitai browsers show a model's card or description and the
images posted with it, next to its file list.

## What it does

- Opening a result in either browser swaps to a detail view with tabs.
  **Files** is the default; the tab chosen last stays chosen for the next model
  while the browser is open. The detail view opens at its top, and **← Back**
  returns to the result list scrolled where it was left — a long browse never
  restarts from the first tile.
- The browsers open at one pinned width (80% of the window), the file browser
  of `03` included — a range let a short listing draw a narrower modal than a
  grid, and the head lost the room its source row needs. All three sites show
  the same tile: the preview filling it, how many examples there are as a chip in the
  top-left corner with the ✓ installed mark beside it, the media's own badges
  in the top-right (image or video, and a video's pause), and the name, its
  author or owner and its numbers over the gradient at the bottom. A
  repository's `owner/name` is split like Civitai's model and creator — name
  on the tile, owner as the byline, the whole id in the tooltip, where
  ModelScope also says when it files a repository as a LoRA. Five tiles to a
  row, so each has room.
- A video tile behaves the same on the three sites: the frame first, then the
  video, and **Play videos** in the modal's head turns them all off — one
  switch for the browsers and the galleries. Civitai's CDN makes a tile-sized
  copy and a still out of the URL, so its tiles never load an original; the
  repository sites serve the file as it is, so a tile there asks for its
  metadata only until it plays.
- HuggingFace: **Files · Model card**. The card is the repository's README,
  rendered. A gated repository whose terms are not accepted says so instead of
  erroring, and names Settings as where the token goes. The **Files** tab says
  it too, before a file is picked: a gate is a property of the repository, and
  a download of a gated one answers 403 until the token's account has been let
  in. `gated: "manual"` — the author admits people one by one — reads
  differently from `"auto"`, where accepting the terms is enough.
- Civitai: a version Civitai **charges for** carries a chip beside its name in
  the Files tab — "💲 paid", or "💲 early access until <date>" when the charge
  lapses. It is per version and not per model, since a model commonly publishes
  a paid version beside free ones (François, 2026-09-18); the model's own
  `hasActivePaidAccess` could only say that one of them is paid, without
  saying which. drift cannot buy one: the download works for an account that
  already owns it, and early access simply ends on its date.
- Civitai: **Files · About · Images**. About is the description, tags, and
  each version's notes and trigger words. Images shows one version at a time
  (newest by default): the author's own showcase first, then posts from
  Civitai's image search ten at a time, re-sortable, **Load More** for the
  next ten. A tile opens a viewer (full size, videos with controls) linking to
  the post on civitai.com. **Include NSFW** (off by default) and **Play
  videos** switches sit in the toolbar; off, tiles load stills only.
- Failures say why beside a **Retry** that re-sends exactly what failed; the
  cursor advances only on success. The opened repository has one of its own,
  above its tabs (`RepositoryDetails`, `RepositoryDetail`): its files and its
  pictures are a single request, and when that fails every tab is empty —
  which reads as an empty repository rather than a failure. ModelScope fails
  it often enough to matter ("获取模型目录树失败", François, 2026-09-22); the next
  attempt usually works.

## Shape

- Backend `routes/ModelDescriptions.scala` renders Markdown with commonmark
  (GFM tables, autolinks) and cleans everything with jsoup's relaxed safelist
  plus `align`, rules, `details` and `video`; relative URLs resolve against
  the repository (`…/resolve/main/`), links open outside drift. The page
  inserts the result only through `components/RichText.scala` — the one
  `innerHTML` in drift.
- Endpoints: `GET /api/hf-card/{owner}/{name}` (README, `gated` on 401/403,
  token from Settings), `GET /api/civitai-model/{id}` (detail incl. cleaned
  description), `GET /api/civitai-images?modelId=&modelVersionId=&sort=&cursor=&nsfw=`
  → `CivitaiImagePage(items, nextCursor)` in `shared/.../Civitai.scala`.
  Upstream failures relay the site's reason as a 502 (`upstreamFailure`).
- The search endpoint sends no descriptions: the grid never shows them and
  they are most of a page's bytes. `CivitaiModelListInfo.examples` turns the
  showcase Civitai ships with a result into `ModelExample`s — pictures first —
  so the Civitai tile shows and counts them like the other two.
- Frontend: `components/{CivitaiBrowser,HuggingFaceBrowser,
  CivitaiImageGallery,CivitaiMedia,CivitaiModelAbout,CivitaiTags,
  BrowserUtils,RetryNotice}.scala`, `services/{CivitaiService,
  HuggingFaceService}.scala`.
- The three browsers are the same two-phase component with site-specific
  parts, factored out (François, 2026-09-17). What they do the same way lives
  in the shell, so a change asked for once is a change everywhere (François,
  2026-09-18); a site browser holds only what its site does its own way.
  `components/ResultBrowser.scala` is that shell — modal and its width, error
  banner, **search field**, filter slot, **busy/empty placeholder**, card grid
  (`cardsPerRow`, five while the row is wide enough for them, as many 280px
  ones as fit below that), Load More, Back, Cancel — over one `SearchService`
  and values for what differs: `query`/`onSearch`/`searchHint`, the `filters`
  rows, the `emptyText` and what gates it, a `Signal` of built cards, a
  `detail: () => HtmlElement`. `BrowserView` is the
  state the two sides share: which view shows, plus the list's scroll,
  captured on `open()` (from a card's click) and restored one frame after
  `back()`, once the list is painted again. Then `BrowserCard` (the tile, taking
  what a result *is* — its previews, how many there are, name, author,
  downloads, the site's approval count and what it calls it — never markup, so
  the three cannot drift apart in how they draw one; counts go through
  `BrowserUtils.formatCount` and a count of zero is left out),
  `BrowserTabs` (the strip),
  `BrowserFilters` (the row of selects and tick boxes under the search field),
  `RepositoryDetail` (HuggingFace's and ModelScope's Examples · Files · Model
  card, over `RepositoryFiles` with its `BrowserFile(path, sizeBytes)` and
  `ModelCardTab` with its `ModelCardContent`: Pending / Absent / Gated /
  Rendered), `BrowserMedia` (one preview component for the three browsers and
  both galleries: candidates tried in order, a spinner, the badges, the
  play/pause and the `playVideos` switch, taking a site's small copies from the
  URL itself where its CDN makes them — `BrowserUtils.resizesOnUrl`), and
  `CivitaiVersionList` (Civitai's versions, their install
  buttons and their file rows, holding what is ticked and what was already
  sent to install). Each site browser keeps its own service, its filters, what
  goes on its tiles, and what it does with a chosen file.
- `services/SearchService.scala` is the same for the services: results,
  `searching`, `hasMore`, the command bus, `enterContext` and the paging
  dedupe (`appendPage`), leaving each service its client, its `Command` enum
  and its own paging — an offset, a page number, an opaque cursor.
  `RepositoryCards` adds the card state the two repository services share, and
  both of them hold the opened repository the same way, as a `detail` signal
  with its own `loadingDetail` — HuggingFace used to push an event its browser
  stored in two `Var`s, which is the same thing said differently.
- `components/RepositoryBrowser.scala` is the same idea one level down: the
  HuggingFace and ModelScope browsers stay two classes, each with its own
  client and service, and it holds everything they do identically — the shell
  and its title, the opening search, the tiles over a common
  `RepositoryResult`, opening a repository, the Examples · Files · Model card
  view, and the **With examples only** filter neither site can ask its server
  for. A site implements what it does its own way: how it sorts and filters,
  what its search command takes, how its results, its files and its card map
  onto those shapes. No type parameters — the shapes are values, as
  `BrowserModal` has it.
- `components/FileSelection.scala` holds what is ticked in a file list and
  builds what shows it: the `bar` over the list (`InstallBar`) and the `box`
  on a row. The Civitai version list, the repository file list and the disk
  browser all tick the same way because they all ask it (`33`).

## Notes

- The browsers' services are one per site and outlive a browser. Each
  remembers the context its results were searched for (what is browsed — a
  checkpoint, a LoRA, an upscaler —, the base models, the starting search): a
  browser opened in another context clears them before it renders, so the
  previous context's results never show while the first search runs.

- Civitai `/api/v1/images`: `nsfw` must be sent as a level (`X` = every
  level, `None` = safe only); absent it answers 503, `true` returns only NSFW
  posts. Ask by `modelVersionId`: by `modelId` alone the query takes 7–20 s,
  times out, and returns nothing for some LoRAs. The cursor is a string per
  version and a bare number per model; `CivitaiCursor` reads either.
- Every post's `meta` (prompt, parameters) comes back empty, so the viewer
  promises no recipe and links to the post instead.
- Civitai CDN renditions: image tiles as `width=450,optimized=true` (WebP); a
  video's still only from `anim=false` on its `.mp4` name; `width=1080`
  videos come back larger than the original, so the viewer plays the original.
  Tiles load lazily, a video only once its still is on screen. The CDN allows
  an hour of caching; drift asks one URL per rendition so the browser cache
  serves repeats.
- A gated card's relative images still fail: the page fetches them without
  the token.
- Laminar has no `controls` attribute; `VideoAttrs.controls` supplies it.

## Post-v1

- Blur NSFW tiles until clicked; fill the form from a post's recipe should
  `meta` return; proxy gated card images with the token; show the card for
  already registered models.
