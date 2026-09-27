# 36 — Examples in the HuggingFace browser

**Status:** done; checked against a test backend in chromium (Z-Image Turbo
and Qwen Image 2512 LoRAs), not yet used for a real install
**Depends on:** 24, 33

A LoRA is chosen by what it makes, which is why Civitai's browser shows images
first. HuggingFace repositories carry examples too, just not where drift
looked: a LoRA card lists them in its YAML header's `widget` (a file and the
prompt that made it), which HuggingFace renders where the README says
`<Gallery />`, and many repositories keep sample images as plain files. drift
now shows both, and the model card no longer loses its gallery.

## What it does

- Results are Civitai-style tiles: the first example fills the tile, the
  repository, author, downloads and likes overlay it, and a count names how
  many examples there are. A repository without any shows "no example"; an
  example whose file is missing (a card naming a file its repository lacks)
  shows "example not found".
- **With examples only** beside the sort order hides the results without
  examples, on by default when browsing for a LoRA. HuggingFace cannot filter
  on this, so a page shows fewer than it fetched.
- An opened repository has an **Examples** tab before Files and Model card:
  the card's gallery with its prompts, then the repository's other image and
  video files, as tiles; a tile opens a viewer with the prompt and a link to
  the repository. Browsing for a LoRA opens on it, and the tab stays chosen
  from one repository to the next unless the next has no examples.
- The model card renders `<Gallery />` as HuggingFace does: the card's
  examples as figures with their prompts.
- Browsing for a LoRA, the title reads "Browse HuggingFace LoRAs".

## Shape

- `ModelExample(url, video, prompt)`; `HuggingFaceModelInfo.preview` and
  `exampleCount`, `HuggingFaceModelDetail.examples` (`shared/.../Api.scala`),
  filled by drift.
- `backend/.../routes/HuggingFaceExamples.scala` reads the examples of a
  listing (`expand[]=cardData`, `expand[]=siblings`, with every other field
  the tiles use: an expanded listing carries only what it names), of a model's
  info, and a card's gallery. `widget` is lenient — a list, a single entry,
  entries that are not objects skipped — and an unreadable card costs its
  page the examples, never the listing, which is decoded apart.
- `ModelDescriptions.withGallery` replaces `<Gallery />` before rendering;
  the card endpoint fetches the model info for it only when the README has
  the tag. The cleaner allows `figure` and `figcaption`.
- Frontend: `components/ModelExampleGallery.scala` (tab, viewer, the
  tile media), `HuggingFaceBrowser` (`forLoras`, tiles, the checkbox, the
  tab), `HuggingFaceService.Event.FilesLoaded(files, examples)`.

## Notes

- HuggingFace serves no smaller copies: a tile loads the example itself,
  lazily once in sight, and a video tile loads its first frame only.
- Widget URLs are absolute or relative to the repository, encoded or not; file
  URLs are encoded per path segment. A card's `-` prompt means none.
- The first example is not always telling (a repository's logo can come first
  among its files).
