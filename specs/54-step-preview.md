# 54 — Step preview: the picture while it is sampled

**Status:** done in code; previews read from the runner's job endpoint on FLUX.2 [klein], Krea 2, Qwen Image 2.1, Mage-Flow and HiDream O1; seen in the Sandbox on an isolated drift (HiDream O1, a batch of two on Qwen Image 2.1, and sd-cpp's own on ERNIE-Image)
**Depends on:** 08 (the inference UI), 11 (the status socket), 42 (the drift runner), 43 (a runner per configuration), 44 (generation progress)

A picture takes seconds to minutes, and the bar says only how far it has
got. While an image is sampled on the drift runner, drift also shows what
the model has so far: its estimate of the finished picture after each step,
under the progress bar. A wrong composition is seen at the second step
instead of the last.

## What it does

- **Under the bar of a running image generation**, a small picture: the
  model's estimate of the result after the last step done, captioned
  "Preview after step 3 of 8". It is redrawn in place at each step and leaves
  when the result (or a failure, or a cancellation) takes the panel.
- It is an estimate, not a decode: soft, a quarter to an eighth of the
  picture's size enlarged by the browser, rough at the first steps and close
  to the result at the last.
- **A batch** previews the image being made; the preview empties when the
  next image starts.
- **A generation queued** behind another shows no preview: the picture is the
  running job's, and the panel's bars already say whose they are.
- Families with a preview: FLUX.2 ([klein] and [dev]), Krea 2, Qwen Image
  2.1, Mage-Flow, Nucleus-Image, HiDream O1. The others (PiD, LLaDA-Image,
  GRN) and every video have none, and their panel is as before.
- **sd-cpp has one too**, its own: `sd-server` draws it for a request that
  asks (`preview`: `proj`, `tae` or `vae`), and drift asks every image for
  `proj`, the latent projection. The same field asks the drift runner, and
  both answer the same way, so the page does not know which engine it is
  watching. A build of sd-cpp older than its preview ignores the field.
- Nothing is stored: a preview lives in memory from the runner to the page
  and is gone with the generation's end. It is never a gallery entry.
- No setting: the preview is always on, and costs a step nothing worth one
  (Notes).

## Shape

- **The estimate** (`runner/.../diffusion`). `ImagePipeline.preview` is a
  hook the server sets around `generate`, as a video pipeline's `stage` and
  `withinStep` are: `preview(step, steps, picture)`, called in each step
  before it is counted, `picture` a function drawn only if called. A
  velocity model's estimate is `x − σ × v`, on the latents the step started
  from and the (guided) velocity it found; at the last step it is the
  latents that are decoded. HiDream O1 predicts the clean picture itself, in
  pixels.
- **From latents to a picture** (`LatentProjection`). No VAE: each token, as
  the transformer takes it (packed, normalized), gives `block × block` RGB
  pixels by one matrix and a bias, `[block² × 3, channels + 1]`. A 1024²
  picture is 64 × 64 tokens in every family, so its preview is 256² (Wan
  2.1's VAE, block 4) or 128² (the others, block 2). The matrices are
  resources, `runner/resources/previews/<name>.txt`:

  | name | VAE | tokens | block | families |
  |---|---|---|---|---|
  | `wan-2.1` | Wan 2.1, packed 2 × 2 | 64 | 4 | Krea 2, Nucleus-Image |
  | `flux.2` | FLUX.2, packed 2 × 2 | 128 | 2 | FLUX.2 [klein], [dev] |
  | `qwen-image-2.1` | Qwen Image 2.1 | 64 | 2 | Qwen Image 2.1 |
  | `mage-flow` | Mage-Flow | 128 | 2 | Mage-Flow |

  `LatentProjection.named(name, channels)` gives none for a file that is
  missing or of another width, and the family then draws no preview.
- **HiDream O1** (`HiDreamO1Pipeline.reduced`): the estimate's patches
  averaged 8 × 8 pixels to one, 128² for a 1024² picture.
- **The fit** (`runner/gpuTest/.../PreviewProjectionFit.scala`, a main):
  encodes pictures with the family's VAE alone, as its pipeline does,
  averages each picture to `block` pixels a token, and solves the least
  squares (`LatentProjection.fit`, the normal equations with a small ridge).
  Each picture is encoded a second time with its colour channels swapped and
  every other one in negative, for the colours the pictures lack. It prints
  the error on the pictures held out and the time a preview takes, and
  writes the resource.
- **The request and the job document are sd-server's** (its
  `examples/server/api.md`): an `img_gen` request's `preview` names a mode,
  anything but `none` asking for previews, and `GET /sdcpp/v1/jobs/{id}`
  then carries `preview` beside the usual fields once a step has run:
  `{pass, step, total_steps, b64_json}`, `pass` the sampling pass from 1 (an
  image of the batch; on sd-cpp a hires pass too). The runner (`ImageServer`)
  takes any mode for its one way of drawing, sends a JPEG where sd-server
  sends a PNG, and lists `preview` among its image features and
  `preview_modes`. A job whose request did not ask draws none.
- **The backend.** `GenerationSubmissions` sends every image with
  `preview: "proj"` (`ImageGenerationParameters.preview`, on what is sent and
  never on what is recorded). `GenerationMonitor` keeps a poll's `preview`
  on the generation's entry (`GenerationEntry.preview`, a `StepPreview`: the
  image of the batch from 0, the picture's type told from how its base64
  starts), emptied when the job ends. `GenerationManager.previews` is every
  entry's, by generation id.
- **The status socket** (11) has a topic for them, `StatusUpdate.previews`:
  a snapshot of the map, sent when it changes, so once a step. It is not a
  field of `Generation` (stored in the sidecar, and resent with the
  session's whole list) nor of `Session` (resent with each progress
  redraw).
- **The page.** `GenerationService.previews` mirrors the topic;
  `StepPreviewView` (`frontend/.../pages/generate`) shows the entry of the
  generation on screen, in `GenerationResult` under `LogProgressView`: one
  `<img>` on a data URL, enlarged by CSS (`.step-preview`). The project page
  and the Sandbox share it through the generation panel.

## Notes

- **Cost.** Reading the two tensors back, drawing the picture and writing
  its JPEG takes 9 ms (128 values a token, block 2) to 18 ms (Wan 2.1,
  block 4) on the grid of a 1024² picture, measured alone and warm
  (`PreviewProjectionFit`). Measured on whole generations at 1024², the same
  seed with and without previews, a step is: FLUX.2 [klein] 4.95 s either
  way; HiDream O1 1.023 s without, 1.027 s with (+0.4 %); Mage-Flow 2.13 s
  without, 2.15 s with (+1 %). Krea 2 and Qwen Image 2.1 were measured on a
  GPU shared with other work, where the difference is lost in the noise.
- **At most one preview a second**, and the last step's always
  (`ImageServer`): the backend polls once a second, and a model whose steps
  are shorter would draw pictures nobody sees. This also bounds the cost at
  about 2 % of the time whatever the model. A job whose request does not
  ask (a redraw's tiles, another client) draws none.
- **A step can be missed** by the page: the poll is a second apart, and the
  last step's preview lives only until the job ends, which is at once for a
  model without a VAE to decode (HiDream O1).
- **Sizes.** For a 1024² picture a preview's JPEG is 3 to 8 KB at 128²
  (Qwen Image 2.1, FLUX.2, Mage-Flow, HiDream O1) and 6 to 17 KB at 256²
  (Krea 2), growing with the detail as the steps go; base64 on the wire, a
  third more.
- **Quality.** The projection is the best linear reading of one token.
  Wan 2.1 latents read well (error 0.10 RMS on colours in [−1, 1], on
  pictures held out, at block 4): Krea 2's last preview is the picture, small.
  Qwen Image 2.1 reads well too (0.09), a little washed out, with a faint
  grid of tokens. FLUX.2 and Mage-Flow read worse (0.16): the composition
  and the tones are right, saturated colours come out dull and the token
  grid shows as speckle. A larger block does not help them: 4 pixels a side
  shows the grid more. HiDream O1's preview is the picture itself, reduced.
- **The fit's pictures** are test pictures made for it (portraits, a room,
  a terrace): few, and mostly people, which the recolouring only partly
  makes up for.
- **sd-cpp.** Its server's preview is a callback of the library
  (`sd_set_preview_callback`), set for the job that asks and published in
  its document; `preview_interval` (1 by default) spaces the frames. Seen
  from drift on `master-948`: the previews arrive, further apart than the
  steps when the GPU is shared, since the server answers a poll between two
  of its own steps. Videos take the same field there; drift does not ask.
- **The estimate is taken before the step moves the latents**, where `x`
  and the velocity are both whole: after it, the velocity is scaled in
  place. HiDream O1's is taken before its sampler reuses the tensor.
- **Inpainting** (a mask) previews the model's estimate of the whole
  picture, the kept part included, before it is restored.

## Remaining

- Not looked at in a page: the project page, a tall picture, a generation
  queued behind another, Cancel.
- FLUX.2 and Mage-Flow deserve a better reading than one matrix a token: a
  small decoder (a tiny autoencoder, as sd-cpp's `--preview tae`), or a
  projection over a token's neighbours, within the same cost.
- Redraw and edit tiles (`NativeJobs`, post-processing) run without a
  preview.
- Videos: a frame or a strip of the estimate.
- sd-cpp's better modes (`tae`, `vae`) as a choice, and its video previews.
