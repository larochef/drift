# 54 — Step preview: the picture while it is sampled

**Status:** done; pictures and videos seen in the page, with the checkbox and a video's Pause
**Depends on:** 08 (the inference UI), 11 (the status socket), 42 (the drift runner), 43 (a runner per configuration), 44 (generation progress)

A picture takes seconds to minutes, a video longer, and the bar says only
how far it has got. While an image or a video is sampled on the drift
runner, drift also shows what the model has so far: its estimate of the
finished picture after each step, under the progress bar — a video's as a
small clip that plays. A wrong composition is seen at the second step
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
- **A video** previews as one picture that moves: the estimate of every
  latent frame, played in a loop at the video's own pace, silent, with the
  same caption. **⏸ Pause** beside the caption holds it on the frame it is
  on, and each later step then shows as its first frame, still, until
  **▶ Play**; paused once, the next videos' previews come paused too, for as
  long as the page is open. It is coarser in time than the video (a picture for every 4
  frames on Wan, 8 on LTX, about 3.4 on MiniMax H3) and small (a quarter to an
  eighth of the video's size). A two-expert model (Wan 2.2 A14B) counts the
  steps of the expert under way, and keeps the high-noise expert's last
  estimate on screen until the low-noise one has its first. The preview
  leaves when the sampling ends: the decoding shows none.
- Families with a preview: FLUX.2 ([klein] and [dev]), Krea 2, Qwen Image
  2.1, Mage-Flow, Nucleus-Image, HiDream O1; the videos Wan 2.2 A14B, LTX
  2.5 and MiniMax H3. The others (PiD, LLaDA-Image, GRN) have none, and
  their panel is as before.
- **sd-cpp has one too**, its own: `sd-server` draws it for a request that
  asks (`preview`: `proj`, `tae` or `vae`), and drift asks for `proj`, the
  latent projection. The same field asks the drift runner, and both answer
  the same way, so the page does not know which engine it is watching. A
  video's on sd-cpp is a still of its first frame. A build of sd-cpp older
  than its preview ignores the field.
- **Preview while generating**, a checkbox over the results of the
  generation panel (the project page and the Sandbox), ticked at first, for
  images and videos alike. Unticked, the generations submitted from then on
  ask for no preview — the request carries no `preview`, the server draws
  none and its job document has none — and the page shows none, the
  running generation's included (which was asked with one, and goes on
  drawing it unseen). drift keeps the choice from a visit to the next.
- Nothing else is stored: a preview lives in memory from the runner to the
  page and is gone with the generation's end. It is never a gallery entry,
  and a generation does not record whether it was asked with one.

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
  | `wan-2.1` | Wan 2.1, packed 2 × 2 | 64 | 4 | Krea 2, Nucleus-Image, Wan 2.2 A14B |
  | `flux.2` | FLUX.2, packed 2 × 2 | 128 | 2 | FLUX.2 [klein], [dev] |
  | `qwen-image-2.1` | Qwen Image 2.1 | 64 | 2 | Qwen Image 2.1 |
  | `mage-flow` | Mage-Flow | 128 | 2 | Mage-Flow |
  | `ltx-2.5` | LTX 2.5's video VAE | 128 | 4 | LTX 2.5 |
  | `minimax-h3` | MiniMax H3's video VAE, packed 2 × 2 | 96 | 4 | MiniMax H3 |

  `LatentProjection.named(name, channels)` gives none for a file that is
  missing or of another width, and the family then draws no preview.
- **A video's estimate** (`VideoPipeline.preview`, the same hook with the
  sampling pass: `preview(pass, step, steps, clip)`). The three pipelines
  hold their video as rows of tokens, a latent frame after the other, and
  `LatentProjection.clip` reads each frame's grid as a picture: a
  `PreviewClip`, its frames drawn one by one when asked, over the clip's
  seconds. Wan's latent frames are the Wan 2.1 VAE's packed as Krea 2's, so
  they read with the same matrix; a video token of LTX 2.5 and MiniMax H3
  covers 32 × 32 pixels and gives 4 × 4. MiniMax H3's velocity points
  towards the data, so its estimate is `x + σ × v`; LTX's latent frames
  held clean by a condition in place (an init image) are read as they are.
- **One picture that moves** (`runner/.../server/PreviewGif.scala`). The
  server writes the clip as an animated GIF that loops (`ImageIO`'s
  sequence writer), each frame shown for its share of the clip's duration,
  enlarged by a whole factor towards 256 pixels on the longer side, on one
  palette of 64 colours cut from the clip's own (median cut, no dither). At
  most 64 frames, evenly picked; a GIF over 200 KB is made again less
  enlarged, then of half the frames, the duration kept. So the job
  document, the backend and the page carry a video's preview as they carry
  an image's.
- **HiDream O1** (`HiDreamO1Pipeline.reduced`): the estimate's patches
  averaged 8 × 8 pixels to one, 128² for a 1024² picture.
- **The fit** (`runner/gpuTest/.../PreviewProjectionFit.scala`, a main):
  encodes pictures with the family's VAE alone, as its pipeline does,
  averages each picture to `block` pixels a token, and solves the least
  squares (`LatentProjection.fit`, the normal equations with a small ridge).
  Each picture is encoded a second time with its colour channels swapped and
  every other one in negative, for the colours the pictures lack. A video
  VAE (LTX 2.5, MiniMax H3) encodes a short clip that holds the picture
  still, and both its latent frames are samples: the first frame's kind and
  the later ones'. It prints
  the error on the pictures held out and the time a preview takes, and
  writes the resource.
- **The request and the job document are sd-server's** (its
  `examples/server/api.md`): an `img_gen` or `vid_gen` request's `preview`
  names a mode, anything but `none` asking for previews, and `GET
  /sdcpp/v1/jobs/{id}` then carries `preview` beside the usual fields once a
  step has run: `{pass, step, total_steps, b64_json}`, `pass` the sampling
  pass from 1 (an image of the batch, Wan's high-noise then low-noise
  expert; on sd-cpp a hires pass too). The runner (`ImageServer`) takes any
  mode for its one way of drawing, sends a JPEG (an image) or a GIF (a
  video) where sd-server sends a PNG, and lists `preview` among its image
  and video features and `preview_modes`. A job whose request did not ask
  draws none. A video's leaves the document at the first stage that is not
  a sampling pass.
- **Asking.** The page decides: `GenerationService` sets `preview: "proj"`
  (`StepPreview.Mode`) on the image or video parameters it submits while
  previews are shown, and leaves the field out otherwise
  (`ImageGenerationParameters.preview`, `VideoGenerationParameters.preview`).
  `GenerationSubmissions` forwards it as it came and takes it off what the
  generation records. The choice is `GenerationPreferences.stepPreview`,
  stored by the backend (`settings/generation-preferences.json`,
  `GET`/`PUT /api/generation-preferences`), shown until one is stored.
- **The backend.** `GenerationMonitor` keeps a poll's `preview` on the
  generation's entry (`GenerationEntry.preview`, a `StepPreview`: the pass
  from 0, the picture's type told from how its base64 starts — JPEG, GIF,
  else PNG), emptied when the job ends. `GenerationManager.previews` is
  every entry's, by generation id.
- **The status socket** (11) has a topic for them, `StatusUpdate.previews`:
  a snapshot of the map, sent when it changes, so once a step. It is not a
  field of `Generation` (stored in the sidecar, and resent with the
  session's whole list) nor of `Session` (resent with each progress
  redraw).
- **The page.** `GenerationService.previews` mirrors the topic;
  `StepPreviewView` (`frontend/.../generate`) shows the entry of the
  generation on screen, in `GenerationResult` under `LogProgressView`: one
  `<img>` on a data URL, enlarged by CSS (`.step-preview`: 24rem wide at
  most, 60 % of the window's height at most, its shape kept), which plays a
  GIF by itself. A clip is coarser and shown at the size it is made at
  (`.step-preview.is-clip`: 16rem a side at most). `StepPreviewToggle` is the checkbox, in `GenerationPanel`
  over the queue and the result. The project page and the Sandbox share
  both through the generation panel.

## Notes

- **Cost.** Reading the two tensors back, drawing the picture and writing
  its JPEG takes 9 ms (128 values a token, block 2) to 18 ms (Wan 2.1,
  block 4) on the grid of a 1024² picture, measured alone and warm
  (`PreviewProjectionFit`). Measured on whole generations at 1024², the same
  seed with and without previews, a step is: FLUX.2 [klein] 4.95 s either
  way; HiDream O1 1.023 s without, 1.027 s with (+0.4 %); Mage-Flow 2.13 s
  without, 2.15 s with (+1 %). Krea 2 and Qwen Image 2.1 were measured on a
  GPU shared with other work, where the difference is lost in the noise.
- **A video's cost.** Timed in the server on Wan 2.2 A14B at 512 × 288, 33
  frames (9 latent frames, a GIF of 256 × 144): reading the two tensors
  back 1 ms, drawing the frames and writing the GIF 52 ms, 180 ms the first
  time of a server's life — 0.6 % of a 9.4 s step, 2 % of the first. Whole
  runs with and without, same seed, alternated, differ by as much from one
  run to the next as from the preview: LTX 2.5 at 704 × 416, 97 frames,
  8.69 s a step without and 8.74 s with (+0.6 %); at 320 × 576, 49 frames,
  3.32 s to 3.36 s without and 3.36 s to 3.39 s with (+1 %); Wan 9.26 s
  and 9.40 s without, 9.46 s and 9.55 s with; MiniMax H3 at 512 × 288, 56
  frames, 6.61 s and 6.78 s without, 6.90 s both times with (+3 %, on runs
  2.5 % apart).
- **At most one preview a second**, and a pass' last step's always
  (`ImageServer`): the backend polls once a second, and a model whose steps
  are shorter would draw pictures nobody sees. This also bounds an image's
  cost at about 2 % of the time whatever the model; a video's steps last
  seconds. A job whose request does not ask (a redraw's tiles, another
  client, previews hidden) draws none.
- **A video's last step** is drawn and hardly ever seen: the decoding
  starts at once and empties it.
- **A step can be missed** by the page: the poll is a second apart, and the
  last step's preview lives only until the job ends, which is at once for a
  model without a VAE to decode (HiDream O1).
- **Sizes.** For a 1024² picture a preview's JPEG is 3 to 8 KB at 128²
  (Qwen Image 2.1, FLUX.2, Mage-Flow, HiDream O1) and 6 to 17 KB at 256²
  (Krea 2), growing with the detail as the steps go; base64 on the wire, a
  third more. A video's GIF weighs far more, a GIF packing soft pictures
  badly: 120 to 146 KB for 13 frames of 176 × 104 (LTX 2.5, 704 × 416, 97
  frames), 63 to 69 KB for 7 of 120 × 216 (LTX 2.5, 320 × 576, 49 frames),
  145 to 149 KB for 9 of 256 × 144 (Wan, 512 × 288, 33 frames), 81 to 96 KB
  for 17 of 128 × 72 (MiniMax H3, 512 × 288, 56 frames — enlarged twice
  rather than four times, to stay under the 200 KB). With 256 colours the
  same clips weigh half as much again and look the same.
- **Quality.** The projection is the best linear reading of one token.
  Wan 2.1 latents read well (error 0.10 RMS on colours in [−1, 1], on
  pictures held out, at block 4): Krea 2's last preview is the picture, small.
  Qwen Image 2.1 reads well too (0.09), a little washed out, with a faint
  grid of tokens. FLUX.2 and Mage-Flow read worse (0.16): the composition
  and the tones are right, saturated colours come out dull and the token
  grid shows as speckle. The video VAEs read as well as the best (LTX 2.5
  0.09, MiniMax H3 0.10, at 8 pixels of the video a pixel): the subject, its
  motion and the tones are there from the first steps, with the token grid
  faintly visible and a cool cast on LTX's whites. A block of 8 on LTX 2.5
  (0.11) shows the grid more for little detail. A larger block does not help them: 4 pixels a side
  shows the grid more. HiDream O1's preview is the picture itself, reduced.
- **The fit's pictures** are test pictures made for it (portraits, a room,
  a terrace): few, and mostly people, which the recolouring only partly
  makes up for.
- **sd-cpp.** Its server's preview is a callback of the library
  (`sd_set_preview_callback`), set for the job that asks and published in
  its document; `preview_interval` (1 by default) spaces the frames. Seen
  from drift on `master-948`: the previews arrive, further apart than the
  steps when the GPU is shared, since the server answers a poll between two
  of its own steps. Videos take the same field there, and answer a still of
  the first latent frame.
- **The estimate is taken before the step moves the latents**, where `x`
  and the velocity are both whole: after it, the velocity is scaled in
  place. HiDream O1's is taken before its sampler reuses the tensor.
- **Inpainting** (a mask) previews the model's estimate of the whole
  picture, the kept part included, before it is restored.

## Remaining

- Not looked at in a page: a tall picture, a generation queued behind
  another, Cancel.
- FLUX.2 and Mage-Flow deserve a better reading than one matrix a token: a
  small decoder (a tiny autoencoder, as sd-cpp's `--preview tae`), or a
  projection over a token's neighbours, within the same cost.
- Redraw and edit tiles (`NativeJobs`, post-processing) run without a
  preview.
- A video's preview that weighs less than a GIF (an animated WebP or a
  short webm needs an encoder the runner does not carry).
- MiniMax H3's projection is fitted on pictures of 256 pixels (its encoder
  takes a minute a clip above); the fit's pictures at the videos' own sizes.
- sd-cpp's better modes (`tae`, `vae`) as a choice.
