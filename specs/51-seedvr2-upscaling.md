# 51 — SeedVR2 on the drift runner: video and image upscaling

**Status:** done in code, run once end to end in drift on a copy of the configuration (a picture and a video through the API, the panel in a browser); not yet used by François
**Depends on:** 15 (post-processing), 26 (PiD), 42 (the drift runner), 43 (a runner per configuration)

drift upscales pictures (ESRGAN, PiD) and has nothing for video. SeedVR2
(ByteDance, Apache 2.0) restores and upscales video and single images in
**one** diffusion step, and is the candidate for both: the missing video
upscaler, and a second image upscaler beside PiD. sd-cpp does not run it,
and its PyTorch code cannot be used on this machine (Notes), so it is built
in the drift runner, as PiD was.

## What it does

- **Upscale a video** from the gallery: a post-processing task on a video's
  detail, beside the soundtrack polish (48). It gives a target size (×2 by
  default, the short side settable), keeps the frame rate, the length and
  the soundtrack, and saves the result as a new entry **Made from** the
  original.
- **Upscale a picture** with it: a fourth choice in the **Redraw & upscale**
  tab beside ESRGAN and PiD (15), on the same `Derivation` record.
- Two models, 3B and 7B, as run configurations of one `seedvr2`
  architecture tagged `upscale`; drift's runner is its only engine. **7B
  and ×4 are the defaults.**
- Both the picture and the video task are built. The model redraws more
  than PiD does; if it invents too much, the task says so in a warning.
- No prompt: the model is conditioned on two fixed text embeddings shipped
  with it.
- The **Original** / **Result** toggle of 15 compares either version at the
  same position and zoom, for a video as for a picture.
- Progress counts batches of frames, as PiD's counts tiles.

## Shape

What the model is, read from its reference code
(`numz/ComfyUI-SeedVR2_VideoUpscaler`, `configs_3b`, `configs_7b`):

- **The transformer (`NaDiT`).** A multimodal DiT over video tokens of
  1 × 2 × 2 latents and text tokens.
  - 3B: width 2560, 20 heads of 128, 32 layers of which the first 10 keep
    video and text apart (`mm_layers`), SwiGLU MLPs, 3D multimodal RoPE.
  - 7B: width 3072, 24 heads, 36 layers, all with video and text apart,
    plain MLPs with biases, a RoPE on the video alone (angles over each
    window from −1 to 1, frequencies π to 128 π), a bare linear output.
  - Attention is **windowed**: each layer attends inside windows of the
    token grid (`window: (4, 3, 3)` over time, height and width, sized
    from a 720p reference), plain and shifted windows alternating layer by
    layer. That is what keeps a long
    or large video affordable, and it is the one attention shape the runner
    does not have yet. The windows keep their size and grow in number: the
    grid is scaled to 720p's area (45 × 80 tokens), cut in 3 × 3 there, and
    that window (about 20 × 20 tokens of a square picture, `ceil(min(t, 30)
    / 4)` frames) is laid over the real grid. The text tokens are repeated
    into every window and their outputs averaged over the windows; RoPE
    positions are counted inside the window, not on the whole grid.
  - Input 33 channels a token: 16 of noise, the 16 of the low-quality
    video's latent as it is (no noise added to it), and a mask channel at
    1; output 16.
- **The video VAE.** A causal 3D VAE, 16 latent channels, 8× in space and
  4× in time, latents scaled by 0.9152; a video is `4n + 1` frames, a
  picture is one. The latent is the posterior's mode: encoding draws
  nothing. `ema_vae_fp16.safetensors`, 0.5 GB. The runner has causal
  video VAEs already (`WanVideoVae`, `LtxVideoVae`, `MiniMaxH3Vae`,
  `CausalModel`); this one is its own network and reuses their layers.
- **Preparation.** The low-quality input is resized (bicubic) to the output
  size, clamped, padded to a multiple of 16 and encoded; the padding is
  cropped from the result.
- **Sampling.** One Euler step of a v-prediction on a `lerp` schedule
  (T = 1000), no guidance: the model is step-distilled, and the result is
  `noise − velocity` at timestep 1000. The reference's shift of the
  timestep by the latent's size is never applied on this path (it serves
  an optional noising of the condition, off by default): the runner does
  not have it.
- **Noise.** The reference's latent is a permuted view, so PyTorch draws
  its noise value by value, channel-first; the runner draws the same
  stream (`SeedVr2Pipeline.noise`), so a seed gives the reference's noise.
- **Text.** No text encoder: `pos_emb.pt` and `neg_emb.pt` (0.6 MB each),
  width 5120, ship with the model and only the positive one is used. It is
  a resource of the runner (`runner/resources/seedvr2/text.safetensors`,
  in Git LFS as every `.safetensors`): it is published only as a PyTorch
  pickle (`ByteDance-Seed/SeedVR2-3B`), which the runner does not read. Were
  a safetensors of it published, it would be a slot of the architecture
  instead.
- **Colour.** The reference matches its result to the source's colour
  (`lab` by default). Here that is drift's post-processing, as for the
  other upscalers: the runner returns the frames as decoded, and drift's
  `colourMatched` (45) corrects them.
- **Long and large inputs.** Frames go through in batches of `4n + 1`
  (21 by default), each starting on the last frames of the one before (4),
  where the two results are cross-faded as the reference does; every batch
  from the same noise; a last batch short of `4n + 1` is filled with its
  last frame. The VAE encodes and decodes in spatial tiles (1024 px
  overlapping by 128 at least, the reference's sizes), blended over the
  overlaps; the transformer takes the whole frame, its attention being
  windowed. The tiling is `VaeTiles`, any VAE's to use: MiniMax H3's cut and
  blend, taken out of it.
- **A still frame** — a picture, or a stream's first frame — goes through
  each causal convolution's three slices summed into one kernel.
- **Weights.** `seedvr2_ema_3b` and `seedvr2_ema_7b`, fp16 or fp8 scaled
  (3.4 GB and 8.2 GB in fp8), from `numz/SeedVR2_comfyUI`; fp8 decodes to
  BF16 as the other ComfyUI quants do. An fp8 file also rounds the RoPE
  frequencies it stores (0.645 to 0.625, the lowest to 0) and the reference
  uses them so; the runner computes the exact ones, and the goldens are made
  with those.
- **The output's modulation** takes the timestep's *attention* shift and
  scale: the reference reads them from its cache under the attention's key.
  The model was trained so, and the runner does the same.

In drift:

- An architecture `seedvr2` (`tool: SdCpp` for its place among image and
  video models, `runners: [DriftRunner]`, tagged `upscale`, model kind
  `seedvr2`), with slots for the transformer and the VAE; the embedding
  travels with the runner. Built-in models from `numz/SeedVR2_comfyUI` (7B,
  7B sharp, 3B in fp8, the VAE) and two starter configurations on the
  runner, `starter-seedvr2-7b` (the panel's default) and
  `starter-seedvr2-3b`.
- The runner's sd-server answers a native job of its own for it,
  `POST /sdcpp/v1/upscale`: `source` (a picture or a video, base64),
  `scale` (4 unless said) or `width` and `height`, `seed`, `batch`,
  `overlap`. A picture comes back as `img_gen`'s result does, a video as
  `vid_gen`'s (a webm, its frame rate and soundtrack kept). Its progress
  bar counts, for every batch, its VAE tiles encoded, the transformer's
  step and its tiles decoded; drift shows it on the job's card, and
  **Stop** stops the job's server, the runner having no way to drop a job
  it is generating.
- A post-processing job kind `seedvr2` (`postprocess/SeedVr2Upscale`,
  `POST /api/outputs/{date}/{file}/seedvr2`, `SeedVr2UpscaleRequest`), the
  only kind that takes a video. The result is a **Made from** entry
  (`Derivation.operation` `seedvr2`, `repeats` its scale).
  - **A picture** is a tiled job, as PiD's is (`TiledJobs`): the source
    padded to multiples of 16, cut in tiles of at most 4352 target px
    overlapping by 256 — one tile for most pictures, 16 for 4096² to 16384²
    — each the runner's `upscale` job on its crop, colour-matched to that
    crop (`colourMatched`) and feather-blended. It pauses and resumes, shows
    its tiles on the picture, and stops at 16384 px on the longest side.
    What a tiled job asks its server for is the one thing that differs
    between models (`TileRequests`: an `img_gen` a tile for PiD, redraw and
    edit; the runner's `upscale` a tile for SeedVR2).
  - **A video** goes whole, on a runner the job starts and stops; the
    runner cuts it itself (batches, VAE tiles). It is saved as the runner
    encoded it, unmatched, and does not pause; **Stop** stops its server.
- One **Upscale** task in the Redraw & upscale tab (`UpscaleTaskPanel`):
  its model select lists the SeedVR2 and PiD configurations and the ESRGAN
  models, and what follows is what the chosen model takes — for SeedVR2 the
  scale (×2, ×4), a seed under Advanced, and the warning that it redraws
  detail. On a video it is the only task, with the SeedVR2 models alone.

Not there yet:

- **A video's colours** are not matched: drift's `colourMatched` works on
  pictures and the backend does not decode videos.
- **Original / Result on a video** swaps the two files; it does not keep
  the playback position.
- **A SeedVR2 configuration launched as a session** has nothing to offer:
  its server has the `upscale` mode alone, and the generation form expects
  `img_gen` or `vid_gen`.
- **The last frame of a video with a soundtrack** can be dropped by the
  runner's webm encoding (`-shortest`), as for the generated videos.
- Inputs travel as base64 in the job's request and a video's frames are
  held in memory by the runner: fine for clips, not for long films.

## How it is checked

As 42 does it: every piece on the `Cpu` oracle first, then on `Hip`, against
golden fixtures.

- The reference is the Python code, **on the CPU only**
  (`HIP_VISIBLE_DEVICES=""`), in float32: `runner/fixtures/tiny_seedvr2.py`
  builds its networks from the released configurations.
- Fixtures per piece, on tiny networks with random weights
  (`fixtures/tiny/seedvr2*`): the VAE's encoding and decoding of one frame
  and of 5; the transformer over a 5 × 20 × 24 token grid — several
  windows on every axis, moved by the shifted layers, time included —
  with every block's output and the first plain and shifted attentions
  tapped; the windows of both methods for a list of grid sizes.
- The whole thing on the real 3B weights: a synthetic 64 × 64 picture and a
  5-frame clip of it upscaled to 128 × 128, every stage kept (prepared
  input, latent, noise, velocity, restored latent, decoded frames, colour
  match). The same picture through the reference's own command line
  agrees to 0.7 / 255 on average.
- On the runner: the tiny VAE within 0.7 % (`Cpu`) and 1.4 % (`Hip`) of the
  reference, the tiny transformer within 0.001 % and 0.4 %, the windows
  identical for every grid, the tiny 7B within 0.001 % and 0.4 %, the real
  7B within 0.14 % on average of its golden picture; the real 3B on `Hip` within 0.12 % on average of
  the golden picture and clip (worst pixel 2 %), and within 0.2 % on
  average on a 416² crop of several windows (worst pixel 11.5 %).
- Then by eye on a purpose-made subject: the 832² crops the redraw study
  used (face, chest, sand), against PiD's result on the same crops, and a
  49-frame pan over it for flicker.

## Steps

1. The Python reference running on the CPU, and its fixtures.
2. The VAE: encode and decode of one frame, then of 5 frames.
3. The transformer: tokens, RoPE, windowed attention, one block, all blocks.
4. One picture end to end on `Cpu`, then on `Hip`; its speed measured here.
5. A clip: batches of `4n + 1`, the temporal overlap, the VAE's tiling.
6. The runner's job and drift's two tasks (video, picture), the gallery
   entries, the comparison toggle on a video.
7. 7B, and fp8 weights.

## Notes

- **Why not the Python code.** Run through PyTorch on ROCm for gfx1151
  (TheRock's wheels, torch 2.11), it stops in the VAE's encoding with
  `HW Exception by GPU … GPU Hang`, twice, and the hang took the desktop
  down (2026-10-04). On the CPU it runs: the reference's command line
  refuses `cpu:0` as a device for safetensors, which a small wrapper maps
  to `cpu` (`~/dev/redraw-experiments/2026-10-04-seedvr2/run_cpu.py`).
- **A first look.** The study's three 832² crops to 1664² with 3B fp8,
  through the CPU reference and through the runner, beside PiD's on the same
  crops (`~/dev/redraw-experiments/2026-10-04-seedvr2/*-compare.png`:
  bicubic, the runner's 3B and 7B without a colour match, PiD's ×4
  reduced; `*-x4-compare.png`: the 416² centres ×4 by the 7B beside
  PiD's). The 7B keeps more of the skin than the 3B. SeedVR2 restores harder: sharper hair, eyes and sand,
  cleaner edges — and it redraws the face a little (its shape, smoother
  skin, fewer freckles) where PiD keeps it. Whether that is better is for
  the eye to say; the video half stands either way, nothing else upscales
  video.

## Open questions

- **Is it better than PiD on pictures**, or only different? A first look
  is in the Notes (PiD's ×4 result reduced to the same size); to be judged
  by eye, and again at step 4 on the runner.
- **How large in one pass.** Windowed attention should take a 4k picture
  whole; the VAE is the part that needs tiles. To be measured, not guessed.
- **Speed.** Measured on this laptop, 3B fp8, 832² to 1664², one pass,
  nothing tiled: 32 s — 9 s encoding, 4 s for the transformer, 19 s decoding
  (the CPU reference: 4 to 5 minutes). The VAE is where the time is: on a
  single picture each causal convolution runs its three temporal slices
  over the same frame, which one summed kernel would do in a third. The 7B
  takes 6 s where the 3B takes 4.
- **×4 needs the tiles.** 832² to 3328² untiled filled the machine's
  121 GB in the VAE's encoding and had to be killed. Tiled, with the 7B:
  102 s (26 s encoding, 24 s for the transformer, 53 s decoding), 36 GB
  of memory in use at the most, no seam to see.
- **Whether fp8 loses anything visible** against the fp16 files.
- **Redraw on video** is not this spec: SeedVR2 restores, it does not
  repaint to a prompt.
