# 10 — Generation-time upscaling

**Status:** done
**Depends on:** 08

sd-cpp's highres-fix driven from the generation form, with the upscaler weights
fetched and managed by drift. Post-hoc upscaling of an existing image is `15`.

## What it does

- The generation form's folded **Hires** section (`pages/generate/HiresSection
  .scala`), shown only when the session's capabilities flag `hires` on
  `img_gen`: an **Upscaler** select grouped into upscaler models, latent modes
  and plain filters (Lanczos, Nearest), a scale or explicit target
  width/height, hires steps, denoising strength, and an upscale tile size for
  model upscalers. Picking an upscaler seeds the denoising strength (model
  0.35, latent 0.65, filter 0.5).
- A folded **VAE tiling** section exposes the native `vae_tiling` block on
  image and video parameters, gated on the capability flag. Ticking it seeds a
  relative tile size of base ÷ target and overlap 0.25.
- The Model Cache page's **Upscalers** tab (`pages/UpscalerSection.scala`)
  installs weights from a curated list of the RRDBNet RealESRGAN releases, a
  free-form URL (optional label, file name, sha256) or the Civitai browser in
  UPSCALER type, lists installed ones with progress, and deletes them.
- Every sd-cpp launch passes the upscaler store as `--hires-upscalers-dir`, so
  an installed upscaler is visible to the next session.

## Shape

- Store: `~/.cache/drift/upscale/`, flat — sd-server scans only the top
  level. `Upscaler(id, label, fileName, downloadUrl, sha256, sizeBytes,
  civitaiModelId, createdAt)` in `shared/.../Upscalers.scala`; `id` is the
  file stem, which is the name sd-server reports and `hires.upscaler`
  carries. Civitai installs are stored `<fileId>-<name>`.
- Endpoints: `GET /api/upscalers`, `POST /api/upscalers/install`
  (`InstallUpscalerRequest(url, label, fileName, sha256)`),
  `POST /api/upscalers/install-civitai`
  (`InstallUpscalerFromCivitaiRequest(civitaiModelId, versionId, fileIds)`),
  `DELETE /api/upscalers/{id}`, `GET /api/upscaler-downloads`.
- Backend `backend/.../upscale/UpscalerManager.scala`, patterned on
  `LoraManager` without the architecture/nsfw structure; it re-queues every
  entity at start-up, and a file already on disk shows as an immediately
  `Completed` job, so the job list doubles as the on-disk answer.
- Request side: `HiresParameters` and `VaeTilingParameters` in
  `shared/.../Generation.scala` mirror the native `img_gen` `hires` and
  `vae_tiling` objects (`target_width`/`target_height` 0 = use scale, `steps`
  0 = auto).

## Notes

- Upscalers are global: nothing on the run configuration, one launch flag on
  every session. A RealESRGAN model upscales pixels whatever produced them.
- sd-cpp loads RRDBNet-class ESRGAN only (`.pth` included), no SRVGG
  "compact" nets, and RealESRGAN **x2plus** fails to load (12-channel
  `conv_first`); x4plus works.
- VAE tile sizes are latent units, not pixels; the pixel ratio depends on the
  model's VAE (8/16/32 by family). Numbers that work at 2×: rel 0.5, overlap
  0.2, upscale tile 512, denoise 0.3–0.35.
- A tiled ESRGAN pass starves sd-server's HTTP loop for minutes while the job
  is healthy; `GenerationManager` judges poll timeouts by silence duration
  (15 min), not by count, while connection-refused still fails fast. Without
  this the finished output is lost.
- The Civitai token is sent only to civitai.com hosts.
