# Bug 54 — A SeedVR2 video upscale fails once every step is done: "Broken pipe"

**Status:** fixed 2026-10-08 (raised by François the same day: a 2 min 28 real-world video, ×2, 783 steps, four
hours, then "sd-server failed the job: Broken pipe"); longer videos fixed 2026-10-09, run end
to end with the model on an isolated drift
**Severity:** high (hours of work lost at the last second)
**Files:** `runner/src/drift/runner/server/VideoFiles.scala`, `runner/src/drift/runner/server/VideoUpscale.scala`,
`runner/src/drift/runner/server/ImageServer.scala`, `runner/src/drift/runner/diffusion/SeedVr2Pipeline.scala`,
`backend/src/drift/backend/sdserver/NativeJobs.scala`, `backend/src/drift/backend/postprocess/SeedVr2Upscale.scala`

## What happened

Not memory. The log (`postprocess-g1791412982533-1001.log`) ends with all 783 steps, `generate_image completed
in 14311.60s`, then `java.io.IOException: Broken pipe`: the failure is in `VideoFiles.webm`, writing the frames
to ffmpeg.

The source's soundtrack (147.33 s) is shorter than its frames (4439 at 30 per second: 147.97 s). The encoder
ran ffmpeg with `-shortest`, so ffmpeg stopped at the end of the sound, closed its input and exited with status
0 while the runner still had 19 frames to write. Reproduced on the processor alone: 300 frames and 9.3 s of
sound gave 279 frames and a writer killed by the pipe; without `-shortest`, 300 frames.

A generated video never met it (its soundtrack is made to its length), nor a short clip (the frames left over
fit the pipe's buffers).

## Fix

`VideoFiles.webm` keeps every frame: no `-shortest`; a soundtrack that runs past the last frame is cut there
before it is written, a shorter one ends in silence. A write that fails now reports ffmpeg's own log instead of
the pipe's error. `VideoFilesTests` covers both lengths.

## Longer videos

A restored video was whole in memory at every stage, and several of those had a hard ceiling:

- the source arrived as base64 in the request, and its decoded frames came out of ffmpeg as **one byte array**
  (`VideoFiles.run`: `readAllBytes`) — 2 GB at most, about 9000 frames of 320 × 240, 2400 of 640 × 480;
- every source frame and every restored frame was a `BufferedImage` on the heap until the end;
- the result was one webm read back into an array, then one base64 string in the job's JSON — 2 GB again;
- nothing was written before the end, so any failure there lost the whole run.

Now the video streams (`specs/51`):

- the request names two files, `source_file` and `output_file`; nothing of a video is in the JSON either way
  (a video sent as `source` is refused);
- the frames are read from ffmpeg one at a time (`ClipFrames`), restored a batch at a time
  (`SeedVr2Pipeline.batched` is read as it is asked for, so the first pass of a ×4 feeds the second batch by
  batch), and written to an ffmpeg that runs from the start (`WebmWriter`);
- the soundtrack is the source's, taken from its file by the encoder; the frame rate is the source's own
  fraction (29.97 is 30000/1001 in the result, where it was rounded to 30 and ran 0.1 % fast against the sound);
- a job stopped, or failed after its first frames, closes the encoder on the frames written, cuts the sound
  where they end, and ends `cancelled` or `failed` *with* its result; drift saves the shorter video, marked
  "partial: the first N of M frames" (`Derivation.warning`). **Stop** asks the runner to stop the job instead
  of stopping its server.

Run with SeedVR2 3B through an isolated drift (`~/dev/redraw-experiments/2026-10-09-video`, the GPU shared
with other jobs): 182 frames at 30000/1001 with a soundtrack a second short, ×2 — every frame, the rate and
the sound kept; 97 frames ×4 in two passes feeding one another — 97 frames at 1024²; a run stopped after its
first batch — the 17 frames made saved as a video with its sound cut there, the entry marked ⚠ "partial: the
first 17 of 182 frames"; a run stopped before its first batch ended — nothing saved. The bar counts a step
of a later pass by its pixels over the first one's, so that it moves with the time spent.

Checked without the GPU, on an upscaler that only resizes (`ResizePass`): `VideoUpscaleTests` (every frame
at 30000/1001 with a shorter and a longer soundtrack, two passes, a run stopped at its second batch, one
stopped before its first frame), `SeedVr2BatchesTests` (the batches read as asked for), and `LongClipCheck`
for the size: 3000 frames of 640 × 480 at 30000/1001 with a soundtrack 0.7 s short — 2.6 GiB of raw frames,
past the array's limit — go through in a 512 MB heap in 75 s, all 3000 in the file; the same ×2 (10.5 GiB of
frames out) in a 1 GB heap in 341 s.

Not covered:

- **A runner that dies** (killed, out of memory, a GPU hang) takes its video with it: nothing ends the file
  properly, and drift removes what is left. Resuming a stopped video is not built either (declined).
- A source larger than 1080 px on its short side is still brought down to it before the upscale
  (`ClipShortEdge`, as every clip the runner decodes).
- The generated videos (`vid_gen`) still come back whole as base64; they are seconds long.
- The frames are converted and written to ffmpeg on the worker's thread, the GPU idle meanwhile (`bugs/48`).
