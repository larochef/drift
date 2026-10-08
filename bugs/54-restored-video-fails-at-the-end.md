# Bug 54 — A SeedVR2 video upscale fails once every step is done: "Broken pipe"

**Status:** fixed 2026-10-08 (raised by François the same day: a 2 min 28 real-world video, ×2, 783 steps, four
hours, then "sd-server failed the job: Broken pipe"); the limits on longer videos below are open
**Severity:** high (hours of work lost at the last second)
**Files:** `runner/src/drift/runner/server/VideoFiles.scala`, `runner/src/drift/runner/server/ImageServer.scala`

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

## Still open: longer videos

A restored video is whole in memory at every stage, and several of those have a hard ceiling:

- the source arrives as base64 in the request, and its decoded frames come out of ffmpeg as **one byte array**
  (`VideoFiles.run`: `readAllBytes`) — 2 GB at most, about 9000 frames of 320 × 240, 2400 of 640 × 480;
- every source frame and every restored frame is a `BufferedImage` on the heap until the end;
- the result is one webm read back into an array, then one base64 string in the job's JSON — 2 GB again;
- nothing is written before the end, so any failure there loses the whole run.

The fix is to stream: frames read from ffmpeg and restored a batch at a time, written to the encoder as they
are made, the result left as a file the backend reads, and what is done kept if the job stops. A 10 minute
video needs it; until then it fails at the start on the 2 GB array (quickly, at least).

Seen on the way, not fixed: the frame rate is rounded to a whole number (29.97 → 30), so the restored video
runs 0.1 % fast against its soundtrack.
