# Bug 33 — A resumed model download can finish full-size and fail its checksum

**Status:** fixed 2026-09-28 on the likely cause (not committed, not reproduced)
**Severity:** medium (costs a whole re-download of several GB; the retry from
zero works, and nothing corrupt ever reaches the cache)
**Files:** `backend/src/drift/backend/download/Downloader.scala`

## Symptom

On 2026-09-28 the starter Krea 2 Turbo weights
(`realrebelai/KREA-2_GGUFs` `TURBO/Krea-2-Turbo-Q4_K_M.gguf`, 7 216 993 376
bytes) downloaded to their full size and then failed:

```
checksum mismatch: expected 273a98be1afe317bc7228403b6434647eaf866cebe6aff1980c401b950473807,
got f05027244dfcd7e11d4b16a1eb5a12e3cb8630f090e51f0625d2cc5f53d06c59
```

HuggingFace still served `273a98be…` (last commit 2026-06-24), so the source
had not changed. The download took about three minutes, so it probably resumed
from a partial left by earlier runs. François had started and stopped drift
many times during the day, and had seen the same failure on an LLM download
earlier. The Qwen3-VL-4B download beside it verified fine.

## What is known

- The failure path is correct: a mismatch deletes the `.part` file and its
  `.chunks` sidecar (`Downloader.scala` ~line 316, and ~line 586 on the
  sequential path), so a retry starts from zero.
- Chunk completion is recorded in the `.chunks` sidecar only after a chunk's
  `downloadRange` returns (`runForeach` ~line 286).

## What does not explain it

Killing drift during a download should not corrupt the partial. Bytes a killed
process has already written are in the page cache and still reach the file.
Only an OS crash or power loss drops them.

## Likely cause, fixed

François's drift runs got interrupted by Ctrl+C on mill and by machine
crashes. A crash (not a process kill) drops what the page cache had not
written back. The `.chunks` sidecar is a small file rewritten on every chunk,
so it can reach the disk while the 64 MB of file data it vouches for does not.
The resume then trusted chunks full of zeros or stale bytes. Now
`channel.force(false)` runs before each `saveChunkState`, so a chunk is recorded
only once its bytes are on disk.

## Other leads, if it happens again

1. **A sidecar from another layout.** Check how a `.chunks` file is adopted when
   its `chunkBytes` or `totalBytes` differ from the current download (a changed
   `Downloader.ChunkBytes`, or a model whose file changed size). Adopting it
   with the new chunk size would mark the wrong byte ranges as present.
2. **The prefix-partial adoption** (sequential `.part` → chunk map). Check that
   only *fully covered* chunks are adopted and that the boundary chunk is
   re-fetched.
3. **Two transfers of one file at once.** A second start of the same model
   while the first still runs (a UI double click, or two pages both pushing
   `Start`) would have two writers on one `.part`.

## Verification

Reproduce by cancelling (not killing) a chunked download midway, restarting
drift, and resuming it. Then repeat with a kill at ~50 %. Both must verify.
Add a `DownloaderTests` case for each lead that turns out to be the cause.
