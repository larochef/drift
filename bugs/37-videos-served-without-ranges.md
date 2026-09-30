# Bug 37 — Videos are served without byte ranges: players wait up to ~30 s

**Status:** fixed (found 2026-09-30, confirmed by François in Firefox
2026-10-01): fixes 1 and 2 done; fix 3 optional, not done
**Severity:** medium (a video in the generation detail can stay blank for
about 30 s; the gallery keeps downloading whole videos it only shows as
tiles)
**Files:** `shared/src/drift/shared/Generation.scala` (`getOutputFile`),
its server logic in the backend's generation routes,
`frontend/src/drift/frontend/pages/gallery/GenerationCard.scala`
(`thumbnail`), `frontend/src/drift/frontend/pages/gallery/GenerationMediaViewer.scala`
(`media`, `stripEntry`)

## Symptom

Opening a video's generation detail sometimes shows nothing for up to ~30 s,
then plays. Firefox's network log shows requests like
`GET /api/outputs/2026-09-17/g1789633032499-2.webm` ending in
`NS_BINDING_ABORTED`.

## Root cause

- `getOutputFile` answers every request with the whole file as one byte
  array (`byteArrayBody`), `200 OK`, and no `Accept-Ranges`: asked for
  `Range: bytes=4000000-` of a 4.2 MB webm, it sends all 4 178 346 bytes.
- The webms (sd-server's and the drift runner's, both written by libwebm /
  ffmpeg) keep their Cues (seek index) at the end of the file. A player reads
  the head, aborts that download (the `NS_BINDING_ABORTED`) and asks for the
  tail by range; it gets the whole file again, so each video costs one or two
  full downloads before it can show a frame.
- Many videos are on the page at once: every gallery card is a
  `<video preload="metadata">`, and the detail page adds the player and one
  `<video>` per strip entry (no `preload`, so the browser's default). With
  HTTP/1.1's six connections per host, the player's request queues behind
  them. The server itself is fast (4 ms for the whole file when idle).

## Suggested fix

1. Serve outputs with range support: `Accept-Ranges: bytes`, `206 Partial
   Content` for a `Range` request (tapir's file bodies with ranges, or
   `staticFilesGetServerEndpoint`), streamed from the file rather than read
   into memory.
2. Stop loading videos the page only shows small: gallery cards and strip
   entries with `preload="none"` and a poster (a still frame, which needs a
   thumbnail of the video, e.g. extracted by ffmpeg when the video is saved),
   so only the player downloads the video.
3. Optional: webms written with the Cues at the front (ffmpeg
   `-cues_to_front 1` with `-reserve_index_space`) for the runner's files; not
   possible for sd-server's.

Fix 1 as done: `getOutputFile` takes the `Range` header and answers a
`FileRange` body, streamed from disk (`206` with `Content-Range` for one range,
`416` with `bytes */<size>` past the end, the whole file otherwise,
`Accept-Ranges: bytes` always); `backend/.../routes/ByteRanges.scala` parses
the header, `ByteRangesTests` covers it. It applies to every output, images
included. The endpoint sets `Content-Length` itself: tapir's Netty server sends
a file range without one and does not close the connection when the range ends
before the file does, so a client asking for bytes 100–199 got them and then
waited forever.

Still waiting ~30 s at times after fix 1: Firefox keeps a removed
`<video>`'s range request open, half read, until the element is
garbage-collected, and a page holds up to six of them (a project's detail
strip of five and its player). `components/VideoRelease.onUnmount` (every
drift-hosted `<video>`: gallery card, detail player and strip, generation
result, project cover, version history) removes the source and reloads on
unmount, which ends the download at once; compiled, to be judged live. If the
waits stop, fix 2 becomes an optimization.

Fix 2 as done, after the release alone still left one ~30 s wait: videos
play only in the detail's player and the generation panel's latest result.
Gallery cards, the detail strip, version histories and project covers show
the first frame: `OutputPreviews.preview` of a video is that still (ffmpeg via
`images/VideoFrames`, cached as `<stem>-still-<side>.png`), and a video
project's cover is it as a JPEG; the frontend asks for
`GenerationMediaViewer.stillUrl`.

## Verification

- `curl -D - -H "Range: bytes=4000000-" …/api/outputs/<date>/<file>.webm`
  answers `206` with `Content-Range` and only the tail.
- With a gallery page of several videos open, a video's detail plays at
  once; the network log shows range requests and no whole-file re-downloads.
