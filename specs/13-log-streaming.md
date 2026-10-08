# 13 — Log streaming, progress and readiness

**Status:** done
**Depends on:** 07

Loading a 30 GB model takes minutes and the reason for a failure is in the log
and nowhere else. The log is also the app's **only** progress signal:
sd-server's job JSON carries status and queue position, no step count. So
drift drains the process output, parses progress bars out of it, and shows the
raw tail on demand.

## What it does

- Readiness is polled (`/sdcpp/v1/capabilities`, `/health` for llama.cpp)
  with a configurable timeout (15 minutes); the log is narrative, never the
  readiness decision. A timeout or crash fails the session with the log tail
  and the last error-looking line first.
- Every session card and the generation panel show a progress bar sourced
  from the log: "loading weights 15/298 · 16.95GB/s" while tensors load,
  "sampling 2/4 (50%) · 6.80s/it" during a generation, with the pass name
  when a model runs several ("sampling (high noise) 5/20"), "decoding 34/100
  (34%) · 1.87%/s" while the drift runner decodes an LTX 2.5 video after its
  last step (`ProgressKind.Decoding`, read from the `%/s`). No bar in flight
  → the last non-progress line is shown instead, sd-cpp's
  `[INFO   ] file.cpp:123 - ` prefix stripped.
- A batch shows two bars in the generation panel: "Image 2 of 4" over a bar of
  the images already done, then the bar of the image in flight. The position
  comes from the `generating image 2/4` line each engine prints
  (`Session.batch`) and ends on `generate_image completed`. A single image
  shows only its own bar.
- A log view with the raw tail, autoscroll, and replay of what already
  happened when opened mid-load.
- A pattern that stops matching degrades to "no bar", never to a wrong or
  stuck percentage.

## Shape

- `shared/.../Logs.scala`: `LogLine(at, text)`, `SessionProgress(kind,
  done, total, detail, note)` with `ProgressKind = Loading | Sampling`,
  `LogProgress.parse` / `parseWithRest` / `clean` / `samplingPassOf` /
  `looksLikeError`. The two bar shapes, told apart by unit only:

  ```
  |###             | 15/298 - 16.95GB/s     (tensors)
  |============>   | 1/4 - 6.77s/it         (sampling steps)
  ```

- Transport split by shape: progress and activity ride the `sessions` topic
  of the status socket as `Session.progress` / `Session.activity` (no new
  topic); the raw tail streams as NDJSON from
  `GET /api/sessions/{id}/logs` (`routes/SessionLogRoutes.scala`, replays the
  ring buffer then follows), opened only while the log view is.
- Backend `session/SessionOutput` drains stdout and stderr on a reader fork
  and mirrors them verbatim to `~/.cache/drift/logs/<sessionId>.log`;
  `session/SessionLog` is the bounded ring buffer that collapses progress
  redraws (each replaces the previous).
- Frontend `components/LogProgressView` — the bar and the last line, over a
  `(Option[SessionProgress], Option[String])` signal, so a session's panel and
  a post-processing job's row (15) show the same thing — and
  `pages/generate/SessionLog`;
  `services/LogService` reads the NDJSON with the same fetch + ReadableStream
  reader the assistant chat uses.

## Notes

- **sd-cpp redraws bars with a carriage return** and ends each redraw with
  `ESC[K`. The reader treats `\r` as a line terminator and strips ANSI before
  matching or display; splitting on `\n` alone delivers a whole generation's
  bar as one line after the work is done.
- sd-cpp does not always newline after a redraw: `5.4s/it[INFO   ] …` arrives
  glued. Units are matched by name (`s/it`, `it/s`, `[KMGTP]?B/s`) and the
  trailing text is re-fed as its own line.
- sd-cpp switches `s/it` ↔ `it/s` and `MB/s` ↔ `GB/s` mid-run; matching one
  form silently drops about half the redraws.
- Many bars per session is normal (a session prints between 6 and 256 bar
  runs; H3 alternates a 624-tensor bar with a 416-tensor one). The UI shows
  whichever bar redrew last with its own totals; never filter by kind or
  stitch bars into one number.
- Any narrative line clears the bar: the phase ended.
- sd-cpp pads log levels (`[ERROR  ]`), so error detection matches `[ERROR`,
  never `[ERROR]`.
- No animated placeholder: Bulma's `.progress:indeterminate` repaints every
  frame forever and makes the whole UI laggy. The log body re-render is
  throttled to 400 ms.
- Bulma's `.content pre` sets `white-space: pre` and out-specifies a bare
  class; the log body and assistant text rules are written
  `.content pre.<class>` or the page stretches to the longest line.
- Video models are not all verified; a differently shaped bar parses to
  nothing and the activity line carries the narrative, which is the intended
  failure.
