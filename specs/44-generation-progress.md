# 44 — Generation progress: batches and the runner's own progress

**Status:** planned
**Depends on:** 15 (post-processing jobs and their progress), 22 (free play), 42 (the drift runner), 43 (a runner per configuration)

A generation's progress shows what the job is doing and how far it has
got, as the tiled post-processing does. A batch has a bar for the whole batch
and a bar for the image being made. On the drift runner, progress comes from
a report the runner writes for drift, not from sd-cpp's terminal bar
reproduced in its log.

## What it does

- **Two bars for a batch** (François). A generation of several images shows:
  - the batch: "image n of m", filled by the images done;
  - the current image: its phase and its steps.

  This is the same pair as a tiled job's "tile n of m" and its step bar. A
  single image shows only the second bar. Both bars are the running job's
  and stay when the panel shows a job queued behind it.
- **The current image's bar covers the whole image.** It moves through
  encoding the text, the steps and decoding, instead of restarting at each
  phase. The phase is named beside it: loading, text, sampling (with its
  pass, for two-pass models), decoding.
- **Time left.** Each bar says the time left once one step (or one image) has
  run: from the step rate for the image, from the images done for the batch.
- **On the drift runner** the bars come from the runner's structured
  progress. It reports:
  - phase, step and total;
  - the image within the batch;
  - the tile within the job, for tiled jobs;
  - elapsed time.

  It stops printing sd-cpp's `|====| i/n - Xs/it` imitation for drift to
  parse (`ImageServer`'s sampling line).
- **On sd-cpp** nothing is lost: the log bar stays the source, parsed as today
  (`LogProgress`). The batch bar counts sd-cpp's sampling bars, a new one
  starting at step 1 for each image.

## Shape (to be settled)

- The runner's report: a line format of its own in the log (for example one
  JSON object per line with a fixed prefix), or a progress field on
  `GET /sdcpp/v1/jobs/{id}`, which drift already polls. The job endpoint is
  preferred: it needs no log parsing, and it is per job, so progress can't be
  read off the wrong job. Check first whether upstream sd-server's job
  endpoint already reports progress (its docs), and match its field names if
  it does.
- `SessionProgress` (`shared/.../Logs.scala`) grows the image index and count
  and a phase that isn't limited to sd-cpp's two bars (`ProgressKind`
  Loading/Sampling).
- One progress component for sessions and post-processing jobs, instead of
  `LogProgressView` and `PostProcessJobList`'s own bar.
  `PostProcessProgress(completed, total)` becomes the outer bar of the same
  shape.
- Where the batch count is known: the request's `batchCount`
  (`shared/.../Generation.scala`).

## Notes

- sd-cpp is told apart from the runner by the runtime's engine (spec 43),
  never by sniffing the log.
- sd-cpp's sampling bar formats its rate adaptively (`s/it` or `it/s`), which
  `LogProgress` already handles. Keep that path as it is.
