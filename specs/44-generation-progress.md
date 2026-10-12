# 44 — Generation progress: batches and the runner's own progress

**Status:** done
**Depends on:** 15 (post-processing jobs and their progress), 22 (free play), 42 (the drift runner), 43 (a runner per configuration)

A generation's progress shows what the job is doing and how far it has
got, as the tiled post-processing does. A batch has a bar for the whole batch
and a bar for the image being made. sd-cpp's progress is read from its log;
the drift runner reports its own on the job drift polls.

## What it does

- **Two bars for a batch.** A generation of several images shows:
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
- **On the drift runner** a generation's job says where it is: the
  document `GET /sdcpp/v1/jobs/{id}` answers carries `progress` while the job
  runs (`JobProgress`, `protocol/.../Jobs.scala`) — the bar (`JobBar`: its
  phase `preparing`, `sampling` or `decoding`, what is done of what, the
  rate), the sampling pass of a two-pass model, the image within the batch,
  the stage within a video. The runner writes it and its log line in one
  place (`diffusion/ProgressReport`), so the two never disagree; drift shows
  the report as it polls (`GenerationMonitor` → `SessionManager.report`) and
  drops it when the job ends. A post-processing job shows the report of the
  server job it waits for the same way (`NativeJobs`' `onProgress` →
  `PostProcessJobs.report`), tile after tile. The log's bars remain what
  drift reads where no job is polled: a launch's loading.
- **On sd-cpp** the log bar is the source
  (`LogProgress`). The batch bar counts sd-cpp's sampling bars, a new one
  starting at step 1 for each image.

## Left open

- One progress component for sessions and post-processing jobs, instead of
  `LogProgressView` and `PostProcessJobList`'s own bar.
- The runner still prints sd-cpp's bar format beside its report, and drift
  still parses it for the runner's sessions when no job reports.

## Notes

- sd-cpp is told apart from the runner by the runtime's engine (spec 43),
  never by sniffing the log.
- sd-cpp's sampling bar formats its rate adaptively (`s/it` or `it/s`), which
  `LogProgress` already handles. Keep that path as it is.
