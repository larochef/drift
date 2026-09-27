# 22 — A completed video result blows the decode buffer, and the poll loop hid it

**Status:** fixed (2026-09-03)

## Symptom

A real video generation completed normally on sd-server (manual poll showed
`status: completed` with the full result), but drift never marked it
completed — it kept polling for ten minutes and then failed with the
misleading "sd-server no longer knows the job". Smaller video jobs
(~1.5 MB webm) worked; the failing one's job response was 5.9 MB.

## Two stacked causes

1. **jsoniter's default `maxCharBufSize` is 4M chars.** A completed video job
   is one giant base64 string; a 3-second webm already exceeds the cap, so
   `readFromString[NativeJob]` threw
   `too long string exceeded 'maxCharBufSize'` on every poll of the completed
   job.
2. **The poll loop reset its failure counter on HTTP 200 *before* decoding**,
   and its catch block logged nothing. A decode failure that repeats forever
   therefore never accumulated the five failures that trip the safety valve:
   the loop silently re-fetched the completed 5.9 MB result every second for
   600 seconds until sd-server's completed-job TTL evicted it — at which
   point the 410 surfaced as "no longer knows the job".

Reproduced deterministically with the stub sd-server inflating its video
payload to 6 MB; after the logging fix the true error appeared verbatim.

## Fix

- The native-job decode uses `ReaderConfig.withMaxCharBufSize(256M chars)`.
- The failure counter resets only after the document is decoded *and*
  applied, and every poll failure is logged (`poll failed (n/5): …`).
- Failure messages carry only the exception's first line, truncated —
  jsoniter appends a multi-line hex dump that was reaching the UI.

Verified: the same 6 MB stub job now goes Queued → Completed with the webm
persisted.
