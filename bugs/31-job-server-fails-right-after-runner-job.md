# Bug 31 — A job server fails to start right after another stopped on its port

**Status:** open (seen once, 2026-09-25; the retry 80 s later worked; cause not verified)
**Severity:** low (the job fails with a misleading reason; retrying works)
**Files:** `backend/src/drift/backend/session/ServerProcesses.scala` (`freePort`, `bindable`), `backend/src/drift/backend/session/JobServers.scala`

## Symptom

A PiD job on the drift runner completed at 21:28:02 and its job server on
:7860 was stopped. A PiD job on sd-cpp (`latest-rocm`) was then started at
21:28:18 and got port 7860 again. sd-server loaded the models and printed
`listening on: http://127.0.0.1:7860`. Two seconds later the job failed with
`sd-server exited with code 0 while loading`. The same request, submitted
again at 21:29:41, ran to completion on the same port.

## Suspected cause (not verified)

The runner's server closes its connections itself, which leaves sockets on
:7860 in TIME_WAIT for about 60 s. `ServerProcesses.bindable` tests the port
with a Java `ServerSocket`, which on Linux sets SO_REUSEADDR by default, so
the test passes. If sd-server's bind does not reuse the address, it fails
after the banner, and sd-server exits 0.

## Suggested fix

1. Reproduce: run a runner PiD job, then an sd-cpp one within 60 s. Check
   `ss -tan | grep 7860` for TIME_WAIT when the second one starts.
2. If confirmed, call `socket.setReuseAddress(false)` in `bindable` before
   `bind`, so a port in TIME_WAIT counts as taken and the next one is used.
3. Either way, "exited with code 0 while loading" after a "listening" line
   should say that the port could not be bound.

## Verification

- Back-to-back runner then sd-cpp PiD jobs both complete.
