# 11 — Status WebSocket

**Status:** done
**Depends on:** 07, 08

One push channel replaces status polling. Commands and CRUD stay REST; the
socket only mirrors state the backend changes on its own — sessions, jobs,
generations — so an idle app makes no requests.

## What it does

- Session transitions, download and install progress, generation updates,
  post-process and conversion jobs arrive on every page without user action,
  including the global downloads panel pinned at the sidebar bottom.
- After a backend restart the app reconnects by itself and shows the true
  (possibly empty) state, not stale jobs.
- Derived behaviours still fire off the pushes: a completed download reloads
  the cache view, a settled install refreshes the runtime list.

## Shape

- Path `GET /api/status` (`statusSocketPath` in `shared/.../Status.scala`),
  a plain WebSocket, not a tapir endpoint: the shared codec and the path
  constant are the whole contract.
- `StatusUpdate(sessions, downloads, loraDownloads, upscalerDownloads,
  runtimeInstalls, generations, postProcessJobs, conversions)`, every field
  optional. A present field is a full snapshot of that topic; an absent field
  means unchanged since the connection's previous message. `generations` is a
  map keyed by session id and diffs per session.
- The first message after (re)connect carries every topic, empty or not: that
  is the resync, and it is how a client learns what is gone.
- Backend `routes/StatusRoutes.scala`: each connection samples the managers
  twice a second (`Flow.tick` + `mapStateful`) and sends only topics whose
  value changed. The managers' `list` methods sort, so equality means really
  unchanged.
- The Vite dev server must proxy the upgrade (`ws: true` in `vite.config.js`):
  without it the socket never connects behind `:5173` and every live update is
  silently dead, which is how a finished runtime install kept showing as queued
  (François, 2026-09-20).
- Frontend `services/StatusSocketService.scala`: a raw `dom.WebSocket` with a
  2-second reconnect loop hung off `onclose`, one typed `EventStream` per
  topic. Mounted once, by `AppShell`. Consuming services subscribe inside their
  own `effects`, so a topic is listened to only while a page has that service
  mounted; pages still seed through the REST loads they push on mount.

## Notes

- Full snapshots, not deltas: the services replace their maps wholesale, so
  the existing reducers and their derived events work unchanged and there is
  no delta bookkeeping to get wrong.
- tapir's netty-sync server logs an error if the outgoing flow does not
  complete when the client closes, and a tick flow never completes on its own:
  `incoming.drain().merge(updates, propagateDoneLeft = true)` ties the two
  lifetimes together.
- A message that does not decode is caught, logged to the console and shown:
  the shell carries a notice with a **Reload** button (`StaleBundleNotice`,
  `StatusSocketService.undecodable`), cleared as soon as a message reads again.
  Every topic travels in one message, so a page that cannot read one reads none
  of them — it goes quiet while looking merely idle, the socket stays up and
  nothing recovers by itself, which is how a finished runtime install can sit
  there saying "queued" (François, 2026-09-20). Throwing out of `onmessage` was
  loud only to a console nobody had open.
- Mounting the service twice opens two sockets (the bugs/15 class); it belongs
  in `AppShell` only.
- What stays REST: all commands and CRUD, the per-page list loads on mount,
  and the generation panel's bounded capabilities retry until the first answer.
- One fallback poll, and only one: a runtime install moves by push alone, and
  a silent socket froze it on the state its POST answered with — so while an
  install is active and nothing has arrived about installs for
  `RuntimeService.InstallPollMillis`, the list is asked for over REST. A
  pushing socket keeps the timestamp fresh, so a working socket polls nothing.
