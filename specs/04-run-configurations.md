# 04 — Run configurations

**Status:** done
**Depends on:** 01, 02

A run configuration is an architecture, a registered model for each of its
checkpoint slots, parameter overrides and default LoRAs: a complete, launchable
description of one `sd-server` or `llama-server` invocation. Everything downstream
consumes it — the cache resolves its models to files ([`05`](05-model-cache-and-downloads.md)),
the launcher turns it into argv ([`07`](07-launch-and-supervision.md)).

## What it does

- The Models page's **Run configurations** tab lists configurations of both
  runners, most recently used first, with the same search and tag filters as the
  architectures tab (search also matches the architecture's name).
- **New run configuration**: label, architecture, a model per slot (dropdowns
  filtered by family), override parameters, default LoRAs
  ([`28`](28-configuration-loras.md)). An edit card offers the same fields.
- Each slot has **+ Add model**: the architecture page's `ModelForm`, its
  source browser opened on the configuration's architecture, then its details
  modal. The saved model is registered in the slot's family and assigned to the
  slot once the server has stored it. LoRAs install from the default-LoRA
  picker's `LoraInstallButton` the same way.
- A card carries a `ready` / `incomplete` tag. When blocked it names each reason:
  unknown architecture, an unassigned required slot, a model no longer registered,
  weights not downloaded (with a **Download missing** button and sizes, counting
  only the weights not already queued or downloading; when all of them are, a
  line says how many are downloading and queued instead), or an unreadable
  file.
- A collapsible command-line preview with a **Copy** button shows the exact argv,
  resolved for the runtime selected in the launch control, plus any resolution
  notes ([`16`](16-parameter-resolution.md)).
- **Launch** on a chosen runtime, **Stop**, and a status tag (starting, ready on
  a port, failed, stopped); the launch panel opens beside the list.
- Deleting a referenced model degrades the configuration to incomplete rather
  than corrupting it; re-registering the model restores it.

## Shape

`RunConfiguration` in `shared/src/drift/shared/Api.scala`:

```
id, label, architectureId, assignments: Map[checkpointName, modelId],
overriddenParameters: Map[flag, value], createdAt, lastUsedAt,
loras: List[ConfiguredLora(loraId, strength)]
assistantTemplateId: Option[String] (the assistant system template this
configuration prefers over its architecture's default — 32)
```

- CRUD at `/api/run-configurations`; storage
  `~/.config/drift/run-configurations/<id>.json`. `lastUsedAt` is set by every
  launch.
- `CommandLine.blockers` (shared) answers `List[LaunchBlocker]`:
  `UnknownArchitecture`, `UnassignedCheckpoint`, `MissingModel`,
  `WeightsNotCached`, `WeightsUnreadable`. `CommandLine.resolve` builds the argv and
  the notes; the preview and the launcher both call it.
- UI: `frontend/.../pages/inference/{RunConfigurationsPage,RunConfigurationCard,
  RunConfigurationEditCard,RunConfigurationForm,CheckpointAssignments}.scala`,
  hosted by `pages/models/ModelsPage.scala`; `services/RunConfigurationService`.

## Notes

- A configuration does not name a runtime: the runtime is chosen at launch, so
  anything that depends on the build is resolved then, never stored here.
- Overrides replace defaults by flag key; an override cannot *remove* an
  architecture default (there is no "drop `--mmap`" — it would need an optional
  value and a remove control).
- Generating from this page is free play: nothing persists unless kept
  ([`22`](22-free-play-and-scratch-generations.md)).
