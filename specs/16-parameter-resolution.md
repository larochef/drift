# 16 — Parameter resolution: architecture, model, runtime, configuration

**Status:** done
**Depends on:** 02, 04, 06, 17

Every flag drift puts on a command line comes from one of four layers, merged in
a fixed order, with the build's vetoes applied on top and reported. A checkpoint's
numbers are not its architecture's (Krea 2 Raw wants 52 steps at cfg 3.5, Krea 2
Turbo 4 at 1.0, under the same slots), and a build's flags are not any model's
(`--diffusion-fa` is unavailable on Vulkan). Since a configuration does not name
a runtime, runtime-dependent values are resolved when the argv is built, never
stored.

## The layers

```
architecture.defaultParameters        true for every model of this shape
  ↓ overridden by
model.parameters                      what THIS checkpoint wants (every assigned
model.removedParameters               model contributes, in slot order; later wins)
  ↓ overridden by
runtime rule defaults                 what THIS build wants
  ↓ overridden by
runConfiguration.overriddenParameters the user's explicit choice
runConfiguration.removedParameters
  ↓ filtered by
runtime rule vetoes                   what THIS build cannot do — each a note
```

A layer does not only override a value: a model and a run configuration can each
**remove** a flag — keep it off the command line whatever a layer below set. An
override cannot say this, because `""` is the convention for a valueless flag
that *is* passed. A removal overrides like a value, so a layer above sets the
flag again if it wants it.

## What it does

- Two models with different step counts coexist under one architecture and each
  produces its own command line.
- Launching on a Vulkan runtime drops `--diffusion-fa` (and `--attn-scale`
  beside it) and says so, naming the
  layer that set it, in the configuration card's preview ("How this command
  line was resolved") and on the session (`Session.parameterNotes`); the same
  configuration on ROCm keeps it.
- A checkpoint that cannot take one of its architecture's defaults says so on
  the model (`--diffusion-fa` on a family member that breaks under it), and a
  configuration can drop any of the three layers below it for one run — without
  editing the architecture, which a built-in does not allow anyway.
- No flag disappears from a command line without a note: a removal of something
  a lower layer had set is reported like a veto, naming both layers. Removing a
  flag nobody set is silent — there is nothing to explain.
- A record carries only what it actually changes. An override repeating the
  value a lower layer already gives, and a removal of a flag nothing below
  sets, are dropped when the entity is stored — so a file says what was
  decided and the effective list stays computed. The cost is that a no-op
  override does not *pin* a value: a configuration that agreed with its
  architecture follows its model when a later assignment moves it. That is the
  same thing an override is for, so agreeing has to mean agreeing.
- The editor offers the flags set below as chips, each naming the layer that
  set it and carrying both actions: ✎ overrides it here, 🚫 removes it. A
  removal is picked, never retyped — a misspelled flag removes nothing while
  looking like it did — and the layer is half the answer: a model that sets
  `--attn-scale` to the number its architecture already had is invisible
  without it, which is how François came to “remove” one by overriding it with
  its own value (2026-09-20).

## Shape

- `CommandLine.resolveParameters(configuration, architecture, models, rule):
  ParameterResolution(arguments, notes)` in `shared/.../CommandLine.scala` folds
  the layers into `Map[flag, (Option[value], ParameterLayer)]` — `None` being a
  removal — and applies the vetoes. The fold is in layer order rather than a
  merge of maps: what a flag was worth before a removal is what names the layer
  in its note.
  `resolve` answers argv and notes together; `argv` is `resolve` without the
  notes. Both, and the forms, go through one private `resolveSources`, which
  answers `List[ResolvedParameter(flag, value, layer)]` — the layer is what
  the UI needs and the argv throws away.
- `CommandLine.inheritedParameters(configuration, architecture, models, rule)`
  is `resolveSources` with the configuration's own `overriddenParameters` and
  `removedParameters` emptied: what a form has to show as “already set below”.
  The form cannot show something the launch would not do, because it is the
  same fold. `rule` is `None` from a form: the runtime is picked at launch, so
  its layer is left out rather than guessed. Covered by
  `backend/test/.../ParameterResolutionTests.scala`, the view included, and the
  pruning by `ParameterOverridesTests.scala` — subtractive rules are the kind
  that fail silently, so each case is a test: what must go, and what must
  survive.
- `ResolutionNote` = `Dropped(flag, from, reason)` | `AddedByRuntime(flag, value)`
  | `Removed(flag, from, by)` (`shared/.../RuntimeRules.scala`). Overrides
  between layers are *not* noted: one note per override would drown the one that
  matters. A removal is, because nothing else on the card says the flag is gone.
  The card's heading over them is "How this command line was resolved" — the
  runtime is no longer the only thing that changes it.
- `ParameterOverrides` (`shared/.../ParameterOverrides.scala`) is the pruning:
  `prunedParameters(parameters, removed, inherited)` for one layer,
  `prunedConfiguration` and `prunedModel` for the two entities. The routes call
  them through `endpointsFor`'s `onCreate` and `mergeUpdate`
  (`backend/.../routes/Routes.scala`), so every client stores the same shape and
  no form has to remember the rule; the services keep the *saved* copy the
  server answers with, not the one they sent.
  - A configuration is pruned against its architecture and its assigned models,
    and never against a runtime: a runtime is chosen at launch and changes under
    a configuration nobody edited, so an override matching one build's default
    is still a decision about every other build.
  - A model is pruned against **every** architecture that can assign it, not the
    one whose card it was edited from: a family's models are shared
    ([`02`](02-model-registry.md)), so a value is redundant only when all of
    them already give that flag that value, while a removal is worth keeping as
    soon as one of them sets the flag. A model no architecture can assign keeps
    everything — there is no layer below it to compare against.
  - Removals are applied to a layer's own values before pruning, as resolution
    does: a flag a layer both sets and removes is never passed, so the value is
    not worth keeping either.
- `ParamEditor` (`frontend/.../components/ParamEditor.scala`) is one list for
  both: each row is a flag with a value or a 🚫 that makes it a removal, and
  the typed value survives the toggle. Its `inherited` attribute is the
  `ResolvedParameter` list above, rendered as the chips. `allowsRemoval = false` for the
  architecture form — the bottom layer has nothing below it to remove, so a
  removal there is a row that should not be written.
- `RuntimeRule(tool, backend, engine, defaults, unsupported:
  List[UnsupportedFlag(flag, reason)])`, shipped as
  `backend/resources/reference/runtime-rules.json`, served read-only at
  `GET /api/runtime-rules`, never seeded into the config. The engine
  (`RuntimeEngine`, `specs/43`) tells the drift runner's images runtime apart
  from sd-cpp's ROCm builds, which share its tool and backend. Two rules:
  - sd-cpp on Vulkan vetoes `--diffusion-fa` and, with it, `--attn-scale`,
    whose only effect is inside flash attention;
  - the drift runner (sd-cpp tool, ROCm) vetoes `--attn-scale`. It takes
    `--guidance` (FLUX.2 [dev]'s distilled guidance) and notes it ignored
    for models without a guidance embedding. The runner itself
    still refuses any flag it doesn't know, so a new sd-server flag shows up
    as a missing rule rather than being ignored.
- `RuntimeRule.forRuntime(runtime, rules)` picks the rule by `(tool, backend,
  engine)`, for the form's preview and for `LaunchArguments` (backend), which
  applies the rule of the runtime actually launched.

## Notes

- A rule is written only after the failure is observed on a real build: a wrong
  veto silently removes a flag that would have worked.
- `--diffusion-fa` stays on the architecture, not the runtime: it is a real
  per-family choice wherever the build supports it, and the veto is what protects
  the launch.
- Frontend loader order: `_rules` is set before `_runtimes` in the same observer,
  and the page pairs `runtimes.combineWith(rules)` as one source, otherwise the
  first preview resolves against an empty rule list.

## Post-v1

- Notes for known-bad model/runtime pairings (int8 weights run on the CPU under
  sd-cpp on ROCm and Vulkan).
- Per-runtime overrides of the built-in rules for adopted builds; the originating
  layer shown beside each parameter in the form.
