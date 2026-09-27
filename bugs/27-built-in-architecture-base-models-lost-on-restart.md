# 27 — Civitai base models configured on a built-in architecture vanish on restart

**Status:** open, found by reading the code during the `specs/33` UI check (not
reproduced live)
**Severity:** low — the edit works until the next start, then silently reverts

## What happens

A built-in architecture with no Civitai base model (SenseNova U1.5, Mage-Flow
before its `MageFlow` entry) shows **+ Configure Civitai base models** on its
card. Adding one saves, the LoRA install control's **+ Civitai** becomes
usable, and after a drift restart the base models are gone again.

## Cause

`ArchitectureCard` saves the base models through `onSave(a.copy(
civitaiBaseModels = …))`, a plain `PUT /api/architectures/{id}`.
`architectureEndpoints` (`backend/src/drift/backend/routes/Routes.scala`) only
guards deletion (`canDelete = !_.builtIn`); the update is stored. On the next
start `StorageService.seedFromResource` rewrites every built-in from
`reference/architectures.json`, dropping the edit.

Prompt templates already show the intended rule: their endpoints pass
`mergeUpdate = (stored, incoming) => if (stored.builtIn) stored else …`.

## Fix

Either refuse the edit — `mergeUpdate` keeping a stored built-in, and the card
not offering **Configure Civitai base models** on a built-in (the missing names
belong in `reference/architectures.json`) — or keep user base models apart from
the built-in entity so the seed does not overwrite them. The first matches how
built-ins are treated everywhere else.

## Verify

On a built-in architecture the card offers no base-model editing, and a `PUT`
changing a built-in's `civitaiBaseModels` leaves the stored entity unchanged.
