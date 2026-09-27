# 02 — Model registry

**Status:** done
**Depends on:** 01

A model is one concrete weight file registered against a family, so an
architecture slot can be filled. One registration serves every architecture whose
slot names that family.

## What it does

- Models are added from inside an architecture card: expand a checkpoint slot,
  **Add Model**, and the family is pre-filled. The source is picked in one of the
  three browsers ([`03`](03-model-discovery.md)) or typed.
- A model registered under a family appears in every architecture that uses it.
- A model carries its own parameters (steps, cfg…) that override the
  architecture's defaults wherever it is assigned, and its own *removals* —
  defaults this checkpoint cannot take ([`16`](16-parameter-resolution.md)).
  Only what actually overrides something is stored, and because one model can
  fill a slot in several architectures, a value counts as redundant only when
  all of them already give it.
- Built-in models come from the reference file ([`23`](23-seeded-vendor-models.md))
  and cannot be edited or deleted — they are rewritten from the reference on
  every start, so an edit would not survive one; converted models are registered
  into the family of their source ([`25`](25-model-conversion.md)).
- A model the user added is edited in place from the same panel that added it
  (✏️ beside it in its slot): its label, its file and its parameters. The id
  is fixed — configurations assign by it, and it is the file name — so a
  rename is still a delete and a re-add.

## Shape

`Model` in `shared/src/drift/shared/Api.scala`:

```
id, familyId, label, source: ModelSource, format, parameters: Map[flag, value],
removedParameters: List[flag], builtIn
```

`ModelSource` = `HuggingFace(repo, filename)` | `Civitai(modelId, versionId,
fileId, filename)` | `Local(path)`, discriminated by a lowercase `type` key that
may sit anywhere in the object.

- CRUD at `/api/models` (`canDelete = !builtIn`); storage
  `~/.config/drift/models/<id>.json`. `PUT` is what the edit panel calls; a
  built-in is not offered it, since seeding overwrites the file on the next
  start.
- `format` derives from the file extension; ids have `/` replaced by `-` because
  the id becomes a file name.
- UI: `pages/architectures/ModelForm.scala`, `SourceTypeSelector.scala`;
  `services/ModelService.scala`.

## Notes

- `Civitai.fileId` is required and undefaulted on purpose: downloads are addressed
  by file id, so a source that cannot be fetched does not compile.
- The checkpoint dropdown filters purely by `familyId`; there is no
  per-architecture ownership of models and there should not be.
- Families are free text, so orphan models (no architecture slot names their
  family) are possible and invisible.
