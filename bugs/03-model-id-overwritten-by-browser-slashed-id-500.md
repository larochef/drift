# Bug 3 — Model ID overwritten by source browser → slashed ID → save fails with 500 (and the Create pipeline dies)

**Status:** fixed
**Severity:** high (creating a model via the HF/Local browser fails; via Civitai it saves under the wrong id)
**Files:** `frontend/src/drift/frontend/pages/architectures/SourceTypeSelector.scala`, `frontend/src/drift/frontend/pages/architectures/ModelForm.scala`, `backend/src/drift/backend/storage/StorageService.scala`, `frontend/src/drift/frontend/services/ModelService.scala`

## Context

drift persists entities as JSON files: `StorageService.save(entityType, id, value)` writes `~/.config/drift/<entityType>/<id>.json` with `Files.writeString` (see `StorageService.scala` lines 48–57). The model "Add Model" form is `ModelForm`; it embeds a `SourceTypeSelector` for choosing HuggingFace / Civitai / Local sources.

## Symptom

Adding a model on an architecture card:

- **HuggingFace**: pick a model via the "Browse" dialog → **Add** → `POST /api/models` returns 500, the model doesn't appear, and *further* create attempts on that page silently do nothing until you navigate away and back.
- **Local**: same 500 (the entity id becomes the full file path).
- **Civitai**: saves, but the model's id is the numeric Civitai model id, not whatever the user typed in the "Model ID" field.

## Root cause

`ModelForm` shares one `Var` between the *entity* id and the *Civitai source* model id:

```scala
// ModelForm.scala (lines ~17, ~24–40)
private val modelId = Var(initialModelId)          // entity id
private val selector = SourceTypeSelector(
  ...
  modelIdParam = modelId                           // shared with the selector
)
```

```scala
// SourceTypeSelector.scala (lines ~23–26)
val modelId = modelIdParam match {
  case null => Var(initialModelId)
  case v    => v                                    // same Var
}
```

The browser `onSelect` handlers then overwrite that shared Var:

```scala
// SourceTypeSelector.scala
// Local browser, line ~174:
onSelect = { path => localPath.set(path); modelId.set(path); ... }      // id := "/home/user/models/x.safetensors"
// HuggingFace browser, line ~189:
onSelect = { (repoVal, file) => repo.set(repoVal); filename.set(file); modelId.set(repoVal); ... }  // id := "owner/repo"
// Civitai browser, line ~205:
onSelect = { (id, fId, fileName, label) => modelId.set(id); ... }       // id := numeric civitai id
```

`ModelForm.snapshot()` (line ~56) then saves `Model(mid, ...)`. For HF/Local the id contains `/`, and:

```scala
// StorageService.scala (lines ~48–57)
def save[T](entityType: String, id: String, value: T)(using Encoder[T]): T = {
  val dir = basePath.resolve(entityType)
  Files.createDirectories(dir)
  Files.writeString(dir.resolve(s"$id.json"), ...)   // "models/owner/repo.json" — parent dir "models/owner" missing
}
```

`Files.writeString` does not create parent directories → `NoSuchFileException` → 500.

Aggravating factor: the frontend uses `toClientThrowErrors` (see `ModelService.scala`), so the rejected promise makes the Create `EventStream` emit an error; with no error handler on the stream, Laminar ends the subscription — the whole Create pipeline for that page is dead until the element remounts.

## Suggested fix

Do all of the following (defense in depth):

1. **Stop sharing the Var.** In `SourceTypeSelector`, keep a *separate* Var for the Civitai model id (don't accept/reuse `modelIdParam`), or stop passing the form's entity-id Var in from `ModelForm`. The entity id must be user-controlled and independent of the source.
2. **Don't overwrite the entity id in `onSelect`** for HF/Local (only set `repo`/`filename` or `localPath`).
3. **Backend hardening** in `StorageService.save`: create the parent directory of the target file before writing (e.g. `Files.createDirectories(dir.resolve(s"$id.json").getParent)`), or reject ids containing `/` with a 400. (This also mitigates Bug 10.)
4. Optionally sanitize the id (replace `/` with `-`) in `ModelForm.snapshot()`.

## Verification

1. Architectures page → any card → "Add Model" → choose HuggingFace → Browse → pick a model → Add.
2. The model must appear in the list; `~/.config/drift/models/` must contain its file with a sane, single-segment name.
3. Repeat with the Local file browser and the Civitai browser (Civitai: the id the user typed must be kept).
4. Create a second model after a failure — the form must still work (no dead pipeline).
