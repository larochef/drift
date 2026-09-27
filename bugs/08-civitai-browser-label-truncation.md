# Bug 8 — Civitai browser truncates model labels that contain "/"

**Status:** open
**Severity:** low (wrong label saved for models whose name contains `/`)
**Files:** `frontend/src/drift/frontend/components/CivitaiBrowser.scala`

## Context

`CivitaiBrowser` is the modal used from `SourceTypeSelector` to pick a Civitai model + file. On file selection it calls `onSelect(modelId, fileId, fileName, label)`, where `label` becomes the model's `label` field in `ModelForm`.

## Symptom

Select a Civitai model whose **name** contains a `/` (e.g. `"Foo/Bar"`). The saved model label is only `Foo` (the first segment after the numeric id).

## Root cause

The selected model is packed into one string as `"<id>/<name>"` and later unpacked with an unbounded split:

```scala
// CivitaiBrowser.scala renderModelCard (~line 289)
selectedModel.set(s"${model.id}/${model.name}")
```

```scala
// CivitaiBrowser.scala renderFileEntry (~line 374)
val modelName = selectedModel.now().split("/")(1)     // stops at the FIRST '/'
val label = if (modelLabel.now().isEmpty) modelName else modelLabel.now()
```

Note `modelLabel` is actually set correctly on card click (`modelLabel.set(model.name)`, ~line 291), so the fallback is only hit when the label Var is empty — but the fallback itself is still wrong, and the packed-string scheme is fragile.

## Suggested fix

Either:

- use a bounded split: `selectedModel.now().split("/", 2)(1)`, or
- drop the fallback entirely and rely on `modelLabel` (it is always set on card click), or
- store `selectedModel` as `id` and keep the name in `modelLabel` only.

(For reference, the newer `SearchBrowser.scala` trait already handles this correctly with `split("/").drop(1).mkString("/")` — although that trait is currently dead code, see the dead-code notes in `bugs/README.md`.)

## Verification

1. In an architecture card's "Add Model" → Civitai → browse for a model whose name contains `/`.
2. Pick a file; the label field (and the saved JSON `label`) must contain the full name.
