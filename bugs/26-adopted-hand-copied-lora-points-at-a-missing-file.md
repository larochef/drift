# 26 — An adopted hand-copied LoRA points at a file that does not exist

**Status:** fixed with `specs/33`, checked by adopting a hand-copied file on a
stub backend: `LoraFile` stores
its on-disk `fileName`, and an adopted hand-copied file keeps its own name with
a disk source. Found by reading the code, never reproduced live
**Severity:** medium — the only way to use a LoRA that is not on Civitai
silently produces a LoRA that cannot load

## What happens

Copy `foo.safetensors` into a folder under `~/.cache/drift/loras/<architecture>/sfw/`,
then **Adopt** it from Model Cache → LoRAs. The LoRA appears on the
architecture card, but its file shows as not downloaded, a download is queued
that fails (no Civitai version to fetch from), and applying it sends sd-cpp a
path that does not exist.

## Cause

`LoraManager.adopt` (`backend/src/drift/backend/lora/LoraManager.scala`) splits
on-disk names as `<fileId>-<fileName>`; a name without a numeric prefix "keeps
its whole name on both sides":

```scala
case _ => (diskName, diskName)
```

so `fileId = fileName = "foo.safetensors"`. Every path is then rebuilt by
concatenation in `Lora.storagePathOf` and `Lora.requestPathOf`
(`shared/src/drift/shared/Loras.scala`):

```scala
s"$folderRelativePath/${file.fileId}-${file.fileName}"
```

giving `foo.safetensors-foo.safetensors`, which is not on disk. `queueDownload`
then sees a missing file and tries to fetch it with an empty `versionId`.

## Fix

Either rename the file to the stored shape on adoption (`<fileId>-<fileName>`
with a synthetic id, e.g. `local`), or store the on-disk name on `LoraFile`
instead of rebuilding it. The second is what non-Civitai LoRA sources need
anyway (a HuggingFace file has no Civitai `fileId`), so it belongs with that
work if it happens soon.

## Verify

Adopt a folder holding a hand-copied `foo.safetensors`: the LoRA shows its file
as present, no download is queued, and a generation applying it logs sd-cpp
loading `…/sfw/<id>/foo.safetensors`.
