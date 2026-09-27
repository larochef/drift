# 34 — Sharded safetensors models

**Status:** done in code; the download, cache status, shard ownership and
repair were exercised on a tiny sharded repository against a stub backend. A
first real run, SenseNova U1.5's official bf16 repository, is still to do
**Depends on:** 05, 23

Official releases often split a model into shards
(`model-00001-of-00008.safetensors` …) listed by a
`model.safetensors.index.json`. sd-cpp loads such a model from its index and
looks for the shards beside it (given the first shard alone, it loads only
that shard). A HuggingFace model whose file is the index is therefore the
whole set: drift downloads, weighs and checks the shards with it.

## What it does

- A model registered with a HuggingFace source on a `*.safetensors.index.json`
  downloads the index, then every shard its `weight_map` names (relative to
  the index's folder), one after the other, into the shared HuggingFace cache
  under one snapshot. Progress runs over the shards' total; a shard already in
  the cache is linked, not fetched; LFS shards are verified by sha256.
- The model is cached only when the index and every shard are readable; its
  size is the shards' total. A missing shard makes it missing again, and the
  next download fetches only what is gone.
- The on-disk view counts every shard, and the index, as used by the model.
- The HuggingFace browser and the local file browser's Models filter offer an
  index as a model file.
- Launching passes the index path to sd-cpp as any model file.
- Seeded: `sensenova-u1.5-8b-mot-bf16`, SenseNova U1.5's official repository
  (8 shards, 32.7 GiB), beside the community `Q5_K_M` GGUF.

## Shape

- `ShardedSafetensors.isIndex(filename)` (`shared/.../Api.scala`);
  `HuggingFaceFileInfo.blobId`.
- `backend/.../cache/SafetensorsIndex.scala`: `shardNames(index)` and
  `completeShards(index)`, used by `ModelCache.resolve` (presence and size, for
  HuggingFace candidates and `Local` paths alike) and
  `CacheInventory.referenceIndex`.
- `HuggingFaceDownloads.downloadSharded` / `fetchShards`;
  `downloadIntoSharedCache` takes the blob name and an optional sha256.

## Notes

- The index is a small file outside LFS: its blob is named by its git blob id,
  which is its ETag, as `huggingface_hub` names it, and it is not verified.
- An index maps every tensor, past jsoniter's default cap of 1024 map entries
  (SenseNova U1.5 has 1116): its codec raises the cap.
- A failed model download logs its reason, and the cache indicator prints it
  beside the failed tag.
- All files come from one listing's commit, so the index and its shards share
  a snapshot even when the repository moves on mid-download.
- Details and conversion (`24`, `25`) read one safetensors header; on an index
  they have nothing to show.

## Post-v1

- The browser's file row for an index could show the shards' total size.
