# Bug 19 — Civitai's default listing filter hides models the site shows (including SFW-flagged ones)

**Status:** fixed
**Severity:** low (listing incomplete; every hidden model is still reachable on civitai.com)
**Files:** `backend/src/drift/backend/routes/CivitaiRoutes.scala`

## Context

Follow-up to bugs/18. After fixing the `q`/`query` parameter and sending the auth
token, the browser still listed fewer models than civitai.com: for base model
`MiniMax H3`, the site showed 18 checkpoints, drift 14.

## Symptom

`GET /api/v1/models?types=Checkpoint&baseModels=MiniMax H3` (authenticated) returns
14 items. The same request with `nsfw=true` returns 21 — a strict superset. The site
shows whatever the account's browsing-level settings allow, which landed between the
two (18).

## Root cause

When the `nsfw` query parameter is absent, the API applies a default filter that
drops not only X-rated models but also some models whose own flags are SFW — e.g.
`MiniMax-H3 comfy-native fl2va` (id 2860330, `nsfw: false`, `nsfwLevel: 2`) was
among the hidden ones, presumably because *some* version or image of it carries a
higher level. Sending the token does **not** make the API apply the account's
browsing settings; only the explicit parameter changes the set.

## Fix

`CivitaiClient.search` always sends the `nsfw` parameter explicitly, never leaving
it absent — the absent-param in-between is the broken state. The value comes from an
**Include NSFW** checkbox in the `CivitaiBrowser` (default checked, re-searches on
toggle): checked → `nsfw=true` (everything, 21 for MiniMax H3), unchecked →
`nsfw=false` (the SFW set, which matches the old absent-param count of 14).

## Verification

```
curl 'http://localhost:4321/civitai-search?types=CHECKPOINT&baseModels=MiniMax%20H3&limit=100&nsfw=true'
```

must return 21+ items, including `H3 Eros MAX GGUF` and
`MiniMax-H3 comfy-native fl2va - w4a8 / nvfp4 / int8 / mxfp8, ComfyUI 0.32+`,
which the default filter hid; with `nsfw=false` it must return the 14-item SFW
set. In the UI, unchecking Include NSFW shrinks the MiniMax H3 listing from 21
to 14.
