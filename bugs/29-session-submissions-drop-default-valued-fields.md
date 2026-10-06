# Bug 29 — Session image/video submissions drop every field equal to its Scala default

**Status:** fixed in code 2026-10-06 — `ServerRequests` holds the one codec (every field written) for the
session path and `NativeJobs`, images and videos; unit-tested (`ServerRequestsTests`), not run live
**Severity:** medium (a request silently runs with the server's launch-flag values instead of what the user asked)
**Files:** `backend/src/drift/backend/sdserver/GenerationSubmissions.scala` (~lines 64–80, ~119), `shared/src/drift/shared/Generation.scala` (~lines 171–178 and the video codec)

## Symptom

In a session, asking for a value that happens to equal the case class default is ignored: sd-server applies the request on top of its **launch-flag** defaults, so a missing field means "whatever the configuration launched with". Examples on Krea 2 (launched with `--steps 4 -W 1024 -H 1024 --cfg-scale 1.0`):
- 20 steps requested → 4 steps run (`sample_steps` default is 20, so it is omitted);
- 512 × 512 requested → 1024 × 1024 comes back (`width`/`height` default 512);
- CFG 7 requested → 1.0 runs (`txt_cfg` default 7.0); strength 0.75, batch count 1 likewise.

## Root cause

`GenerationSubmissions` serialises with `writeToString(parameters)`, which picks up the shared codec in `ImageGenerationParameters`'s companion:

```scala
given JsonValueCodec[ImageGenerationParameters] = JsonCodecMaker.make(
  CodecMakerConfig
    .withDiscriminatorFieldName(None)
    .withFieldNameMapper(JsonCodecMaker.enforce_snake_case)
)
```

jsoniter-scala's default (`transientDefault = true`) omits every field equal to its default value. `NativeJobs.scala` (~lines 66–76) already knows this and defines its own codec with `.withTransientDefault(false)` ("Every field written, even one equal to its default: a 512 px tile must not fall back to the server's own width"), but that given is private to `NativeJobs`, so the session path doesn't use it. The same applies to `VideoGenerationParameters` (line ~119).

## Suggested fix

Write every field on the session path too: either move `.withTransientDefault(false)` into the shared companions' codecs (check first that nothing relies on the omission, e.g. `hires`/`vaeTilingParams` are `Option`s and stay omitted when `None` since `transientNone` is separate), or give `GenerationSubmissions` the same explicit codecs as `NativeJobs` for both image and video parameters.

## Verification

- Unit: serialise `ImageGenerationParameters(prompt = "x")` with the codec the session path uses; the JSON must contain `"width":512`, `"height":512`, `"sample_steps":20`, `"txt_cfg":7.0`.
- Live: launch Krea 2 (4 steps, 1024²), request 20 steps at 512 × 512 in a session; the sd-server log must show 20 steps and a 512 × 512 image.
