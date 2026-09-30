# 48 — Soundtrack polish

**Status:** planned — big picture only; groom before building
**Depends on:** 12, 15, 42

Video models with a soundtrack (LTX 2.x, MiniMax H3, on sd-cpp or the drift
runner) hand back their audio as the model made it: its level depends on the
model and the prompt (−50 dB RMS on a one-line LTX prompt, −15 dB on a long
caption), long captions give dense, compressed voices, and the vocoders leave
hiss and a harsh upper midrange. Polish is a post-processing step on a
finished video, like the upscales of 15: it reworks the soundtrack only,
keeps the original, and can be tried again with other settings in about a
second, where a new generation takes minutes.

## What it does

- A video's detail offers **Polish sound** beside the other post-processing.
  It runs the video's soundtrack through a chain of audio filters and saves
  the result as a new gallery entry **Made from** the original; the video
  frames are copied, not re-encoded.
- The chain, each step on or off with its setting:
  - **Loudness**: normalized to a target (−16 LUFS by default, the usual
    level for online video), so clips from any model or prompt play at the
    same level.
  - **De-harsh**: a de-esser and a gentle cut around 3–6 kHz, for the
    dense voices long captions produce.
  - **Denoise**: a light spectral denoise, for the vocoders' hiss.
  - **Low cut**: a high-pass around 80 Hz, for rumble.
- A **preset** holds the chain's settings; one built-in preset (loudness,
  low cut, light de-harsh) is the default, and the last settings used are
  remembered.
- The **Original** / **Result** toggle of 15 plays either version from the
  same position, to compare by ear.
- A video without a soundtrack does not offer it.

## Shape

- **ffmpeg** on the backend's machine: the drift runner already needs it to
  encode its webm (42), and drift and its runtimes run on the same machine.
  One process per job: `-c:v copy`, the chain as `-af` (`loudnorm`,
  `deesser`, `equalizer`, `afftdn`, `highpass`), Vorbis for webm.
- A post-processing job kind beside ESRGAN and PiD (`backend/.../postprocess`),
  recorded like them: its settings in the entry's parameters, its parent the
  original entry.

## Notes

- Polish cannot give back dynamics a model compressed away: a long-caption
  LTX voice has a crest factor of about 15 dB against 20 dB for a short
  prompt's. It makes the result consistent and pleasanter, not different.
- The runner already scales a soundtrack whose peak passes full scale instead
  of clamping it (42, `Soundtrack.fitted`); sd-cpp's videos come clamped, so
  their peaks may already be flattened before polish sees them.

## Remaining

Everything; to groom first:

- Automatic or on demand: a per-configuration "polish every video" switch,
  or only the button.
- Which settings the panel shows, and their ranges; whether presets are
  editable records or a fixed list.
- `loudnorm` in one pass (fast, approximate) or two (exact, reads the track
  twice).
- Whether the gallery thumbnail and the project views show polished and
  original versions as one entry with a toggle, or two entries.
- How the Sandbox (47) offers it on a result that is not saved yet.
