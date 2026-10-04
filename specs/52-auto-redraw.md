# 52 — Auto redraw: the assistant reads the picture and sets the redraw

**Status:** planned — big picture only; the tests come first, groom before building
**Depends on:** 20 (assistant), 27 (redraw), 32 (prompt library), 45 (redraw steps and reference), 11 of 42 (vision on the runner)

A redraw asks its user for a restoration prompt, a strength, a softening, and
which part of the picture to repaint. Every picture is different and has its
own trouble — soft, noisy, over-sharpened, a broken hand in an otherwise fine
frame — and choosing among those is more than a user should have to think
about. **Auto** hands the choice to the assistant: one call, on the whole
picture, downscaled, before the redraw starts. It answers with what the
redraw should be, the form shows it, and the user starts it or corrects it.

It is not a caption. 45 measured that describing the content changes nothing
once the model has its reference, and that automatic captions of close crops
are wrong often enough to hurt. What the assistant is asked is what kind of
repair the picture needs, which is a judgement on the whole picture — the
kind a vision model is good at.

## What it does

- The redraw panel's prompt has an **Auto** mode beside the templates, the
  default when an assistant that reads images is available. Without one the
  panel is as today, on the default template.
- Auto makes **one call** to the assistant with the picture to redraw,
  downscaled, and the list of what it may choose from. The answer fills the
  form:
  - the **restoration template** among those of the library (32), with a
    line saying why;
  - **instructions** of its own for this picture, appended as the field is
    today, when the picture has something no template names;
  - **settings**: the strength, and whether to soften first;
  - **regions**: the parts worth redrawing when the rest is fine, as boxes
    on the picture, each with its reason — or the whole picture.
    This is the repair of what an earlier step got wrong: a nipple with a
    strange shape after an upscale and a repaint, an eye an ESRGAN pass
    deformed. Redrawing that tile alone, or the few that hold it, should put
    it right without touching the rest.
- Nothing runs unseen: the form shows what was chosen and why, every field
  stays editable, and the redraw starts on the user's click. Asking again is
  one click; the answer is kept with the form, not recomputed on every
  change.
- The choice is recorded on the result (`Derivation`), with the fact that it
  was the assistant's.
- Never per tile: the call is once per redraw, whatever the picture's size.

## Shape

- The assistant is the one drift already has (20), through a model that
  reads images (the runner's vision, 42 step 11, or llama.cpp with an
  mmproj). A prompt of its own kind in the library (`RedrawDiagnosis`, say),
  built-in and editable like the others.
- The answer is **structured**: a template id out of the list it was given,
  numbers within the ranges it was given, boxes in fractions of the picture.
  An answer that does not parse, names a template that does not exist or
  steps out of a range falls back to the default template and says so.
- The picture is sent at a size the model reads well and that still shows
  the trouble (to be measured: softness and grain disappear when a 16k
  picture is brought down to 1024).
- **Memory**: the assistant and the redraw model are both large. The call
  comes before the redraw's server starts; whether the assistant is then
  unloaded, and what a live chat session does to that, is to be decided on
  measurements.

## Tests to run before building

Every test that redraws judges two things, not one: the redrawn part on its
own, and **how it sits in the whole picture**. A tile in the middle of the
picture, redrawn with what the assistant chose, must still join its
neighbours — no seam, no step in colour, sharpness or grain against the
tiles around it, redrawn or not. A choice that makes one tile better and the
picture worse is a failure. So each run is looked at twice: the tile at
full size, and the whole picture with the tile pasted back.

And a redraw of the **whole** picture has a result to reach, not only one
to avoid: more detail and more sharpness everywhere than the source had. A
choice that keeps the picture coherent and leaves it as soft as it was has
not done the job either. Both are judged on every whole-picture run:
coherent across tiles, and visibly sharper and more detailed than the source.

Each on the study's subjects (`~/dev/redraw-experiments`, 45) plus new
purpose-made ones, since the point is that pictures differ: a soft portrait,
a noisy night scene, an over-sharpened landscape, an illustration, a picture
with one broken region (hand, text on a sign), pictures an upscale then a
repaint left with a malformed nipple or a strange eye (the defects seen most
often), a picture that needs nothing.

1. **Does the choice matter at all?** The same picture redrawn with each
   restoration template, and with the default template at several strengths
   and with and without softening. If the results do not differ visibly, as
   captions did not in 45, Auto reduces to the regions and the rest of this
   spec is dropped. This test decides whether anything else is run.
2. **Can the assistant diagnose?** For each subject, its answer against a
   judgement written beforehand (what is wrong, which template, which
   strength). Several vision models: Qwen 3.6, Qwen 3.8 Flash Next, the 27B.
   Count agreements, and note the confident wrong answers.
3. **Is it stable?** The same picture asked five times, and at two seeds:
   the template and the strength should not change from one call to the
   next.
4. **At what size does it still see the trouble?** The same subject sent at
   512, 768, 1024 and 1536 on the long side, and a 4k and a 16k picture
   brought down to each: where softness, grain and small broken regions stop
   being named.
5. **Regions.** On the subjects with one broken part: are the boxes on it,
   how tight, and how often does it invent trouble in a clean picture. Boxes
   compared with ones drawn by hand; the redraw run on both.
6. **Its own instructions.** Redraws with the assistant's added instructions
   against the template alone, on the subjects where it wrote some: better,
   same, or worse — instructions that name content are the captions of 45
   again and may hurt.
7. **Auto against a person.** The whole chain on each subject, against the
   same subject set by hand in a reasonable time, and against the untouched
   default. Judged by eye, blind to which is which.
8. **The structured answer.** How often it parses, across the models, with
   and without a grammar; what the fallback costs.
9. **Cost.** The call's time on this laptop for each model and size, and the
   memory beside each redraw model: whether both fit loaded, or the swap's
   time when they do not.
10. **A picture that needs nothing.** Whether it says so, rather than
    prescribing a redraw.

## Open questions

- **How much is left to choose** after test 1: template, strength,
  softening, regions — any of them may turn out not to matter.
- **Which assistant model** by default, and what Auto does when the
  configured assistant cannot read images.
- **Regions as boxes or as a mask**, and whether several regions are one
  job or several.
- **Edit** (39) and **Upscale** are out of scope. Edit comes next, once
  redraw's Auto is right; nothing here is built with either in mind.
