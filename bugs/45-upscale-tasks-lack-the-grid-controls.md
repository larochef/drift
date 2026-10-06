# Bug 45 — The upscale tasks' grid is not drawn, and they lack the grid controls of a redraw and an edit

**Status:** fixed in code 2026-10-06, as the one refactoring asked for (Done, below) — compiled, gate green,
not seen in a browser (asked by François 2026-10-05: "we could have a grid for the upscale tasks too, as for
the edit and redraw since it is a tiled job too")
**Severity:** medium (the checkbox is there and draws nothing; the controls are a missing feature)
**Files:** `frontend/src/drift/frontend/pages/gallery/TileAreaFields.scala` (redraw's and edit's),
`PidUpscalePanel.scala`, `SeedVr2UpscalePanel.scala`, `UpscaleTaskPanel.scala`;
`shared/src/drift/shared/PostProcess.scala` (`PidUpscaleRequest.tilesFor`, `SeedVr2UpscaleRequest.tilesFor`:
the grid falls where it falls, `offsetX = 0`, `offsetY = 0`, the tile size the runtime's maximum)

## Done (2026-10-06)

- **Two gates kept the grid off an upscale, not one.** The picture's size was only recorded for redraw and
  edit (below), and `GenerationMediaViewer.media` only added the grid's element in its `selectable` branch.
  The size is now the picture's whatever task is open, reset when the picture on screen changes; the grid is
  drawn and can be dragged while any tiled task is open (`DetailPicture.tiled`), the box only while one that
  takes a selection is (`selectable`, `PostProcessSection.SelectionTasks`).
- **One piece for where the tiles fall**: `TileAreaFields`, held by redraw, edit, SeedVR2 and PiD — the tile
  field, the grid's switch and reset, the plan, and what it publishes to the picture. The one difference is
  its `Cut`: `Selection` (the model's multiple; window and margin fields) or `Whole` (an upscale's tiling in
  target px and its scale to the picture). The picture reads one `TileGeometry` (`RedrawGeometry`,
  `UpscaleGeometry`, `NoTiles` for ESRGAN): `pidTiles`, `seedVr2Tiles` and the upscale branch of
  `DetailPicture.drawnTiles` are gone.
- **The upscale requests carry the tile size and the grid's shift** (`tileSize`, `gridOffsetX`,
  `gridOffsetY`, in target px), and both sides lay the tiles out from one `UpscaleTiling`
  (`PidUpscaleRequest.tilingFor`, `SeedVr2UpscaleRequest.tilingFor`). The tile is kept between 1024 px and the
  largest the runtime takes; the overlap stays `Tiling.Overlap`.
- Not done: an overlap the user can set (`bugs/42`'s other idea). A shifted grid's end tiles can be as short
  as two overlaps (512 target px), as a redraw's can — how SeedVR2 and PiD take tiles that small is not
  measured.

## Before

- **Redraw and Edit** (`TileAreaFields`): the grid drawn over the picture, the tile size as a field, the grid
  moved by hand so that a seam does not cross a face, the plan (how many tiles) before the job starts.
- **PiD and SeedVR2**: **show the grid** draws the tiles the job will cut and the line before the button says
  how many — and that is all. The tile is the largest the runtime takes, the grid cannot be moved, and nothing
  can be chosen.
- **ESRGAN**: no grid.

## The grid that exists is not drawn (François, 2026-10-05, on an upscale task)

**show the grid** is ticked and nothing is drawn over the picture. He remembers it working when PiD was a task
of its own, before the one **Upscale** task (`UpscaleTaskPanel`) took PiD, SeedVR2 and ESRGAN in.

Read in the code, not verified in a browser:

- **Ruled out by him:** the inner panels publishing with `tiles.changes` (the value already there left out) —
  changing the scale ×2 ↔ ×4 with the box ticked draws nothing.
- **The picture's size is only recorded while a redraw or an edit is open.** `GenerationMediaViewer` sets
  `viewed` — the picture's width and height — in two places, the answer of `sizeOf` and the image's `onLoad`,
  both under `if (selectable)`; and `DetailPicture.selectable` is true only for `PostProcessSection.TiledTasks`
  = redraw and edit (the selection box belongs to those: François, 2026-09-19). On the Upscale task `viewed`
  stays empty, so the panels compute no tiles (`tiles` and `planned` both start from `viewed`) and the viewer's
  `tileGrid` draws nothing without a size. When PiD was a task of its own this was presumably not gated so.
  To check: open **Redraw** on the picture, then come back to **Upscale** — the size is then known and the grid
  should appear. Fix: record the size whatever the task (the two `if (selectable)`), keeping the *selection*
  alone tied to `selectable`; and see that `viewed` is reset when the picture on screen changes, since nothing
  but a video clears it.

## Wanted

The upscale tasks cut the picture as a redraw does, so they get the same piece: the grid on by the same
control, a tile size, and the grid moved by hand, sent with the request and used by the backend's layout
(the browser and the backend already share `tilesFor`).

## Do it as one refactoring, not two patches (François, 2026-10-05)

"We should use this opportunity to factor all of this a bit better: all these tasks are tiled tasks, and
behave in a very similar way. The only thing is that upscale cannot work a partial tile set."

Redraw, Edit and the upscales (PiD, SeedVR2) are one kind of task — a picture cut in tiles, a grid drawn over
it, a plan before the job, a job that runs tile by tile — and the frontend should say so once:

- **One piece for where the tiles fall**, held by every tiled task: the grid shown, the tile size, the grid
  moved by hand, the plan. Today `TileAreaFields` is redraw's and edit's, and each upscale panel computes and
  publishes tiles of its own (`seedVr2Tiles`, `pidTiles`, picked again in `UpscaleTaskPanel` and once more in
  `DetailPicture.drawnTiles`).
- **One difference, as a parameter** rather than a second path: a task either takes a *selection* (redraw,
  edit: a box on the picture, the window around it, the paste back) or works the *whole picture only*
  (upscale). `PostProcessSection.TiledTasks` then means what its name says — every tiled task — and what is
  tied to redraw and edit today is "takes a selection": the box on the picture, `DetailPicture.selectable`.
- **The picture's size belongs to the picture**, not to the selection: recorded whatever task is open, reset
  when the picture changes. That is the grid bug above, fixed by the factoring rather than beside it.
- The scale is the upscale's own: its tiles are laid out in target pixels and drawn in the source's
  (`tile / scale`), which the shared piece has to carry.
- The backend already has this shape (`TiledJobs`, `TiledArea`, `TileRequests`); the request types would gain
  the tile size and the grid offset the upscales lack (`PidUpscaleRequest.tilesFor`, `SeedVr2UpscaleRequest
  .tilesFor`: `offsetX = 0`, `offsetY = 0`, the runtime's largest tile).

## Related

- `bugs/42`: the tiling seen on some tiles of a SeedVR2 ×2 then ×2. His idea there — more tiles, or bigger
  ones — is the same control: a tile size and an overlap the user can set would let him try it by hand, and
  moving the grid takes a seam off a face.
- A selection (upscale a part only) is not asked for here.
