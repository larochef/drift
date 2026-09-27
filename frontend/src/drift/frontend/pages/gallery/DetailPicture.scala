package drift.frontend.pages.gallery

import drift.shared.*

import com.raquo.laminar.api.L.*

/** The picture the detail view shows and everything drawn over it
  * (`specs/12-gallery.md`, `specs/15-post-hoc-resize.md`): which output is on
  * screen, the box dragged on it, the tile grid and its offset, and the tiles a
  * job is painting there right now.
  *
  * These belong together and are read together — the viewer draws them, the
  * post-processing panels write them — and they are one signal chain: what is
  * shown decides which job's tiles are drawn, which decides what each tile
  * looks like. Kept in the detail view's body among its view fragments, a new
  * link in that chain lands above the one it reads and is `null` when the class
  * is built (`specs/29-split-oversized-files.md`). Here the order is the file's
  * own, and the compiler holds it.
  */
class DetailPicture(
    generation: Generation,
    /** The original this entry was made from, once the gallery holds it. */
    parent: Signal[Option[Generation]],
    /** Every post-processing job, to find the one on this picture. */
    jobs: Signal[List[PostProcessJob]],
    /** Which section of the right column is open, and whether it is folded
      * away: a box may only be drawn while a task that acts on one is on
      * screen.
      */
    openSection: Var[String],
    panelHidden: Var[Boolean],
    openTask: Var[String],
    initialOutputIndex: Int
) {

  val selectedIndex = Var(initialOutputIndex)

  /** Comparing a derived entry with what it was made from. The header offers
    * it, the viewer shows what it names.
    */
  val showOriginal = Var(false)

  /** The parent's output this entry was made from. */
  val originalOutput: Signal[Option[GenerationOutput]] =
    parent
      .map(loaded =>
        generation.derivation.map { d =>
          loaded
            .flatMap(_.outputs.find(_.fileName == d.parentFileName))
            .getOrElse(
              GenerationOutput(
                date = d.parentDate,
                fileName = d.parentFileName,
                url = s"/api/outputs/${d.parentDate}/${d.parentFileName}",
                mimeType = "image/png",
                format = "png"
              )
            )
        }
      )
      .distinct

  /** Whether a box may be drawn on the picture: only on the result, and only
    * while the redraw panel is the one on screen — the upscalers have nothing
    * to do with a selection. Drawing one that nothing on screen can act on was
    * the tool appearing where it does not belong (François, 2026-09-19).
    */
  val selectable: Signal[Boolean] =
    showOriginal.signal
      .combineWith(openSection.signal, panelHidden.signal, openTask.signal)
      .map((original, section, hidden, task) =>
        !original && !hidden && section == "post" &&
          PostProcessSection.TiledTasks.contains(task)
      )
      .distinct

  /** The output on screen: the one the strip points at, or the original. */
  val shownOutput: Signal[Option[GenerationOutput]] =
    selectedIndex.signal
      .combineWith(showOriginal.signal, originalOutput)
      .map((index, original, parentOutput) =>
        if (original) parentOutput else generation.outputs.lift(index)
      )
      .distinct

  /** The image the viewer shows and the part of it a redraw should repaint
    * (`specs/27-redraw.md`): the viewer writes it, the redraw panel reads it.
    */
  val viewed = Var(Option.empty[ViewedImage])

  /** What the redraw panel would do with a selection — its tile size, margin
    * and window. The panel writes it as its fields are typed; the viewer reads
    * it to count tiles under the box and to stick to their boundaries.
    */
  val redrawGeometry = Var(RedrawGeometry())

  /** Whether the tiles a redraw would run are drawn over the picture. The
    * redraw panel's checkbox writes it, the viewer reads it; it belongs to the
    * image on screen rather than to the host, like the rest of the panel's
    * fields.
    */
  val showTileGrid = Var(true)

  /** The tiles the PiD panel would decode, in the picture's own pixels. */
  val pidTiles = Var(List.empty[ImageRegion])

  /** Where that grid is cut. Dragging a line of it on the picture moves it, so
    * a face can be put inside one tile instead of across the seam between two;
    * the redraw panel sends it with the job.
    */
  val gridOffset = Var(TileOffset(0, 0))

  /** The job whose tiles the picture shows, if one is on this very output: only
    * one job runs on an image at a time (`specs/15-post-hoc-resize.md`), so
    * there is never a choice to make.
    */
  val jobOnPicture: Signal[Option[PostProcessJob]] =
    jobs
      .combineWith(shownOutput)
      .map((all, output) =>
        output.flatMap(shown =>
          all.find(job =>
            job.sourceDate == shown.date &&
              job.sourceFileName == shown.fileName &&
              job.tiles.nonEmpty &&
              (job.state.isActive || job.state == PostProcessState.Paused)
          )
        )
      )
      .distinct

  /** The tiles drawn over the picture, each with what has become of it
    * (`specs/15-post-hoc-resize.md`): a job's own while one is on this image —
    * done, running, still to come — else the open task's, which follow the
    * panel's fields and the box drawn on the picture.
    */
  val drawnTiles: Signal[List[TilePaint]] =
    jobOnPicture
      .combineWith(
        openTask.signal,
        viewed.signal,
        redrawGeometry.signal,
        pidTiles.signal
      )
      .map { (job, task, shown, geometry, pid) =>
        job match {
          case Some(running) =>
            val done = running.progress.fold(0)(_.completed)
            val active = running.state.isActive
            running.tiles.zipWithIndex.map { (tile, index) =>
              if (index < done)
                TilePaint(
                  tile,
                  TileState.Done,
                  Some(
                    s"/api/post-process-jobs/${running.id}/tiles/$index?side=512"
                  )
                )
              else if (index == done && active)
                TilePaint(tile, TileState.Running)
              else TilePaint(tile, TileState.Waiting)
            }
          case None =>
            val planned =
              if (task == PostProcessSection.PidTask) pid
              else
                shown.toList.flatMap(picture =>
                  geometry
                    .layoutFor(picture.selection, picture.width, picture.height)
                )
            planned.map(TilePaint(_, TileState.Waiting))
        }
      }
      .distinct
}
