package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage

/** One tile's dealings with the model: what it is handed — itself, or itself
  * inside a window of its surroundings under a mask (`specs/27-redraw.md`) —
  * and what becomes of what it returns, where a job finishes its tiles itself
  * (`specs/39-seamless-edit.md`).
  */
private[postprocess] object TileWindow {

  /** What an edit (`specs/39-seamless-edit.md`) makes of a tile the model
    * returned, before it joins the picture: given the crop the model was handed
    * and what came back, the image painted in, a line for the job log, and
    * images kept beside the tile's input and output when tiles are kept, by the
    * part of their file name.
    */
  type FinishTile = (BufferedImage, BufferedImage) => FinishedTile

  case class FinishedTile(
      image: BufferedImage,
      note: Option[String],
      kept: List[(String, BufferedImage)],
      /** What the finished picture's gallery entry should say about this tile,
        * when it should say anything.
        */
      warning: Option[String]
  )

  /** A tile painted in as it came back: what a redraw with context does. */
  val keepReturned: FinishTile = (_, returned) =>
    FinishedTile(returned, None, List.empty, None)

  /** What a tile's img_gen request is built from: the tile; the window the
    * model paints — the tile itself, or the tile with its context around it;
    * the window's pixels as a data URL; and, with context, the mask that has
    * the model repaint the tile (white) and keep the rest (black).
    */
  case class TileInput(
      tile: Tiling.Tile,
      window: Tiling.Tile,
      image: String,
      mask: Option[String]
  )

  /** The picture shown around each tile (`specs/27-redraw.md`): up to `margin`
    * px on every side, cut short at the picture's edges and trimmed so the
    * window stays on the model's `sizeMultiple`.
    */
  case class TileContext(margin: Int, sizeMultiple: Int)

  /** `tile` with its context around it, inside a `width`×`height` picture. The
    * tile's own sides are on the multiple, so only the margins are trimmed —
    * the one after the tile first.
    */
  def windowOf(
      tile: Tiling.Tile,
      width: Int,
      height: Int,
      context: TileContext
  ): Tiling.Tile = {
    def axis(start: Int, length: Int, total: Int): (Int, Int) = {
      val before = context.margin min start
      val after = context.margin min (total - start - length)
      val excess = (length + before + after) % context.sizeMultiple
      val fromAfter = excess min after
      val kept = before - (excess - fromAfter)
      (start - kept, length + kept + after - fromAfter)
    }
    val (x, windowWidth) = axis(tile.x, tile.width, width)
    val (y, windowHeight) = axis(tile.y, tile.height, height)
    Tiling.Tile(x, y, windowWidth, windowHeight)
  }

  /** The mask of `tile` inside `window`: white where the model repaints, black
    * where it keeps what is there.
    */
  def maskOf(tile: Tiling.Tile, window: Tiling.Tile): BufferedImage = {
    val mask =
      BufferedImage(window.width, window.height, BufferedImage.TYPE_BYTE_GRAY)
    val graphics = mask.createGraphics()
    try {
      graphics.setColor(java.awt.Color.WHITE)
      graphics.fillRect(
        tile.x - window.x,
        tile.y - window.y,
        tile.width,
        tile.height
      )
    } finally graphics.dispose()
    mask
  }
}
