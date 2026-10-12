package drift.backend.postprocess

import drift.backend.images.PostProcessImages
import drift.shared.*

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

/** What every tiled pass does to a tile, whoever runs it: the judgement of the
  * picture a tile came back as, and a whole pass held in memory — each tile cut
  * from its source, made, judged, corrected and blended over the overlaps — for
  * the passes that are one step of a larger job. A tiled job of its own, which
  * keeps its tiles on disk to be paused and shown as it goes, is a `TileRun`.
  */
private[postprocess] object TilePass {

  /** Why the picture a tile came back as cannot be kept, if it cannot: none
    * readable, not the tile's size, or entirely black — a NaN result decodes as
    * a well-formed black picture that its server calls a success.
    */
  def refusal(
      size: Option[(Int, Int)],
      tile: Tiling.Tile,
      black: => Boolean
  ): Option[String] = size match {
    case None => Some("no readable image was written")
    case Some((width, height))
        if width != tile.width || height != tile.height =>
      Some(
        s"the image is ${width}×$height, not ${tile.width}×${tile.height}"
      )
    case Some(_) =>
      Option.when(black)(
        "the image is entirely black — the model's result was NaN"
      )
  }

  /** `source` ×`scale` in the tiles of `rows`, one after another: `make` turns
    * a tile's crop of the source into the bytes of its picture, `correct` gives
    * the picture what it must keep of the crop, and `made` is told how many
    * tiles are done. Ends at the first tile that fails, naming it, and before
    * the next one once `cancelled`.
    */
  def run(
      rows: List[List[Tiling.Tile]],
      source: BufferedImage,
      scale: Int,
      cancelled: () => Boolean,
      make: BufferedImage => Either[String, Array[Byte]],
      correct: (BufferedImage, BufferedImage) => BufferedImage,
      made: Int => Unit
  ): Either[String, BufferedImage] = {
    val tiles = rows.flatten
    tiles.zipWithIndex
      .foldLeft[Either[String, Map[Tiling.Tile, BufferedImage]]](
        Right(Map.empty)
      ) { case (done, (tile, index)) =>
        done.flatMap { images =>
          val crop = PostProcessImages.copyOf(
            source.getSubimage(
              tile.x / scale,
              tile.y / scale,
              tile.width / scale,
              tile.height / scale
            )
          )
          (if (cancelled()) Left("cancelled") else make(crop))
            .flatMap { bytes =>
              val returned = Option(ImageIO.read(ByteArrayInputStream(bytes)))
              refusal(
                returned.map(image => (image.getWidth, image.getHeight)),
                tile,
                returned.exists(PostProcessImages.isBlack)
              ).toLeft(returned.get)
            }
            .left
            .map(reason => s"tile ${index + 1} of ${tiles.size}: $reason")
            .map { returned =>
              made(index + 1)
              images + (tile -> correct(returned, crop))
            }
        }
      }
      .map(images =>
        TileBlending.blend(
          rows,
          source.getWidth * scale,
          source.getHeight * scale,
          images
        )
      )
  }
}
