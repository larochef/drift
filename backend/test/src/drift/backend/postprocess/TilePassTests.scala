package drift.backend.postprocess

import drift.backend.images.PostProcessImages
import drift.shared.*

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

import utest.*

/** A tiled pass held in memory (`TilePass`): every tile made from its own crop
  * and blended, a tile's picture judged as a tiled job judges it, and the pass
  * ended at the first tile that fails or once it is cancelled.
  */
object TilePassTests extends TestSuite {

  private def filled(width: Int, height: Int, colour: Int): BufferedImage = {
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    (0 until height).foreach(y =>
      (0 until width).foreach(x => image.setRGB(x, y, colour))
    )
    image
  }

  private def png(image: BufferedImage): Array[Byte] = {
    val bytes = ByteArrayOutputStream()
    ImageIO.write(image, "png", bytes)
    bytes.toByteArray
  }

  private val scale = 2
  private val source = filled(1024, 512, 0x336699)
  private val rows =
    SeedVr2UpscaleRequest.tilesFor(
      (source.getWidth, source.getHeight),
      scale,
      tileSize = Some(1024)
    )
  private val tiles = rows.flatten

  /** The crop doubled, as an upscaler gives it back. */
  private def doubled(crop: BufferedImage): Either[String, Array[Byte]] =
    Right(
      png(
        PostProcessImages
          .scaledCopy(crop, crop.getWidth * scale, crop.getHeight * scale)
      )
    )

  val tests = Tests {
    test("every tile is made from its crop and the whole is blended") {
      assert(tiles.size > 1)
      var made = List.empty[Int]
      val whole = TilePass.run(
        rows,
        source,
        scale,
        () => false,
        doubled,
        (returned, _) => returned,
        count => made = made :+ count
      )
      assert(made == (1 to tiles.size).toList)
      val picture = whole.toOption.get
      assert(
        picture.getWidth == source.getWidth * scale,
        picture.getHeight == source.getHeight * scale
      )
      // one flat colour in, the same colour out, seams included
      val seen = for {
        x <- 0 until picture.getWidth by 97
        y <- 0 until picture.getHeight by 89
      } yield picture.getRGB(x, y) & 0xffffff
      assert(seen.distinct == Seq(0x336699))
    }
    test("a tile of another size ends the pass, named") {
      var asked = 0
      val whole = TilePass.run(
        rows,
        source,
        scale,
        () => false,
        crop => {
          asked += 1
          if (asked == 2) Right(png(filled(8, 8, 0xffffff))) else doubled(crop)
        },
        (returned, _) => returned,
        _ => ()
      )
      assert(asked == 2)
      val reason = whole.left.toOption.get
      assert(
        reason.startsWith(s"tile 2 of ${tiles.size}: the image is 8×8, not ")
      )
    }
    test("a black tile is a NaN result, not a picture") {
      val whole = TilePass.run(
        rows,
        source,
        scale,
        () => false,
        crop =>
          Right(png(filled(crop.getWidth * scale, crop.getHeight * scale, 0))),
        (returned, _) => returned,
        _ => ()
      )
      assert(whole.left.toOption.get.contains("entirely black"))
    }
    test("bytes that are no picture are refused") {
      val whole = TilePass.run(
        rows,
        source,
        scale,
        () => false,
        _ => Right(Array[Byte](1, 2, 3)),
        (returned, _) => returned,
        _ => ()
      )
      assert(whole.left.toOption.get.contains("no readable image"))
    }
    test("a cancel is read before each tile") {
      var asked = 0
      val whole = TilePass.run(
        rows,
        source,
        scale,
        () => asked == 1,
        crop => { asked += 1; doubled(crop) },
        (returned, _) => returned,
        _ => ()
      )
      assert(asked == 1, whole == Left(s"tile 2 of ${tiles.size}: cancelled"))
    }
  }
}
