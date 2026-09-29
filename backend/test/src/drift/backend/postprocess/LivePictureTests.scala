package drift.backend.postprocess

import drift.shared.{ImageRegion, PostProcessPicture, Tiling}
import utest.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.concurrent.duration.*

import ox.*
import ox.channels.Channel

/** The picture a tiled job shows while it runs (`specs/15-post-hoc-resize.md`).
  *
  * Worth testing because it is painted apart from the job, tile by tile, and
  * has to end where the job's own blend ends: a picture that drifted from the
  * result would show a seam, or a colour, the job never made.
  */
object LivePictureTests extends TestSuite {

  private val Red = 0xffcc2020
  private val Blue = 0xff2040cc
  private val Green = 0xff20aa40

  private def filled(width: Int, height: Int, rgb: Int): BufferedImage = {
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    for {
      y <- 0 until height
      x <- 0 until width
    } image.setRGB(x, y, rgb)
    image
  }

  private def written(image: BufferedImage, directory: Path): Path = {
    val file = Files.createTempFile(directory, "tile", ".png")
    ImageIO.write(image, "png", file.toFile)
    file
  }

  private def decoded(bytes: Array[Byte]): BufferedImage =
    ImageIO.read(java.io.ByteArrayInputStream(bytes))

  private def pixels(image: BufferedImage): Seq[Int] =
    image
      .getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth)
      .toSeq

  // Two tiles side by side over a 96×32 picture, overlapping by 32 px.
  private val first = Tiling.Tile(0, 0, 64, 32)
  private val second = Tiling.Tile(32, 0, 64, 32)
  private val rows = List(List(first, second))
  private val overlaps = Map(first -> (0, 0), second -> (32, 0))

  private def picture(painted: Channel[Int])(using Ox) =
    LivePicture(
      "test",
      filled(96, 32, Red),
      scale = 1,
      overlaps,
      target = (96, 32),
      finish = PictureFinish.AsPainted,
      fullSizeFile =
        Files.createTempDirectory("live-picture").resolve("picture.png"),
      onPainted = count => painted.send(count)
    )

  private def fullSize(live: LivePicture): BufferedImage =
    ImageIO.read(live.fullSize().get.toFile)

  /** Whether `shown` is `expected` but for resampling at the seams of the
    * parts it was brought up to date by.
    */
  private def nearly(shown: BufferedImage, expected: BufferedImage): Boolean =
    shown.getWidth == expected.getWidth &&
      shown.getHeight == expected.getHeight && {
        val differences = pixels(shown).zip(pixels(expected)).map { (a, b) =>
          Seq(16, 8, 0)
            .map(shift => math.abs(((a >> shift) & 0xff) - ((b >> shift) & 0xff)))
            .max
        }
        differences.max <= 16 && differences.sum.toDouble / differences.size < 1
      }

  // Two tiles side by side, wider together than the screen's copy, so it is
  // brought up to date scaled.
  private val wide = Tiling.Tile(0, 0, 2304, 2048)
  private val wider = Tiling.Tile(2048, 0, 2304, 2048)
  private val wideRows = List(List(wide, wider))
  private val wideTiles =
    Map(wide -> filled(2304, 2048, Blue), wider -> filled(2304, 2048, Green))

  /** The screen's copy once both wide tiles are in, painted over `reference`
    * and finished by `finish`.
    */
  private def shownAfterWideTiles(
      reference: BufferedImage,
      finish: PictureFinish
  ): BufferedImage = supervised {
    val directory = Files.createTempDirectory("live-picture")
    val painted = Channel.unlimited[Int]
    val live = LivePicture(
      "test",
      reference,
      scale = 1,
      Map(wide -> (0, 0), wider -> (256, 0)),
      target = (4352, 2048),
      finish,
      directory.resolve("picture.png"),
      count => painted.send(count)
    )
    wideTiles.foreach((tile, image) =>
      live.paint(tile, written(image, directory))
    )
    awaitPainted(painted, 1)
    awaitPainted(painted, 2)
    decoded(live.forScreen(PostProcessPicture.ScreenSide).get)
  }

  private def awaitPainted(painted: Channel[Int], count: Int) =
    assert(timeout(10.seconds)(painted.receive()) == count)

  val tests = Tests {

    test("what is not painted yet is the source") {
      val directory = Files.createTempDirectory("live-picture")
      val painted = Channel.unlimited[Int]
      supervised {
      val live = picture(painted)
        live.paint(first, written(filled(64, 32, Blue), directory))
        awaitPainted(painted, 1)
        val shown = fullSize(live)
        assert(shown.getRGB(10, 10) == Blue)
        assert(shown.getRGB(90, 10) == Red)
      }
    }

    test("once every tile is in, it is the job's own blend") {
      val directory = Files.createTempDirectory("live-picture")
      val painted = Channel.unlimited[Int]
      supervised {
      val live = picture(painted)
        val tiles = Map(
          first -> filled(64, 32, Blue),
          second -> filled(64, 32, Green)
        )
        live.paint(first, written(tiles(first), directory))
        live.paint(second, written(tiles(second), directory))
        awaitPainted(painted, 1)
        awaitPainted(painted, 2)
        val blended = TileBlending.blend(rows, 96, 32, tiles)
        assert(pixels(fullSize(live)) == pixels(blended))
      }
    }

    test("scaled for the screen, it keeps the picture's shape") {
      val painted = Channel.unlimited[Int]
      supervised {
      val live = picture(painted)
        // The base is built on the painter; a paint queued behind it is how a
        // caller knows it is there.
        val directory = Files.createTempDirectory("live-picture")
        live.paint(first, written(filled(64, 32, Blue), directory))
        awaitPainted(painted, 1)
        val shown = decoded(live.forScreen(48).get)
        assert(shown.getWidth == 48, shown.getHeight == 16)
      }
    }

    test("at full size, it is written once for as long as no tile lands") {
      val directory = Files.createTempDirectory("live-picture")
      val painted = Channel.unlimited[Int]
      supervised {
      val live = picture(painted)
        live.paint(first, written(filled(64, 32, Blue), directory))
        awaitPainted(painted, 1)
        val file = live.fullSize().get
        val writtenAt = Files.getLastModifiedTime(file)
        Thread.sleep(20)
        assert(live.fullSize().contains(file))
        assert(Files.getLastModifiedTime(file) == writtenAt)
        live.paint(second, written(filled(64, 32, Green), directory))
        awaitPainted(painted, 2)
        assert(fullSize(live).getRGB(90, 10) == Green)
      }
    }

    test("the screen's copy, kept tile by tile, is the whole picture scaled") {
      val source = filled(4352, 2048, Red)
      val shown = shownAfterWideTiles(source, PictureFinish.AsPainted)
      val expected = PostProcessImages.fitWithin(
        TileBlending.blend(wideRows, 4352, 2048, wideTiles),
        PostProcessPicture.ScreenSide
      )
      assert(nearly(shown, expected))
    }

    test("the screen's copy of a partial redraw is pasted into the source") {
      val source = filled(6000, 3000, Red)
      val window = ImageRegion(1000, 500, 4352, 2048)
      val region = ImageRegion(1400, 700, 3500, 1600)
      val finish = PictureFinish.PastedInto(source, window, region)
      val shown = shownAfterWideTiles(
        source.getSubimage(window.x, window.y, window.width, window.height),
        finish
      )
      val expected = PostProcessImages.fitWithin(
        finish(TileBlending.blend(wideRows, 4352, 2048, wideTiles)),
        PostProcessPicture.ScreenSide
      )
      assert(nearly(shown, expected))
    }

    test("a closed picture answers nothing") {
      val painted = Channel.unlimited[Int]
      supervised {
        val live = picture(painted)
        live.close()
        assert(live.fullSize().isEmpty, live.forScreen(48).isEmpty)
      }
    }
  }
}
