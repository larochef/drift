package drift.backend.postprocess

import drift.shared.Tiling
import utest.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import javax.imageio.ImageIO

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

  private def picture(painted: LinkedBlockingQueue[Int]) =
    LivePicture(
      "test",
      filled(96, 32, Red),
      scale = 1,
      overlaps,
      target = (96, 32),
      finish = identity,
      onPainted = count => painted.put(count)
    )

  private def awaitPainted(painted: LinkedBlockingQueue[Int], count: Int) =
    assert(Option(painted.poll(10, TimeUnit.SECONDS)).contains(count))

  val tests = Tests {

    test("what is not painted yet is the source") {
      val directory = Files.createTempDirectory("live-picture")
      val painted = LinkedBlockingQueue[Int]()
      val live = picture(painted)
      try {
        live.paint(first, written(filled(64, 32, Blue), directory))
        awaitPainted(painted, 1)
        val shown = decoded(live.snapshot(None).get)
        assert(shown.getRGB(10, 10) == Blue)
        assert(shown.getRGB(90, 10) == Red)
      } finally live.close()
    }

    test("once every tile is in, it is the job's own blend") {
      val directory = Files.createTempDirectory("live-picture")
      val painted = LinkedBlockingQueue[Int]()
      val live = picture(painted)
      try {
        val tiles = Map(
          first -> filled(64, 32, Blue),
          second -> filled(64, 32, Green)
        )
        live.paint(first, written(tiles(first), directory))
        live.paint(second, written(tiles(second), directory))
        awaitPainted(painted, 1)
        awaitPainted(painted, 2)
        val blended = TileBlending.blend(rows, 96, 32, tiles)
        assert(pixels(decoded(live.snapshot(None).get)) == pixels(blended))
      } finally live.close()
    }

    test("scaled for the screen, it keeps the picture's shape") {
      val painted = LinkedBlockingQueue[Int]()
      val live = picture(painted)
      try {
        // The base is built on the painter; a paint queued behind it is how a
        // caller knows it is there.
        val directory = Files.createTempDirectory("live-picture")
        live.paint(first, written(filled(64, 32, Blue), directory))
        awaitPainted(painted, 1)
        val shown = decoded(live.snapshot(Some(48)).get)
        assert(shown.getWidth == 48, shown.getHeight == 16)
      } finally live.close()
    }

    test("a closed picture answers nothing") {
      val painted = LinkedBlockingQueue[Int]()
      val live = picture(painted)
      live.close()
      assert(live.snapshot(None).isEmpty)
    }
  }
}
