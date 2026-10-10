package drift.runner

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.util.SplittableRandom
import javax.imageio.ImageIO
import javax.imageio.metadata.IIOMetadataNode

import utest.*

import drift.runner.diffusion.PreviewClip
import drift.runner.server.PreviewGif

/** A video's step preview as a GIF (`specs/54-step-preview.md`): its frames,
  * their pace, the loop, and the caps on its frames and its weight.
  */
object PreviewGifTests extends TestSuite {

  /** A GIF read back: each frame's size and delay (hundredths of a second), and
    * whether it carries Netscape's loop-for-ever extension.
    */
  private def read(gif: Array[Byte]): (Seq[(Int, Int, Int)], Boolean) = {
    val reader = ImageIO.getImageReadersByFormatName("gif").next()
    reader.setInput(
      ImageIO.createImageInputStream(new ByteArrayInputStream(gif))
    )
    def nodes(root: IIOMetadataNode, name: String) = {
      val found = root.getElementsByTagName(name)
      (0 until found.getLength).map(found.item(_).asInstanceOf[IIOMetadataNode])
    }
    try {
      val trees = (0 until reader.getNumImages(true)).map { index =>
        val metadata = reader.getImageMetadata(index)
        metadata
          .getAsTree(metadata.getNativeMetadataFormatName)
          .asInstanceOf[IIOMetadataNode]
      }
      val frames = trees.zipWithIndex.map { (tree, index) =>
        (
          reader.getWidth(index),
          reader.getHeight(index),
          nodes(tree, "GraphicControlExtension").head
            .getAttribute("delayTime")
            .toInt
        )
      }
      val loops = nodes(trees.head, "ApplicationExtension").exists(node =>
        node.getAttribute("applicationID") == "NETSCAPE" &&
          node.getAttribute("authenticationCode") == "2.0" &&
          node.getUserObject.asInstanceOf[Array[Byte]].toSeq == Seq[Byte](
            1,
            0,
            0
          )
      )
      (frames, loops)
    } finally reader.dispose()
  }

  /** A clip of `frames` pictures of `width × height`, each of `pixel(frame, x,
    * y)`.
    */
  private def clip(frames: Int, width: Int, height: Int, seconds: Double)(
      pixel: (Int, Int, Int) => Int
  ) =
    PreviewClip(
      frames,
      seconds,
      frame => {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        for {
          y <- 0 until height
          x <- 0 until width
        } image.setRGB(x, y, pixel(frame, x, y))
        image
      }
    )

  val tests = Tests {
    test("the frames kept are spread evenly, the first always") {
      assert(
        PreviewGif.picked(5, 64) == Seq(0, 1, 2, 3, 4),
        PreviewGif.picked(8, 4) == Seq(0, 2, 4, 6),
        PreviewGif.picked(200, 64).size == 64,
        PreviewGif.picked(200, 64).head == 0,
        PreviewGif.picked(200, 64).last < 200
      )
    }
    test("the delays add up to the clip, whatever the frames kept") {
      assert(
        // Wan: 21 latent frames over 81 frames at 16 a second
        PreviewGif.delays(21, 81 / 16.0).sum == 506,
        PreviewGif.delays(21, 81 / 16.0).forall(d => d == 24 || d == 25),
        PreviewGif.delays(3, 1.0) == Seq(33, 34, 33),
        PreviewGif.delays(64, 20.0).sum == 2000,
        // never under what browsers play as written
        PreviewGif.delays(64, 0.5).forall(_ == 2)
      )
    }
    test("a clip is a looping GIF at its own pace, enlarged") {
      // 5 latent frames of 52 × 30 (a 16:9 clip), a soft gradient that moves
      val gif = PreviewGif.encode(clip(5, 52, 30, 1.25) { (frame, x, y) =>
        val level = (x * 4 + frame * 10) & 0xff
        (level << 16) | ((y * 8) << 8) | (255 - level)
      })
      val (frames, loops) = read(gif)
      assert(
        new String(gif.take(6), "US-ASCII") == "GIF89a",
        frames.size == 5,
        // 52 → 208, the whole factor that stays within 256
        frames.forall((width, height, _) => width == 208 && height == 120),
        frames.map(_._3) == Seq(25, 25, 25, 25, 25),
        loops,
        // one palette for the file, not one a frame
        gif.length < 5 * 208 * 120 / 2
      )
    }
    test("a tall clip is enlarged by its height") {
      val (frames, _) =
        read(PreviewGif.encode(clip(2, 9, 16, 1.0)((_, _, _) => 0x808080)))
      assert(frames.forall((width, height, _) => width == 144 && height == 256))
    }
    test("a long clip keeps its duration on fewer frames") {
      val (frames, loops) = read(
        PreviewGif.encode(clip(200, 16, 16, 20.0)((frame, _, _) => frame))
      )
      assert(
        frames.size == PreviewGif.MaximumFrames,
        frames.map(_._3).sum == 2000,
        loops
      )
    }
    test("a preview too heavy is made smaller, then of fewer frames") {
      // the worst there is: 64 frames of noise, which nothing compresses
      val random = new SplittableRandom(3)
      val noise = Array.fill(64 * 128 * 128)(random.nextInt(1 << 24))
      val gif = PreviewGif.encode(clip(64, 128, 128, 4.0) { (frame, x, y) =>
        noise((frame * 128 + y) * 128 + x)
      })
      val (frames, loops) = read(gif)
      assert(
        gif.length <= PreviewGif.MaximumBytes,
        // not enlarged, and frames dropped
        frames.forall((width, height, _) => width == 128 && height == 128),
        frames.size < 64 && frames.size >= 2,
        frames.map(_._3).sum == 400,
        loops
      )
    }
  }
}
