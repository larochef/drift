package drift.runner.server

import drift.runner.diffusion.PreviewClip

import java.awt.RenderingHints
import java.awt.image.{BufferedImage, IndexColorModel}
import java.io.ByteArrayOutputStream
import javax.imageio.*
import javax.imageio.metadata.IIOMetadataNode
import javax.imageio.stream.MemoryCacheImageOutputStream

/** A video's step preview as one picture that moves
  * (`specs/54-step-preview.md`): the clip's latent frames as an animated GIF
  * that loops at the clip's own pace, which an `<img>` plays by itself. The
  * frames are enlarged to a longer side of about `LongerSide`, drawn on one
  * palette cut from their own colours, and at most `MaximumFrames` of them,
  * evenly picked over the clip; a GIF over `MaximumBytes` is made again
  * smaller, then of half the frames. The duration stays the clip's whatever is
  * dropped.
  */
object PreviewGif {

  /** The most frames a preview holds. */
  val MaximumFrames = 64

  /** The longer side the frames are enlarged towards, by a whole factor. */
  val LongerSide = 256

  /** The most bytes a preview weighs, while it has frames or size to give. */
  val MaximumBytes = 200_000

  /** The shortest delay browsers play as written, in hundredths of a second.
    */
  private val ShortestDelay = 2

  /** The colours of a preview's palette: few, since a GIF of a soft picture
    * weighs by its colours and an estimate shows no more.
    */
  private val Colours = 64

  /** `clip` as a looping GIF. */
  def encode(clip: PreviewClip): Array[Byte] = {
    val drawn = picked(clip.frames, MaximumFrames).map(clip.frame)
    val longer = drawn.headOption.fold(1)(f => f.getWidth max f.getHeight)
    def made(frames: Seq[BufferedImage], factor: Int) = {
      val enlarged = frames.map(pixels(_, factor))
      val (colours, indexOf) = palette(enlarged.map(_._1))
      write(
        enlarged.map((rgb, width) => indexed(rgb, width, colours, indexOf)),
        delays(frames.size, clip.seconds)
      )
    }
    var (frames, factor) = (drawn, math.max(1, LongerSide / longer))
    var gif = made(frames, factor)
    while (gif.length > MaximumBytes && (factor > 1 || frames.size > 2)) {
      if (factor > 1) factor /= 2
      else frames = picked(frames.size, frames.size / 2).map(frames)
      gif = made(frames, factor)
    }
    gif
  }

  /** `limit` of `count` frames at most, evenly spread from the first. */
  def picked(count: Int, limit: Int): Seq[Int] =
    if (count <= limit) 0 until count
    else (0 until limit).map(i => (i.toLong * count / limit).toInt)

  /** Each of `count` frames' delay, in hundredths of a second, for a clip of
    * `seconds`: rounded so that their sum is the clip's, none under what
    * browsers play.
    */
  def delays(count: Int, seconds: Double): Seq[Int] = {
    def until(frame: Int) = math.round(frame * seconds * 100 / count)
    (0 until count).map(frame =>
      math.max(ShortestDelay.toLong, until(frame + 1) - until(frame)).toInt
    )
  }

  /** `frame` enlarged `factor` times (bilinear): its pixels, row by row, and
    * its width.
    */
  private def pixels(frame: BufferedImage, factor: Int): (Array[Int], Int) = {
    val (width, height) = (frame.getWidth * factor, frame.getHeight * factor)
    val enlarged =
      if (factor == 1) frame
      else {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        graphics.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION,
          RenderingHints.VALUE_INTERPOLATION_BILINEAR
        )
        graphics.drawImage(frame, 0, 0, width, height, null)
        graphics.dispose()
        image
      }
    (enlarged.getRGB(0, 0, width, height, null, 0, width), width)
  }

  /** A pixel's colour on 5 bits a channel: what the palette is cut from. */
  private def coarse(pixel: Int): Int =
    ((pixel >> 9) & 0x7c00) | ((pixel >> 6) & 0x3e0) | ((pixel >> 3) & 0x1f)

  /** One palette for every frame of a clip, so the file carries it once and no
    * colour flickers from a frame to the next: the median cut of the frames'
    * colours (on 5 bits a channel) into `Colours` boxes, each box its colours'
    * mean; and the palette's index of each coarse colour. No dither: an
    * estimate is soft, and noise in it would weigh more than it shows.
    */
  private def palette(
      frames: Seq[Array[Int]]
  ): (IndexColorModel, Array[Byte]) = {
    val counts = new Array[Int](1 << 15)
    frames.foreach(_.foreach(pixel => counts(coarse(pixel)) += 1))
    def channel(colour: Int, axis: Int) = (colour >> (10 - 5 * axis)) & 31

    /** A box of colours, with the axis it is longest on and what it weighs in
      * the choice of the next to cut: its length there by its pixels.
      */
    final case class Box(colours: Array[Int]) {
      val (length, axis) = (0 until 3).map { axis =>
        val values = colours.map(channel(_, axis))
        (values.max - values.min, axis)
      }.max
      val weight: Long = length.toLong * colours.map(counts(_).toLong).sum
    }
    var boxes = Vector(Box(counts.indices.filter(counts(_) > 0).toArray))
    var cuttable = boxes.head.colours.nonEmpty
    while (boxes.size < Colours && cuttable) {
      val box = boxes.maxBy(_.weight)
      if (box.length == 0) cuttable = false
      else {
        // at the median pixel along the longest axis, a colour on each side
        val sorted = box.colours.sortBy(channel(_, box.axis))
        val half = sorted.map(counts(_).toLong).sum / 2
        var (at, before) = (0, 0L)
        while (at < sorted.length - 2 && before + counts(sorted(at)) <= half) {
          before += counts(sorted(at))
          at += 1
        }
        val (low, high) = sorted.splitAt(at + 1)
        boxes = boxes.filterNot(_ eq box) :+ Box(low) :+ Box(high)
      }
    }
    val indexOf = new Array[Byte](1 << 15)
    val channels = Array.ofDim[Byte](3, Colours)
    boxes.zipWithIndex.foreach { (box, index) =>
      val pixels = math.max(1L, box.colours.map(counts(_).toLong).sum)
      (0 until 3).foreach { axis =>
        // a coarse colour's 5 bits spread over 8
        val mean = box.colours.map { colour =>
          val value = channel(colour, axis)
          ((value << 3) | (value >> 2)) * counts(colour).toLong
        }.sum / pixels
        channels(axis)(index) = mean.toByte
      }
      box.colours.foreach(indexOf(_) = index.toByte)
    }
    (
      new IndexColorModel(8, Colours, channels(0), channels(1), channels(2)),
      indexOf
    )
  }

  /** A frame's pixels on the clip's palette. */
  private def indexed(
      rgb: Array[Int],
      width: Int,
      colours: IndexColorModel,
      indexOf: Array[Byte]
  ): BufferedImage = {
    val height = rgb.length / width
    val image =
      new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, colours)
    image.getRaster.setDataElements(
      0,
      0,
      width,
      height,
      rgb.map(pixel => indexOf(coarse(pixel)))
    )
    image
  }

  /** `frames`, on one palette and each shown for its delay (hundredths of a
    * second), as a GIF that loops for ever.
    */
  def write(frames: Seq[BufferedImage], delays: Seq[Int]): Array[Byte] = {
    val bytes = new ByteArrayOutputStream()
    val output = new MemoryCacheImageOutputStream(bytes)
    val writer = ImageIO.getImageWritersByFormatName("gif").next()
    try {
      writer.setOutput(output)
      writer.prepareWriteSequence(null)
      frames.zip(delays).zipWithIndex.foreach { case ((frame, delay), index) =>
        val metadata = writer.getDefaultImageMetadata(
          ImageTypeSpecifier.createFromRenderedImage(frame),
          null
        )
        val format = metadata.getNativeMetadataFormatName
        val root = metadata.getAsTree(format).asInstanceOf[IIOMetadataNode]
        def child(name: String): IIOMetadataNode =
          (0 until root.getLength)
            .map(root.item)
            .collectFirst {
              case node: IIOMetadataNode if node.getNodeName == name => node
            }
            .getOrElse {
              val node = new IIOMetadataNode(name)
              root.appendChild(node)
              node
            }
        val control = child("GraphicControlExtension")
        control.setAttribute("disposalMethod", "none")
        control.setAttribute("userInputFlag", "FALSE")
        control.setAttribute("transparentColorFlag", "FALSE")
        control.setAttribute("delayTime", delay.toString)
        control.setAttribute("transparentColorIndex", "0")
        if (index == 0) {
          // Netscape's looping extension: 1, then the count on 16 bits, 0 for
          // ever
          val loop = new IIOMetadataNode("ApplicationExtension")
          loop.setAttribute("applicationID", "NETSCAPE")
          loop.setAttribute("authenticationCode", "2.0")
          loop.setUserObject(Array[Byte](1, 0, 0))
          child("ApplicationExtensions").appendChild(loop)
        }
        metadata.setFromTree(format, root)
        writer.writeToSequence(new IIOImage(frame, null, metadata), null)
      }
      writer.endWriteSequence()
      output.flush()
    } finally {
      output.close()
      writer.dispose()
    }
    bytes.toByteArray
  }
}
