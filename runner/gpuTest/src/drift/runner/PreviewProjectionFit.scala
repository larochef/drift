package drift.runner

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO

import drift.runner.diffusion.{Images, LatentProjection, MiniMaxH3Conditions}
import drift.runner.models.{
  FluxVae,
  LtxVideoVae,
  MageVae,
  MiniMaxH3VideoEncoder,
  Posterior,
  QwenImage21Vae,
  WanVae
}
import drift.runner.ops.{HipOps, MatVecInputs, Ops}
import drift.runner.tensor.{DType, Shape}

/** Fits a VAE family's step-preview projection (`specs/54-step-preview.md`):
  * encodes pictures with the family's VAE as its pipeline does, regresses each
  * token's `block × block` pixels of the picture on the token, and writes the
  * projection where the runner reads it (`runner/resources/previews`). Each
  * picture is encoded a second time with its colours swapped around
  * (`recoloured`), so the fit sees colours the pictures lack. A video VAE
  * encodes each picture as a clip that holds it still, and every latent frame
  * of the clip is a sample of the picture: the first frame's kind and the later
  * ones'. Every fifth picture is held out of a first fit, which is what the
  * printed error is measured on; the projection written is fitted on all of
  * them. Each picture beside its projection goes to `pairs` as a PNG. The time
  * a step's preview takes is printed too: the two tensors read back, the
  * picture drawn and its JPEG written, on the first picture's grid.
  *
  * `./mill runner.gpuTest.runMain drift.runner.PreviewProjectionFit <family>
  * <vae> <block> <projection.txt> <pairs> <picture>…`, `family` one of
  * `LatentProjection`'s names.
  */
object PreviewProjectionFit {

  /** How a family reads a picture: `encode` gives the tokens its transformer
    * takes, a grid a latent frame, and the pixels a token covers along a side;
    * the pictures' sides are multiples of `multiple`, the longest `longestSide`
    * at most; `close` closes the VAE.
    */
  final private case class Family(
      encode: BufferedImage => (Seq[Array[Float]], Int),
      close: () => Unit,
      multiple: Int = 16,
      longestSide: Int = 1024
  )

  def main(arguments: Array[String]): Unit = {
    val Array(family, file, blockText, output, pairs) = arguments.take(5)
    val pictures = arguments.drop(5).toSeq.map(Paths.get(_))
    val block = blockText.toInt
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val read = encoder(ops, family, Paths.get(file))
    val close = read.close
    try {
      // each picture's samples, its own then its recoloured one's, a latent
      // frame each
      val perPicture = pictures.zipWithIndex.map { (path, index) =>
        val file = ImageIO.read(path.toFile)
        val reduction =
          math.min(
            1.0,
            read.longestSide.toDouble / (file.getWidth max file.getHeight)
          )
        def side(length: Int) =
          (length * reduction).toInt / read.multiple * read.multiple
        val own =
          Images.resized(file, side(file.getWidth), side(file.getHeight))
        val name = path.getFileName.toString.stripSuffix(".png")
        Seq(own -> name, recoloured(own, index) -> s"$name-recoloured")
          .flatMap { (picture, pictureName) =>
            val (grids, scale) = read.encode(picture)
            val (gridHeight, gridWidth) =
              (picture.getHeight / scale, picture.getWidth / scale)
            println(
              s"${path.getFileName}: ${grids.size} × $gridHeight × $gridWidth tokens"
            )
            val pixels = averaged(picture, scale / block)
            grids.zipWithIndex.map { (tokens, frame) =>
              val frameName =
                if (grids.size == 1) pictureName else s"$pictureName-$frame"
              (frameName, (tokens, pixels, gridHeight, gridWidth))
            }
          }
      }
      val names = perPicture.flatten.map(_._1)
      val samples = perPicture.flatten.map(_._2)
      val channels = samples.head._1.length /
        (samples.head._3 * samples.head._4)
      val (heldOut, kept) = {
        val (out, in) =
          perPicture.zipWithIndex.partition((_, index) => index % 5 == 4)
        (out.flatMap(_._1), in.flatMap(_._1))
      }
      val partial = LatentProjection.fit(channels, block, kept.map(_._2))
      println(
        f"$family, block $block: held-out error ${error(partial, heldOut.map(_._2))}%.4f, " +
          f"fitted ${error(partial, kept.map(_._2))}%.4f (RMS, colours in [−1, 1])"
      )
      val projection = LatentProjection.fit(channels, block, samples)
      Files.writeString(Paths.get(output), projection.text)
      val (first, _, firstHeight, firstWidth) = samples.head
      val held =
        ops.fromFloats(Shape.of(first.length / channels, channels), first)
      try {
        def preview(): Int = {
          val drawn = projection.estimate(
            ops.toFloats(held),
            ops.toFloats(held),
            0f,
            firstHeight,
            firstWidth
          )
          val jpeg = new ByteArrayOutputStream()
          ImageIO.write(drawn, "jpg", jpeg)
          jpeg.size
        }
        (0 until 20).foreach(_ => preview())
        val started = System.nanoTime()
        val bytes = (0 until 50).map(_ => preview()).last
        println(
          f"a preview of $firstHeight × $firstWidth tokens: ${(System.nanoTime() - started) / 50e6}%.2f ms, " +
            s"${firstWidth * block} × ${firstHeight * block} pixels, $bytes bytes of JPEG"
        )
      } finally ops.release(held)
      Files.createDirectories(Paths.get(pairs))
      samples.zip(names).foreach { case ((tokens, pixels, h, w), name) =>
        val projected = projection.picture(tokens, h, w)
        val wanted = Images.toImage(pixels, w * block, h * block)
        val both = new BufferedImage(
          2 * projected.getWidth,
          projected.getHeight,
          BufferedImage.TYPE_INT_RGB
        )
        val graphics = both.createGraphics()
        graphics.drawImage(wanted, 0, 0, null)
        graphics.drawImage(projected, projected.getWidth, 0, null)
        graphics.dispose()
        ImageIO.write(
          both,
          "png",
          Paths.get(pairs).resolve(s"$family-$name.png").toFile
        )
      }
    } finally {
      close()
      ops.close()
    }
  }

  /** How `family` reads a picture, with its VAE `file`. */
  private def encoder(ops: Ops, family: String, file: Path): Family = {
    def pixels(picture: BufferedImage, alpha: Boolean) = ops.fromFloats(
      Shape.of(picture.getHeight, picture.getWidth, if (alpha) 4 else 3),
      Images.pixels(picture, alpha)
    )
    family match {
      case LatentProjection.Wan21 =>
        val vae = WanVae.open(ops, file)
        Family(
          picture => {
            val image = pixels(picture, alpha = false)
            val encoded =
              try vae.encode(image)
              finally ops.release(image)
            val Seq(height, width, channels) = encoded.shape.dimensions
            val packed = ops.allocate(
              DType.F32,
              Shape.of(height / 2 * (width / 2), channels * 4)
            )
            try {
              ops.packPatches(encoded, 2, packed)
              (Seq(ops.toFloats(packed)), 2 * vae.scale)
            } finally {
              ops.release(encoded)
              ops.release(packed)
            }
          },
          () => vae.close()
        )
      case LatentProjection.Flux2 =>
        val vae = FluxVae.open(ops, file)
        Family(
          picture => {
            val image = pixels(picture, alpha = false)
            val encoded =
              try vae.encode(image)
              finally ops.release(image)
            try (Seq(ops.toFloats(encoded)), vae.patch)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case LatentProjection.QwenImage21 =>
        val vae = QwenImage21Vae.open(ops, file)
        Family(
          picture => {
            val image = pixels(picture, alpha = vae.imageChannels == 4)
            val encoded =
              try vae.encode(image)
              finally ops.release(image)
            try (Seq(ops.toFloats(encoded)), vae.scale)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case LatentProjection.MageFlow =>
        val vae = MageVae.open(ops, file)
        Family(
          picture => {
            val image = pixels(picture, alpha = false)
            val encoded =
              try vae.encode(image, None)
              finally ops.release(image)
            try (Seq(ops.toFloats(encoded)), vae.scale)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case LatentProjection.Ltx25 =>
        val vae = LtxVideoVae.open(ops, file)
        Family(
          picture => {
            // 9 frames: a latent frame of the first, one of the 8 after
            val frames =
              Seq.fill(VideoFrames.Ltx)(pixels(picture, alpha = false))
            val encoded =
              try vae.encode(frames)
              finally frames.foreach(ops.release)
            try (encoded.map(ops.toFloats), VideoTokenPixels)
            finally encoded.foreach(ops.release)
          },
          () => vae.close(),
          multiple = VideoTokenPixels,
          longestSide = VideoLongestSide
        )
      case LatentProjection.MiniMaxH3 =>
        val vae = MiniMaxH3VideoEncoder.open(ops, file)
        Family(
          picture => {
            val (height, width) =
              (picture.getHeight / 16, picture.getWidth / 16)
            val still = Images.pixels(picture)
            // 5 frames: a latent frame of the first, one of the 4 after
            val latents = vae.encode(
              Seq.fill(VideoFrames.MiniMaxH3)(still),
              picture.getHeight,
              picture.getWidth,
              Posterior.Mean
            )
            val channels = vae.latentChannels
            val frames = latents.length / (height * width * channels)
            val rows = MiniMaxH3Conditions
              .patchify(latents, frames, height, width, channels)
            (rows.grouped(rows.length / frames).toSeq, VideoTokenPixels)
          },
          () => vae.close(),
          multiple = VideoTokenPixels,
          // its encoder takes a clip tile by tile, a minute a picture at 512:
          // one tile
          longestSide = 256
        )
      case other =>
        throw new IllegalArgumentException(s"no VAE family named $other")
    }
  }

  /** The pixels a video family's token covers along a side. */
  private val VideoTokenPixels = 32

  /** The longest side a video VAE encodes a picture at: a clip of it. */
  private val VideoLongestSide = 768

  /** The frames of the clip a video VAE reads a picture as: the fewest that
    * give two latent frames.
    */
  private object VideoFrames {
    val Ltx = 9
    val MiniMaxH3 = 5
  }

  /** `picture` with its channels in another order, one of the five there are by
    * `index`, and every other one in negative.
    */
  private def recoloured(picture: BufferedImage, index: Int): BufferedImage = {
    val order = Seq(0, 1, 2).permutations.toSeq(index % 5 + 1)
    val negative = if (index % 2 == 1) 0xff else 0
    val out = new BufferedImage(
      picture.getWidth,
      picture.getHeight,
      BufferedImage.TYPE_INT_RGB
    )
    for {
      y <- 0 until picture.getHeight
      x <- 0 until picture.getWidth
    } {
      val rgb = picture.getRGB(x, y)
      val channels =
        Seq((rgb >> 16) & 0xff, (rgb >> 8) & 0xff, rgb & 0xff)
      val Seq(red, green, blue) = order.map(channels(_) ^ negative)
      out.setRGB(x, y, (red << 16) | (green << 8) | blue)
    }
    out
  }

  /** `picture` with each `box × box` pixels averaged into one: `[H / box, W /
    * box, 3]` in [−1, 1].
    */
  private def averaged(picture: BufferedImage, box: Int): Array[Float] = {
    val full = Images.pixels(picture)
    val (height, width) = (picture.getHeight / box, picture.getWidth / box)
    val out = new Array[Float](height * width * 3)
    for {
      y <- 0 until height
      x <- 0 until width
      c <- 0 until 3
    } {
      var sum = 0f
      for {
        dy <- 0 until box
        dx <- 0 until box
      } sum += full(((y * box + dy) * picture.getWidth + x * box + dx) * 3 + c)
      out((y * width + x) * 3 + c) = sum / (box * box)
    }
    out
  }

  /** The root mean square of `projection`'s error on `samples`. */
  private def error(
      projection: LatentProjection,
      samples: Seq[(Array[Float], Array[Float], Int, Int)]
  ): Double = {
    var (sum, count) = (0.0, 0L)
    samples.foreach { (tokens, pixels, gridHeight, gridWidth) =>
      val drawn =
        Images.pixels(projection.picture(tokens, gridHeight, gridWidth))
      pixels.indices.foreach { i =>
        val difference = drawn(i) - pixels(i)
        sum += difference * difference
      }
      count += pixels.length
    }
    math.sqrt(sum / math.max(1, count))
  }
}
