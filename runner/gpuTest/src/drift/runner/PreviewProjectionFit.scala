package drift.runner

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO

import drift.runner.diffusion.{Images, LatentProjection}
import drift.runner.models.{FluxVae, MageVae, QwenImage21Vae, WanVae}
import drift.runner.ops.{HipOps, MatVecInputs, Ops}
import drift.runner.tensor.{DType, Shape}

/** Fits a VAE family's step-preview projection (`specs/54-step-preview.md`):
  * encodes pictures with the family's VAE as its pipeline does, regresses each
  * token's `block × block` pixels of the picture on the token, and writes the
  * projection where the runner reads it (`runner/resources/previews`). Each
  * picture is encoded a second time with its colours swapped around
  * (`recoloured`), so the fit sees colours the pictures lack. Every fifth
  * picture is held out of a first fit, which is what the printed error is
  * measured on; the projection written is fitted on all of them. Each picture
  * beside its projection goes to `pairs` as a PNG. The time a step's preview
  * takes is printed too: the two tensors read back, the picture drawn and its
  * JPEG written, on the first picture's grid.
  *
  * `./mill runner.gpuTest.runMain drift.runner.PreviewProjectionFit <family>
  * <vae> <block> <projection.txt> <pairs> <picture>…`, `family` one of
  * `LatentProjection`'s names.
  */
object PreviewProjectionFit {

  /** The longest side a picture is encoded at. */
  private val LongestSide = 1024

  def main(arguments: Array[String]): Unit = {
    val Array(family, file, blockText, output, pairs) = arguments.take(5)
    val pictures = arguments.drop(5).toSeq.map(Paths.get(_))
    val block = blockText.toInt
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val (encode, close) = encoder(ops, family, Paths.get(file))
    try {
      val names = pictures.flatMap { path =>
        val own = path.getFileName.toString.stripSuffix(".png")
        Seq(own, s"$own-recoloured")
      }
      val samples = pictures.zipWithIndex.flatMap { (path, index) =>
        val read = ImageIO.read(path.toFile)
        val reduction =
          math.min(
            1.0,
            LongestSide.toDouble / (read.getWidth max read.getHeight)
          )
        def side(length: Int) = (length * reduction).toInt / 16 * 16
        val own =
          Images.resized(read, side(read.getWidth), side(read.getHeight))
        Seq(own, recoloured(own, index)).map { picture =>
          val (tokens, scale) = encode(picture)
          val (gridHeight, gridWidth) =
            (picture.getHeight / scale, picture.getWidth / scale)
          println(s"${path.getFileName}: $gridHeight × $gridWidth tokens")
          (tokens, averaged(picture, scale / block), gridHeight, gridWidth)
        }
      }
      val channels = samples.head._1.length /
        (samples.head._3 * samples.head._4)
      val (heldOut, kept) =
        samples.zipWithIndex.partition((_, index) => index / 2 % 5 == 4)
      val partial = LatentProjection.fit(channels, block, kept.map(_._1))
      println(
        f"$family, block $block: held-out error ${error(partial, heldOut.map(_._1))}%.4f, " +
          f"fitted ${error(partial, kept.map(_._1))}%.4f (RMS, colours in [−1, 1])"
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

  /** How `family` turns a picture (sides multiples of 16) into the tokens its
    * transformer takes, with the pixels a token covers along a side; and what
    * closes the VAE.
    */
  private def encoder(
      ops: Ops,
      family: String,
      file: Path
  ): (BufferedImage => (Array[Float], Int), () => Unit) = {
    def pixels(picture: BufferedImage, alpha: Boolean) = ops.fromFloats(
      Shape.of(picture.getHeight, picture.getWidth, if (alpha) 4 else 3),
      Images.pixels(picture, alpha)
    )
    family match {
      case LatentProjection.Wan21 =>
        val vae = WanVae.open(ops, file)
        (
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
              (ops.toFloats(packed), 2 * vae.scale)
            } finally {
              ops.release(encoded)
              ops.release(packed)
            }
          },
          () => vae.close()
        )
      case LatentProjection.Flux2 =>
        val vae = FluxVae.open(ops, file)
        (
          picture => {
            val image = pixels(picture, alpha = false)
            val encoded =
              try vae.encode(image)
              finally ops.release(image)
            try (ops.toFloats(encoded), vae.patch)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case LatentProjection.QwenImage21 =>
        val vae = QwenImage21Vae.open(ops, file)
        (
          picture => {
            val image = pixels(picture, alpha = vae.imageChannels == 4)
            val encoded =
              try vae.encode(image)
              finally ops.release(image)
            try (ops.toFloats(encoded), vae.scale)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case LatentProjection.MageFlow =>
        val vae = MageVae.open(ops, file)
        (
          picture => {
            val image = pixels(picture, alpha = false)
            val encoded =
              try vae.encode(image, None)
              finally ops.release(image)
            try (ops.toFloats(encoded), vae.scale)
            finally ops.release(encoded)
          },
          () => vae.close()
        )
      case other =>
        throw new IllegalArgumentException(s"no VAE family named $other")
    }
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
