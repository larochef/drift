package drift.runner

import drift.runner.diffusion.{Images, SeedVr2Options, SeedVr2Pipeline}
import drift.runner.ops.{HipOps, MatVecInputs}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*

/** The released SeedVR2 on the GPU, against the reference's goldens
  * (`fixtures/tiny_seedvr2.py picture clip`) or on a picture of one's own:
  * `MODEL VAE golden picture|clip|picture_7b`, `MODEL VAE upscale IN.png
  * OUT.png SHORT_SIDE [SEED]` (the input resized so its short side is
  * `SHORT_SIDE`, padded to multiples of 16, restored, cropped; no colour
  * match), or `MODEL VAE video IN_DIR OUT_DIR SHORT_SIDE BATCH OVERLAP` (a
  * folder of PNG frames, in name order, as one video).
  */
object SeedVr2Check {

  private def timed[A](what: String)(body: => A): A = {
    val start = System.nanoTime()
    val result = body
    println(f"  $what: ${(System.nanoTime() - start) / 1e9}%.1f s")
    result
  }

  /** A stage callback that prints the time each stage took. */
  final private class Stages {
    private var (last, started) = ("", System.nanoTime())
    def apply(name: String): Unit = {
      val now = System.nanoTime()
      if (last.nonEmpty) println(f"  $last: ${(now - started) / 1e9}%.1f s")
      last = name
      started = now
    }
  }

  private def golden(pipeline: SeedVr2Pipeline, subject: String) =
    Fixtures.withSafetensors(s"tiny/seedvr2_$subject/expected.safetensors") {
      expected =>
        val Seq(_, count, height, width) =
          expected("video").shape.dimensions.map(_.toInt)
        val video = expected("video").decode()
        // [3, T, H, W] → frames, channels-last
        val frames = (0 until count).map { t =>
          Array.tabulate(height * width * 3) { i =>
            val (pixel, c) = (i / 3, i % 3)
            video((c * count + t) * height * width + pixel)
          }
        }
        val noise = expected("noise").decode()
        val own = pipeline.noise(42, count, height, width)
        println(
          s"  seed 42's noise differs from the reference's by ${noise.indices.map(i => math.abs(noise(i) - own(i))).max}"
        )
        val stages = new Stages
        val images =
          pipeline.restoreBatch(frames, height, width, noise, stage = stages(_))
        stages("")
        // [T, 3, H, W]
        val decoded = expected("decoded").decode()
        val scale = decoded.map(math.abs).max
        val errors = for {
          t <- 0 until count
          i <- 0 until height * width * 3
        } yield math.abs(
          images(t)(i) - decoded((t * 3 + i % 3) * height * width + i / 3)
        )
        println(
          f"  decoded frames: worst error ${errors.max / scale * 100}%.3f%%, mean ${errors.sum / errors.size / scale * 100}%.4f%% of the largest"
        )
    }

  /** `source` resized so its short side is `shortSide`, on a black canvas of
    * the next multiples of 16 (as the reference pads): its pixels, the
    * picture's size and the canvas's.
    */
  private def prepared(
      pipeline: SeedVr2Pipeline,
      source: BufferedImage,
      shortSide: Int
  ): (Array[Float], (Int, Int), (Int, Int)) = {
    val factor =
      shortSide.toDouble / math.min(source.getWidth, source.getHeight)
    def even(side: Int) = math.round(side * factor).toInt / 2 * 2
    val (width, height) = (even(source.getWidth), even(source.getHeight))
    def padded(side: Int) =
      (side + pipeline.multiple - 1) / pipeline.multiple * pipeline.multiple
    val (fullWidth, fullHeight) = (padded(width), padded(height))
    val canvas =
      new BufferedImage(fullWidth, fullHeight, BufferedImage.TYPE_INT_RGB)
    val graphics = canvas.createGraphics()
    try graphics.drawImage(Images.resized(source, width, height), 0, 0, null)
    finally graphics.dispose()
    (Images.pixels(canvas), (width, height), (fullWidth, fullHeight))
  }

  /** `inputs` restored as one video into `outputs`, with the options given. */
  private def upscale(
      pipeline: SeedVr2Pipeline,
      inputs: Seq[Path],
      outputs: Seq[Path],
      shortSide: Int,
      seed: Long,
      options: SeedVr2Options
  ): Unit = {
    val first = ImageIO.read(inputs.head.toFile)
    val (_, (width, height), (fullWidth, fullHeight)) =
      prepared(pipeline, first, shortSide)
    println(
      s"  ${inputs.size} × ${first.getWidth} × ${first.getHeight} → $width × $height"
    )
    val stages = new Stages
    val remaining = outputs.iterator
    pipeline.restore(
      inputs.iterator.map(file =>
        prepared(pipeline, ImageIO.read(file.toFile), shortSide)._1
      ),
      inputs.size,
      fullHeight,
      fullWidth,
      seed,
      image =>
        ImageIO.write(
          Images
            .toImage(image, fullWidth, fullHeight)
            .getSubimage(0, 0, width, height),
          "png",
          remaining.next().toFile
        ),
      options,
      (batch, batches, stage) =>
        stages(s"batch ${batch + 1} of $batches, $stage")
    )
    stages("")
  }

  def main(arguments: Array[String]): Unit = {
    val Array(model, vae, mode, rest*) = arguments: @unchecked
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    try {
      val pipeline = timed("load")(
        new SeedVr2Pipeline(ops, Paths.get(model), Paths.get(vae))
      )
      try
        (mode, rest) match {
          case ("golden", Seq(subject)) => golden(pipeline, subject)
          case ("upscale", Seq(in, out, side, seed*)) =>
            upscale(
              pipeline,
              Seq(Paths.get(in)),
              Seq(Paths.get(out)),
              side.toInt,
              seed.headOption.fold(42L)(_.toLong),
              SeedVr2Options.Default
            )
          case ("video", Seq(in, out, side, batch, overlap)) =>
            val inputs = Files
              .list(Paths.get(in))
              .toList
              .asScala
              .toSeq
              .filter(_.toString.endsWith(".png"))
              .sorted
            Files.createDirectories(Paths.get(out))
            upscale(
              pipeline,
              inputs,
              inputs.map(file => Paths.get(out).resolve(file.getFileName)),
              side.toInt,
              42L,
              SeedVr2Options.Default
                .copy(batch = batch.toInt, overlap = overlap.toInt)
            )
          case _ => sys.error(s"unknown arguments: ${arguments.mkString(" ")}")
        }
      finally pipeline.close()
    } finally ops.close()
  }
}
