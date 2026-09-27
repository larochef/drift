package drift.runner

import java.nio.file.Paths
import javax.imageio.ImageIO

import drift.runner.diffusion.{ImagePipeline, ImageRequest, Images}
import drift.runner.ops.{HipOps, MatVecInputs}

/** One image (the family by the checkpoint), timed, saved as a PNG; after the
  * positional arguments, `--lora FILE[:MULTIPLIER]` (repeated), `--cfg S` and
  * `--negative TEXT` (guidance, off at the default 1), `--guidance G` (the
  * distilled guidance, 3.5 by default, for the models that embed one), and
  * where the family takes them `--init FILE`, `--strength S`,
  * `--reference FILE` (repeated) and `--llm-vision FILE` (the text encoder's
  * mmproj, which Qwen Image 2.1 edits with).
  *
  * `./mill runner.gpuTest.runMain drift.runner.ImageDemo <diffusion> <vae> <llm> <out.png> "<prompt>" [steps] [size] [seed] [--lora FILE[:M]]… [--cfg S] [--negative TEXT] [--init FILE] [--strength S] [--reference FILE]…`
  */
object ImageDemo {
  def main(arguments: Array[String]): Unit = {
    val (positional, named) = arguments.span(!_.startsWith("--"))
    val Array(diffusion, vae, llm, output, prompt) = positional.take(5)
    val steps = positional.lift(5).fold(4)(_.toInt)
    val size = positional.lift(6).fold(1024)(_.toInt)
    val seed = positional.lift(7).fold(42L)(_.toLong)
    val options = named.grouped(2).map(pair => pair(0) -> pair(1)).toSeq
    def image(path: String) = ImageIO.read(Paths.get(path).toFile)
    val init = options.collectFirst { case ("--init", path) => image(path) }
    val strength =
      options
        .collectFirst { case ("--strength", s) => s.toFloat }
        .getOrElse(0.75f)
    val loras = options.collect { case ("--lora", spec) =>
      spec.split(':') match {
        case Array(path, multiplier) => Paths.get(path) -> multiplier.toFloat
        case _                       => Paths.get(spec) -> 1f
      }
    }
    val cfg =
      options.collectFirst { case ("--cfg", s) => s.toFloat }.getOrElse(1f)
    val guidance = options
      .collectFirst { case ("--guidance", g) => g.toFloat }
      .getOrElse(3.5f)
    val negative =
      options.collectFirst { case ("--negative", text) => text }.getOrElse("")
    val references = options.collect { case ("--reference", path) =>
      image(path)
    }
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val loading = System.nanoTime()
    val pipeline = ImagePipeline.open(
      ops,
      Paths.get(diffusion),
      Paths.get(vae),
      Paths.get(llm),
      options.collectFirst { case ("--tokenizer", path) => Paths.get(path) },
      options.collectFirst { case ("--llm-vision", path) => Paths.get(path) }
    )
    println(
      f"${pipeline.family} loaded in ${(System.nanoTime() - loading) / 1e9}%.1f s"
    )
    try {
      val started = System.nanoTime()
      var last = started
      val image = pipeline.generate(
        ImageRequest(
          prompt,
          negative,
          size,
          size,
          steps,
          cfg,
          guidance,
          seed,
          1.15,
          loras,
          init.map(Images.resized(_, size, size)),
          strength,
          references
        ),
        (step, total) => {
          val now = System.nanoTime()
          println(f"  step $step/$total: ${(now - last) / 1e9}%.2f s")
          last = now
        }
      )
      println(f"image in ${(System.nanoTime() - started) / 1e9}%.1f s")
      ImageIO.write(image, "png", Paths.get(output).toFile)
    } finally {
      pipeline.close()
      ops.close()
    }
  }
}
