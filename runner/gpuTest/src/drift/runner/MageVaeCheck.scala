package drift.runner

import java.nio.file.Paths
import javax.imageio.ImageIO

import drift.runner.diffusion.Images
import drift.runner.formats.SafetensorsModel
import drift.runner.models.MageVae
import drift.runner.ops.{HipOps, MatVecInputs}

/** The Mage-Flow VAE on its released weights beside the official one: a
  * safetensors of `image`, `moments`, `latent` and `decoded` (channels-last,
  * made on the CPU by microsoft/Mage's `MageVAE` in float32) against the
  * runner's moments of the image and its decoding of the latent, the latter
  * saved as a PNG.
  *
  * `./mill runner.gpuTest.runMain drift.runner.MageVaeCheck <vae> <golden.safetensors> <out.png>`
  */
object MageVaeCheck {
  def main(arguments: Array[String]): Unit = {
    val Array(file, expected, output) = arguments.take(3)
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val vae = MageVae.open(ops, Paths.get(file))
    val golden = SafetensorsModel.open(Paths.get(expected))
    try {
      def load(name: String) =
        ops.fromFloats(golden(name).shape, golden(name).decode())
      def report(name: String, actual: Array[Float]): Unit = {
        val wanted = golden(name).decode()
        val differences =
          wanted.indices.map(i => math.abs(actual(i) - wanted(i)).toDouble)
        println(
          f"$name: worst ${differences.max / wanted.map(math.abs).max * 100}%.3f%% of the largest, mean difference ${differences.sum / differences.size}%.5f"
        )
      }
      report("moments", ops.toFloats(vae.moments(load("image"))))
      val started = System.nanoTime()
      val decoded = ops.toFloats(vae.decode(load("latent")))
      println(f"decoded in ${(System.nanoTime() - started) / 1e9}%.2f s")
      report("decoded", decoded)
      val Seq(height, width, _) =
        golden("decoded").shape.dimensions.map(_.toInt)
      ImageIO.write(
        Images.toImage(decoded, width, height),
        "png",
        Paths.get(output).toFile
      )
    } finally {
      golden.close()
      vae.close()
      ops.close()
    }
  }
}
