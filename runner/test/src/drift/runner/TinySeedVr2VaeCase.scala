package drift.runner

import drift.runner.models.SeedVr2Vae
import drift.runner.ops.Ops
import drift.runner.tensor.{Shape, Tensor}

/** The golden tiny SeedVR2 VAE (`fixtures/tiny_seedvr2.py`, the reference's
  * `VideoAutoencoderKLWrapper`) on a backend: one frame and 5 frames of 48 × 64
  * encoded (the posterior's mode) and a latent of each length decoded.
  */
object TinySeedVr2VaeCase {

  private val Folder = "tiny/seedvr2_vae"

  /** Frame `t` of a `[C, T, H, W]` video, channels-last. */
  private def frame(
      values: Array[Float],
      dimensions: Seq[Int],
      t: Int
  ): Array[Float] = {
    val Seq(channels, frames, height, width) = dimensions
    Array.tabulate(height * width * channels) { i =>
      val (pixel, c) = (i / channels, i % channels)
      values((c * frames + t) * height * width + pixel)
    }
  }

  private def worst(actual: Seq[Array[Float]], expected: Seq[Array[Float]]) = {
    assert(actual.size == expected.size)
    val scale = expected.flatMap(_.map(math.abs)).max.toDouble
    actual
      .zip(expected)
      .flatMap { (a, e) =>
        assert(a.length == e.length)
        e.indices.map(i => math.abs(a(i) - e(i)))
      }
      .max / scale
  }

  /** The worst errors of the latents and of the decoded frames, each relative
    * to its largest magnitude, for `subject` (`picture` or `clip`).
    */
  def errors(ops: Ops, subject: String): (Double, Double) = {
    val vae = SeedVr2Vae.open(ops, Fixtures.path(s"$Folder/model.safetensors"))
    try
      Fixtures.withSafetensors(s"$Folder/expected.safetensors") { golden =>
        def frames(name: String): (Seq[Array[Float]], Seq[Int]) = {
          val tensor = golden(s"$subject.$name")
          val dimensions = tensor.shape.dimensions.map(_.toInt)
          val values = tensor.decode()
          (
            (0 until dimensions(1)).map(frame(values, dimensions, _)),
            dimensions
          )
        }
        def upload(
            images: Seq[Array[Float]],
            dimensions: Seq[Int]
        ): Seq[Tensor] =
          images.map(
            ops.fromFloats(
              Shape.of(dimensions(2), dimensions(3), dimensions(0)),
              _
            )
          )

        val (pixels, pixelDimensions) = frames("pixels")
        val video = upload(pixels, pixelDimensions)
        val latents = vae.encode(video)
        val encoded = latents.map(ops.toFloats)
        (video ++ latents).foreach(ops.release)

        val (noise, latentDimensions) = frames("latent")
        val inputs = upload(noise, latentDimensions)
        val decoded = Seq.newBuilder[Array[Float]]
        vae.decode(inputs, image => decoded += ops.toFloats(image))
        inputs.foreach(ops.release)

        (
          worst(encoded, frames("mean")._1),
          worst(decoded.result(), frames("latent_decoded")._1)
        )
      }
    finally vae.close()
  }
}
