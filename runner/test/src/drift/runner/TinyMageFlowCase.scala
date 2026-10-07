package drift.runner

import drift.runner.models.Flux2.Grid
import drift.runner.models.{MageFlow, MageVae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny Mage-Flow (`fixtures/tiny_mage_flow.py`, microsoft/Mage's
  * own modules) on a backend: one velocity at σ 0.7 of a 5 × 6 target, alone
  * and with a 4 × 3 reference after it; and its VAE both ways, a 96 × 80 image
  * to its mean and log variance and a 6 × 5 latent to its image, through
  * windows of 4.
  */
object TinyMageFlowCase {

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worse of the two velocities' errors, relative to the largest. */
  def velocityError(ops: Ops): Double = {
    val model =
      MageFlow.open(ops, Fixtures.path("tiny/mage_flow/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/mage_flow/expected.safetensors") {
        golden =>
          val config = model.config
          def grid(name: String) = {
            val Seq(height, width) = golden(name).decode().map(_.toInt).toSeq
            Grid(height, width)
          }
          val (target, reference) = (grid("target"), grid("reference"))
          val features = ops.fromFloats(
            golden("txt").shape,
            golden("txt").decode()
          )
          val length = features.shape.dimensions.head
          val text = ops.allocate(DType.F32, Shape.of(length, config.hidden))
          model.embedText(features, text)
          val channels = config.latentChannels
          val all = golden("img").decode()
          val latents = ops.fromFloats(
            Shape.of(target.tokens, channels),
            all.take(target.tokens * channels)
          )
          val clean = ops.fromFloats(
            Shape.of(reference.tokens, channels),
            all.drop(target.tokens * channels)
          )
          val sigma = golden("sigma").decode().head
          val out = ops.allocate(DType.F32, latents.shape)
          Seq(
            Seq(clean -> reference) -> "output",
            Nil -> "output_alone"
          ).map { (references, expected) =>
            model.velocity(latents, target, references, text, sigma, out)
            relativeError(
              ops.toFloats(out),
              golden(expected).decode().take(target.tokens * channels)
            )
          }.max
      }
    finally model.close()
  }

  /** The VAE's worst errors: the moments of an image, a latent decoded. */
  def vaeErrors(ops: Ops): (Double, Double) =
    Fixtures.withSafetensors("tiny/mage_vae/expected.safetensors") { golden =>
      val vae = MageVae.open(
        ops,
        Fixtures.path("tiny/mage_vae/model.safetensors"),
        window = golden("window").decode().head.toInt
      )
      try {
        def load(name: String) =
          ops.fromFloats(golden(name).shape, golden(name).decode())
        val moments = vae.moments(load("image"))
        val decoded = vae.decode(load("latent"))
        (
          relativeError(ops.toFloats(moments), golden("moments").decode()),
          relativeError(ops.toFloats(decoded), golden("decoded").decode())
        )
      } finally vae.close()
    }
}
