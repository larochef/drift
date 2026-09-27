package drift.runner

import drift.runner.models.WanVae
import drift.runner.ops.Ops
import drift.runner.tensor.Shape

/** The golden tiny Wan 2.1 VAE decoder (`fixtures/tiny_diffusion.py`) on a
  * backend: the fixture's latents decoded, compared with diffusers'.
  */
object TinyWanVaeCase {

  /** The worst error of the image relative to its largest magnitude. */
  def imageError(ops: Ops): Double = {
    val vae = WanVae.open(ops, Fixtures.path("tiny/wan_vae/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/wan_vae/expected.safetensors") { golden =>
        val latents =
          ops.fromFloats(Shape.of(4, 4, 16), golden("latents").decode())
        val image = vae.decode(latents)
        val (actual, expected) = (ops.toFloats(image), golden("image").decode())
        ops.release(image)
        val scale = expected.map(math.abs).max.toDouble
        expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
      }
    finally vae.close()
  }
}
