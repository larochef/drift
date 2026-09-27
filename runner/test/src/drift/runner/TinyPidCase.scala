package drift.runner

import drift.runner.models.{FluxVae, Pid, WanVae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny PiDs (`fixtures/tiny_pid.py`, the official `PidNet`) on a
  * backend: one velocity at σ 0.866 on a 128 × 64 image (patches of 8), the
  * latent upsampled ×4, a RoPE reference of 32 px, at degrade σ 0 and 0.3; and
  * the official latent encoders of the FLUX.1 and Qwen Image variants.
  */
object TinyPidCase {

  /** The FLUX.2 network (`tiny/pid`: a packed latent) or the FLUX.1 and Qwen
    * Image one (`tiny/pid_sixteen`: 16 channels as they are).
    */
  enum Variant(val folder: String) {
    case Flux2 extends Variant("tiny/pid")
    case Sixteen extends Variant("tiny/pid_sixteen")
  }

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worse of the two velocities' errors, relative to the largest. */
  def velocityError(ops: Ops, variant: Variant = Variant.Flux2): Double = {
    val model =
      Pid.open(
        ops,
        Fixtures.path(s"${variant.folder}/model.safetensors"),
        ropeReference = 32
      )
    try
      Fixtures.withSafetensors(s"${variant.folder}/expected.safetensors") {
        golden =>
          val Seq(_, _, height, width) =
            golden("x").shape.dimensions.map(_.toInt)
          val Seq(_, features, latentHeight, latentWidth) =
            golden("latent").shape.dimensions.map(_.toInt)
          // NCHW → channels-last
          def channelsLast(
              values: Array[Float],
              channels: Int,
              h: Int,
              w: Int
          ) =
            Array.tabulate(h * w * channels) { i =>
              val (pixel, c) = (i / channels, i % channels)
              values(c * h * w + pixel)
            }
          val x = ops.fromFloats(
            Shape.of(height, width, 3),
            channelsLast(golden("x").decode(), 3, height, width)
          )
          val latent = ops.fromFloats(
            Shape.of(latentHeight.toLong * latentWidth, features),
            channelsLast(
              golden("latent").decode(),
              features,
              latentHeight,
              latentWidth
            )
          )
          val Seq(_, length, textWidth) =
            golden("y").shape.dimensions.map(_.toInt)
          val text =
            ops.fromFloats(Shape.of(length, textWidth), golden("y").decode())
          val sigma = golden("t").decode().head / 1000f
          val patch = model.config.patch
          Seq(
            "degrade_sigma" -> "output",
            "degrade_sigma_alt" -> "output_alt"
          ).map { (degrade, expected) =>
            val conditioning = model.conditionOn(
              text,
              latent,
              latentHeight,
              height / patch,
              width / patch,
              golden(degrade).decode().head
            )
            try {
              val out = ops.allocate(DType.F32, x.shape)
              model.velocity(x, sigma, conditioning, out)
              val actual = ops.toFloats(out)
              val wanted =
                channelsLast(golden(expected).decode(), 3, height, width)
              relativeError(actual, wanted)
            } finally conditioning.release()
          }.max
      }
    finally model.close()
  }

  /** The FLUX.1 VAE (`flux_vae.py`'s `AutoEncoder`, no quant convolutions): an
    * image encoded to its normalized latent, and a latent decoded; the worst
    * errors of both.
    */
  def flux1VaeErrors(ops: Ops): (Double, Double) = {
    val vae =
      FluxVae.open(ops, Fixtures.path("tiny/pid_flux1_vae/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/pid_flux1_vae/expected.safetensors") {
        golden =>
          def load(name: String) =
            ops.fromFloats(golden(name).shape, golden(name).decode())
          val Seq(height, width, channels) = golden("latent").shape.dimensions
          val latents = vae.encode(load("image"))
          val decoded = vae.decode(
            load("latent").view(height * width, channels),
            height.toInt
          )
          (
            relativeError(ops.toFloats(latents), golden("latents").decode()),
            relativeError(ops.toFloats(decoded), golden("decoded").decode())
          )
      }
    finally vae.close()
  }

  /** The Wan 2.1 VAE's encoder (`qwenimage_vae.py`'s `WanVAE2d_`, saved in the
    * 3-D checkpoint's shapes): an image encoded to the Qwen Image latent; the
    * worst error.
    */
  def qwenImageLatentError(ops: Ops): Double = {
    val vae = WanVae.open(
      ops,
      Fixtures.path("tiny/pid_qwen_image_vae/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/pid_qwen_image_vae/expected.safetensors") {
        golden =>
          val image =
            ops.fromFloats(golden("image").shape, golden("image").decode())
          relativeError(
            ops.toFloats(vae.encode(image)),
            golden("latents").decode()
          )
      }
    finally vae.close()
  }
}
