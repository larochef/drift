package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.models.Krea2
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny Krea 2 transformer (`fixtures/tiny_diffusion.py`) on a
  * backend: the fixture's text taps fused, then one velocity, compared with
  * diffusers'.
  */
object TinyKrea2Case {

  /** The worst error of the velocity relative to its largest magnitude. */
  def velocityError(ops: Ops): Double = run(ops, _ => (), "velocity")

  /** With the fixture's LoRA at 0.8 (all its targets matched), then without it
    * again: the worst of both errors.
    */
  def loraVelocityError(ops: Ops): Double = {
    val lora = Lora.open(ops, Fixtures.path("tiny/krea2/lora.safetensors"))
    try {
      val withLora = run(
        ops,
        model => {
          val unmatched = model.useLoras(Seq(lora -> 0.8f))
          assert(unmatched.isEmpty, s"unmatched LoRA targets: $unmatched")
        },
        "lora_velocity"
      )
      val back = run(
        ops,
        model => {
          model.useLoras(Seq(lora -> 0.8f))
          model.useLoras(Nil)
        },
        "velocity"
      )
      math.max(withLora, back)
    } finally lora.close()
  }

  private def run(
      ops: Ops,
      prepare: Krea2 => Unit,
      expectedName: String
  ): Double = {
    val model = Krea2.open(ops, Fixtures.path("tiny/krea2/model.safetensors"))
    try {
      prepare(model)
      Fixtures.withSafetensors("tiny/krea2/expected.safetensors") { golden =>
        val config = model.config
        val Seq(gridHeight, gridWidth) =
          golden("grid").decode().map(_.toInt).toSeq
        val latents = golden("latents").decode()
        // diffusers takes the text token-major [L, taps, width]; the encoder
        // gives it tap-major [taps × L, width]
        val tokenMajor = golden("text").decode()
        val (layers, width) = (config.textLayers, config.textWidth)
        val tokens = tokenMajor.length / (layers * width)
        val tapMajor = Array.tabulate(layers * tokens * width) { i =>
          val (tap, token, value) =
            (i / (tokens * width), i / width % tokens, i % width)
          tokenMajor((token * layers + tap) * width + value)
        }
        val taps =
          ops.fromFloats(Shape.of(layers.toLong * tokens, width), tapMajor)
        val text = ops.allocate(DType.F32, Shape.of(tokens, config.hidden))
        model.encodeText(taps, text)
        val latentTensor =
          ops.fromFloats(
            Shape.of(gridHeight * gridWidth, config.latentChannels),
            latents
          )
        val out = ops.allocate(DType.F32, latentTensor.shape)
        model.velocity(
          latentTensor,
          text,
          golden("timestep").decode().head,
          gridHeight,
          gridWidth,
          out
        )
        val (actual, expected) =
          (ops.toFloats(out), golden(expectedName).decode())
        val scale = expected.map(math.abs).max.toDouble
        expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
      }
    } finally model.close()
  }
}
