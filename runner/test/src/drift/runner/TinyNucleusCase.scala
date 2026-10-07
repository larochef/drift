package drift.runner

import drift.runner.models.Nucleus
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny Nucleus-Image (`fixtures/tiny_nucleus.py`, diffusers'
  * `NucleusMoEImageTransformer2DModel`) on a backend: one velocity at σ 0.75 of
  * a 5 × 6 grid on 7 text tokens, through a dense block and two whose experts
  * choose their tokens (all of them, then half).
  */
object TinyNucleusCase {

  /** The velocity's worst error, relative to the largest. */
  def velocityError(ops: Ops): Double = {
    val model =
      Nucleus.open(ops, Fixtures.path("tiny/nucleus/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/nucleus/expected.safetensors") { golden =>
        def load(name: String) =
          ops.fromFloats(golden(name).shape, golden(name).decode())
        val Seq(height, width) = golden("grid").decode().map(_.toInt).toSeq
        val text = model.text(load("text"), height, width)
        try {
          val out = ops.allocate(
            DType.F32,
            Shape.of(height.toLong * width, model.config.outChannels)
          )
          model.velocity(
            load("latents"),
            text,
            golden("sigma").decode().head,
            height,
            width,
            out
          )
          // diffusers' model gives the velocity's opposite
          val expected = golden("output").decode().map(-_)
          val actual = ops.toFloats(out)
          val scale = expected.map(math.abs).max.toDouble
          expected.indices
            .map(i => math.abs(actual(i) - expected(i)))
            .max / scale
        } finally text.close()
      }
    finally model.close()
  }
}
