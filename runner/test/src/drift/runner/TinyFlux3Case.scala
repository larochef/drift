package drift.runner

import drift.runner.lora.Lora
import drift.runner.models.Flux2.Grid
import drift.runner.models.Flux3
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny FLUX 3 transformers (`fixtures/tiny_flux3.py`, BFL's own
  * modules) on a backend: one velocity of `flux3` (the flow map, two
  * references) or `flux3_base` (one reference), and `flux3`'s with its LoRA.
  */
object TinyFlux3Case {

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worst error of `folder`'s velocity relative to its largest magnitude.
    */
  def velocityError(ops: Ops, folder: String): Double =
    run(ops, folder, _ => (), "velocity")

  /** With `flux3`'s LoRA at 0.8 (BFL's names, the MLP's fused input whole, and
    * kohya's; all its targets matched), then without it again: the worst of
    * both errors.
    */
  def loraVelocityError(ops: Ops): Double = {
    val lora = Lora.open(ops, Fixtures.path("tiny/flux3/lora.safetensors"))
    try {
      val withLora = run(
        ops,
        "flux3",
        model => {
          val unmatched = model.useLoras(Seq(lora -> 0.8f))
          assert(unmatched.isEmpty, s"unmatched LoRA targets: $unmatched")
        },
        "lora_velocity"
      )
      val back = run(
        ops,
        "flux3",
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
      folder: String,
      prepare: Flux3 => Unit,
      expectedName: String
  ): Double = {
    val model =
      Flux3.open(ops, Fixtures.path(s"tiny/$folder/model.safetensors"))
    try
      Fixtures.withSafetensors(s"tiny/$folder/expected.safetensors") { golden =>
        prepare(model)
        val config = model.config
        val sides = golden("grid").decode().map(_.toInt).toSeq
        val grid = Grid(sides(0), sides(1))
        val referenceGrids =
          sides.drop(2).grouped(2).map(pair => Grid(pair(0), pair(1))).toSeq
        def tensor(name: String) =
          ops.fromFloats(golden(name).shape, golden(name).decode())
        val features = tensor("text")
        val length = features.shape.dimensions.head
        val text = ops.allocate(DType.F32, Shape.of(length, config.hidden))
        model.embedText(features, text)
        val latents = tensor("latents")
        val references = tensor("references")
        val placed = referenceGrids
          .scanLeft(0L)(_ + _.tokens)
          .zip(referenceGrids)
          .map((first, g) => references.rows(first, g.tokens) -> g)
        val out = ops.allocate(DType.F32, latents.shape)
        model.velocity(
          latents,
          grid,
          placed,
          text,
          golden("timestep").decode().head,
          Option.when(golden.tensors.contains("next"))(
            golden("next").decode().head
          ),
          out
        )
        relativeError(ops.toFloats(out), golden(expectedName).decode())
      }
    finally model.close()
  }
}
