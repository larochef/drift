package drift.runner

import drift.runner.diffusion.FlowSchedule
import drift.runner.formats.SafetensorsModel
import drift.runner.lora.Lora
import drift.runner.models.Flux2.Grid
import drift.runner.models.MossImage
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape, Tensor}

/** The golden tiny MOSS-Image transformer (`fixtures/tiny_moss_image.py`,
  * OpenMOSS's own modules) on a backend: the text refiner's output, one
  * velocity, the same without block 1, and with its LoRA — from the weights as
  * released (q, k and v apart) or fused (`proj.qkv`). And the scheduler's
  * levels.
  */
object TinyMossImageCase {

  private val Folder = "tiny/moss_image"

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worst error of the text refiner's output and of the velocities (whole,
    * then without block 1) of `weights`, relative to their largest magnitudes.
    */
  def errors(ops: Ops, weights: String): (Double, Double, Double) = {
    val model = MossImage.open(ops, Fixtures.path(s"$Folder/$weights"))
    try
      Fixtures.withSafetensors(s"$Folder/expected.safetensors") { golden =>
        val (text, embedded) = embed(ops, model, golden)
        val velocity = run(ops, model, golden, text, Set.empty)
        val skipped = run(ops, model, golden, text, Set(1))
        (
          relativeError(embedded, golden("embedded").decode()),
          relativeError(velocity, golden("velocity").decode()),
          relativeError(skipped, golden("skipped_velocity").decode())
        )
      }
    finally model.close()
  }

  /** With the LoRA at 0.8 (OpenMOSS's names, split and fused, and kohya's; all
    * its targets matched) on the fused weights, then without it again: the
    * worst of both errors.
    */
  def loraVelocityError(ops: Ops): Double = {
    val lora = Lora.open(ops, Fixtures.path(s"$Folder/lora.safetensors"))
    val model =
      MossImage.open(ops, Fixtures.path(s"$Folder/model_fused.safetensors"))
    try
      Fixtures.withSafetensors(s"$Folder/expected.safetensors") { golden =>
        val unmatched = model.useLoras(Seq(lora -> 0.8f))
        assert(unmatched.isEmpty, s"unmatched LoRA targets: $unmatched")
        val withLora =
          run(ops, model, golden, embed(ops, model, golden)._1, Set.empty)
        model.useLoras(Nil)
        val back =
          run(ops, model, golden, embed(ops, model, golden)._1, Set.empty)
        math.max(
          relativeError(withLora, golden("lora_velocity").decode()),
          relativeError(back, golden("velocity").decode())
        )
      }
    finally {
      model.close()
      lora.close()
    }
  }

  /** The scheduler's levels for each step count and shift the fixture holds,
    * the worst difference and whether every count matched.
    */
  def scheduleError(): (Double, Boolean) =
    Fixtures.withSafetensors(s"$Folder/expected.safetensors") { golden =>
      val Name = """schedule_(\d+)_(\d+)""".r
      val compared = golden.tensors.keys.toSeq.collect {
        case name @ Name(steps, shift) =>
          val expected = golden(name).decode()
          val actual = FlowSchedule.snapped(steps.toInt, shift.toDouble)
          (
            actual
              .zip(expected)
              .map((a, e) => math.abs(a - e).toDouble)
              .maxOption
              .getOrElse(0.0),
            actual.size == expected.length
          )
      }
      assert(compared.nonEmpty)
      (compared.map(_._1).max, compared.forall(_._2))
    }

  /** The fixture's text through the refiner: the tensor every step reads, and
    * its values.
    */
  private def embed(
      ops: Ops,
      model: MossImage,
      golden: SafetensorsModel
  ) = {
    val features =
      ops.fromFloats(golden("text").shape, golden("text").decode())
    val length = features.shape.dimensions.head
    val text = ops.allocate(DType.F32, Shape.of(length, model.config.hidden))
    model.embedText(features, text)
    (text, ops.toFloats(text))
  }

  private def run(
      ops: Ops,
      model: MossImage,
      golden: SafetensorsModel,
      text: Tensor,
      skip: Set[Int]
  ): Array[Float] = {
    val sides = golden("grid").decode().map(_.toInt)
    val grid = Grid(sides(0), sides(1))
    val latents =
      ops.fromFloats(golden("latents").shape, golden("latents").decode())
    val out = ops.allocate(DType.F32, latents.shape)
    model.velocity(
      latents,
      grid,
      text,
      golden("timestep").decode().head,
      skip,
      out
    )
    ops.toFloats(out)
  }
}
