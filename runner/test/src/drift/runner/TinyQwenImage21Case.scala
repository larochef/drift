package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.models.{PrefixReference, QwenImage21, QwenImage21Vae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny Qwen Image 2.1 (`fixtures/tiny_diffusion.py`) on a backend:
  * the transformer's velocity from the fixture's text made a prefix, with and
  * without a LoRA, and the VAE both ways, compared with diffusers'.
  */
object TinyQwenImage21Case {

  /** The worst error of the velocity relative to its largest magnitude. */
  def velocityError(ops: Ops): Double = run(ops, _ => (), "velocity")

  /** With the fixture's LoRA at 0.8 (all its targets matched), then without it
    * again: the worst of both errors.
    */
  def loraVelocityError(ops: Ops): Double = {
    val lora =
      Lora.open(ops, Fixtures.path("tiny/qwen_image21/lora.safetensors"))
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

  /** Editing: two references among the text (slots 2–5 and 7–9) in the prefix,
    * then the image's velocity.
    */
  def editError(ops: Ops): Double = {
    val model =
      QwenImage21.open(
        ops,
        Fixtures.path("tiny/qwen_image21/model.safetensors")
      )
    try
      Fixtures.withSafetensors("tiny/qwen_image21/expected.safetensors") {
        golden =>
          def tensor(name: String, rows: Int) =
            ops.fromFloats(Shape.of(rows, 64), golden(name).decode())
          val (text, first, second, latents) = (
            tensor("edit_text", 12),
            tensor("edit_first", 16),
            tensor("edit_second", 12),
            tensor("edit_latents", 24)
          )
          val out = ops.allocate(DType.F32, latents.shape)
          val prefix = model.prefix(
            text,
            24,
            Seq(
              PrefixReference(7, 3, second, 2, 6),
              PrefixReference(2, 4, first, 4, 4)
            )
          )
          try {
            model.velocity(latents, prefix, 0.75f, 4, 6, out)
            relativeError(ops.toFloats(out), golden("edit_velocity").decode())
          } finally {
            prefix.close()
            Seq(text, first, second, latents, out).foreach(ops.release)
          }
      }
    finally model.close()
  }

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  private def run(
      ops: Ops,
      prepare: QwenImage21 => Unit,
      expectedName: String
  ): Double = {
    val model =
      QwenImage21.open(
        ops,
        Fixtures.path("tiny/qwen_image21/model.safetensors")
      )
    try {
      prepare(model)
      Fixtures.withSafetensors("tiny/qwen_image21/expected.safetensors") {
        golden =>
          val config = model.config
          val Seq(gridHeight, gridWidth) =
            golden("grid").decode().map(_.toInt).toSeq
          val images = gridHeight * gridWidth
          val values = golden("text").decode()
          val text = ops.fromFloats(
            Shape.of(values.length / config.textWidth, config.textWidth),
            values
          )
          val latents = ops.fromFloats(
            Shape.of(images, config.latentChannels),
            golden("latents").decode()
          )
          val out = ops.allocate(DType.F32, latents.shape)
          val prefix = model.prefix(text, images)
          try
            model.velocity(
              latents,
              prefix,
              golden("timestep").decode().head,
              gridHeight,
              gridWidth,
              out
            )
          finally prefix.close()
          val error =
            relativeError(ops.toFloats(out), golden(expectedName).decode())
          Seq(text, latents, out).foreach(ops.release)
          error
      }
    } finally model.close()
  }

  /** The worst errors of the encoded latents and the decoded image, each
    * relative to its largest magnitude.
    */
  def vaeErrors(ops: Ops): (Double, Double) = {
    val vae = QwenImage21Vae.open(
      ops,
      Fixtures.path("tiny/qwen_image21_vae/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/qwen_image21_vae/expected.safetensors") {
        golden =>
          val Seq(gridHeight, gridWidth) =
            golden("grid").decode().map(_.toInt).toSeq
          val image = ops.fromFloats(
            Shape.of(
              gridHeight.toLong * vae.scale,
              gridWidth.toLong * vae.scale,
              vae.imageChannels
            ),
            golden("image").decode()
          )
          val latents = vae.encode(image)
          val packed = ops.fromFloats(
            Shape.of(gridHeight, gridWidth, vae.channels),
            golden("packed").decode()
          )
          val decoded = vae.decode(packed)
          val errors = (
            relativeError(ops.toFloats(latents), golden("latents").decode()),
            relativeError(ops.toFloats(decoded), golden("decoded").decode())
          )
          Seq(image, latents, packed, decoded).foreach(ops.release)
          errors
      }
    finally vae.close()
  }
}
