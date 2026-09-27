package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.models.Flux2.Grid
import drift.runner.models.{Flux2, FluxVae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny FLUX.2 transformers and VAE (`fixtures/tiny_diffusion.py`)
  * on a backend, compared with diffusers: one velocity with a reference, from
  * `flux2` (Klein) or `flux2_dev` (dev: the guidance embedding, at the
  * fixture's scale); an image encoded to packed latents, and packed latents
  * decoded.
  */
object TinyFlux2Case {

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worst error of `folder`'s velocity relative to its largest magnitude.
    */
  def velocityError(ops: Ops, folder: String): Double =
    run(ops, folder, _ => (), "velocity")

  /** With `folder`'s LoRA at 0.8 (Klein's: both namings, fused and split
    * targets, the final modulation's halves swapped; dev's: kohya's names and
    * the guidance MLP; all their targets matched), then without it again: the
    * worst of both errors.
    */
  def loraVelocityError(ops: Ops, folder: String): Double = {
    val lora = Lora.open(ops, Fixtures.path(s"tiny/$folder/lora.safetensors"))
    try {
      val withLora = run(
        ops,
        folder,
        model => {
          val unmatched = model.useLoras(Seq(lora -> 0.8f))
          assert(unmatched.isEmpty, s"unmatched LoRA targets: $unmatched")
        },
        "lora_velocity"
      )
      val back = run(
        ops,
        folder,
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
      prepare: Flux2 => Unit,
      expectedName: String
  ): Double = {
    val model =
      Flux2.open(ops, Fixtures.path(s"tiny/$folder/model.safetensors"))
    try
      Fixtures.withSafetensors(s"tiny/$folder/expected.safetensors") { golden =>
        prepare(model)
        val config = model.config
        val Seq(gridHeight, gridWidth, referenceHeight, referenceWidth) =
          golden("grid").decode().map(_.toInt).toSeq
        val (grid, referenceGrid) =
          (Grid(gridHeight, gridWidth), Grid(referenceHeight, referenceWidth))
        def tensor(name: String, rows: Int, columns: Int) =
          ops.fromFloats(Shape.of(rows, columns), golden(name).decode())
        val features = tensor("text", 7, config.textWidth)
        val text = ops.allocate(DType.F32, Shape.of(7, config.hidden))
        model.embedText(features, text)
        val latents = tensor("latents", grid.tokens, config.latentChannels)
        val reference =
          tensor("reference", referenceGrid.tokens, config.latentChannels)
        val out = ops.allocate(DType.F32, latents.shape)
        model.velocity(
          latents,
          grid,
          Seq(reference -> referenceGrid),
          text,
          golden("timestep").decode().head,
          if (golden.tensors.contains("guidance"))
            golden("guidance").decode().head
          else 0f,
          out
        )
        relativeError(ops.toFloats(out), golden(expectedName).decode())
      }
    finally model.close()
  }

  /** The worst errors of the encoded latents and the decoded image, the VAE's
    * weights from `file`: the original names (`model.safetensors`) or
    * diffusers' (`diffusers.safetensors`).
    */
  def vaeErrors(ops: Ops, file: String): (Double, Double) = {
    val vae = FluxVae.open(ops, Fixtures.path(s"tiny/flux2_vae/$file"))
    try
      Fixtures.withSafetensors("tiny/flux2_vae/expected.safetensors") {
        golden =>
          val Seq(gridHeight, gridWidth) =
            golden("grid").decode().map(_.toInt).toSeq
          val image = ops.fromFloats(
            Shape.of(
              gridHeight.toLong * vae.patch,
              gridWidth.toLong * vae.patch,
              3
            ),
            golden("image").decode()
          )
          val latents = vae.encode(image)
          val packed = ops.fromFloats(
            Shape.of(gridHeight.toLong * gridWidth, vae.features),
            golden("packed").decode()
          )
          val decoded = vae.decode(packed, gridHeight)
          (
            relativeError(ops.toFloats(latents), golden("latents").decode()),
            relativeError(ops.toFloats(decoded), golden("decoded").decode())
          )
      }
    finally vae.close()
  }
}
