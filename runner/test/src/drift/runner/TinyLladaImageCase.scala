package drift.runner

import drift.runner.models.Flux2.Grid
import drift.runner.models.{
  Llada2,
  LladaImage,
  LladaQueryFormer,
  LladaSigVq,
  LladaTextProjection,
  WeightSource
}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny LLaDA-Image (`fixtures/tiny_llada_image.py`, inclusionAI's
  * own modules) on a backend.
  */
object TinyLladaImageCase {

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The transformer's velocity at σ 0.5 of a 5 × 6 latent on 7 caption tokens:
    * text to image, and editing (a source latent and 5 semantic tokens); both
    * errors, relative to the largest.
    */
  def velocityErrors(ops: Ops): (Double, Double) = {
    val model =
      LladaImage.open(ops, Fixtures.path("tiny/llada_image/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/llada_image/expected.safetensors") {
        golden =>
          def load(name: String) =
            ops.fromFloats(golden(name).shape, golden(name).decode())
          val Seq(height, width) = golden("grid").decode().map(_.toInt).toSeq
          val grid = Grid(height, width)
          val sigma = golden("sigma").decode().head
          val out = ops.allocate(
            DType.F32,
            Shape.of(grid.tokens, model.config.latentChannels)
          )
          def error(editing: Boolean) = {
            val condition = model.condition(
              load("caption"),
              Option.when(editing)(load("semantic")),
              grid.tokens,
              editing
            )
            try {
              model.velocity(
                load("latents"),
                grid,
                condition,
                Option.when(editing)(load("source")),
                sigma,
                out
              )
              // the model gives the velocity's opposite
              relativeError(
                ops.toFloats(out),
                golden(if (editing) "output_editing" else "output")
                  .decode()
                  .map(-_)
              )
            } finally condition.close()
          }
          (error(false), error(true))
      }
    finally model.close()
  }

  /** The QueryFormer's queries of 7 token embeddings and the text projection's
    * caption features of 15 hidden states; both errors.
    */
  def connectorErrors(ops: Ops): (Double, Double) = {
    val source = WeightSource.open(
      ops,
      Fixtures.path("tiny/llada_connectors/model.safetensors")
    )
    val queryFormer = new LladaQueryFormer(ops, source, "queryformer.")
    val projection = new LladaTextProjection(ops, source, "text_projection.")
    try
      Fixtures.withSafetensors("tiny/llada_connectors/expected.safetensors") {
        golden =>
          def load(name: String) =
            ops.fromFloats(golden(name).shape, golden(name).decode())
          (
            relativeError(
              ops.toFloats(queryFormer(load("embeds"))),
              golden("queries").decode()
            ),
            relativeError(
              ops.toFloats(projection(load("hidden"))),
              golden("projected").decode()
            )
          )
      }
    finally {
      queryFormer.close()
      projection.close()
      source.close()
    }
  }

  /** The LLaDA2 text model's last hidden state of 7 token embeddings then 8
    * queries, the text blind to the queries; its error.
    */
  def backboneError(ops: Ops): Double = {
    val model = Llada2.open(ops, Fixtures.path("tiny/llada2/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/llada2/expected.safetensors") { golden =>
        val embeds =
          ops.fromFloats(golden("embeds").shape, golden("embeds").decode())
        relativeError(
          ops.toFloats(
            model.encode(embeds, golden("text").decode().head.toInt)
          ),
          golden("hidden").decode()
        )
      }
    finally model.close()
  }

  /** SigVQ on a 48 × 32 image: whether its codes are the official ones, and its
    * semantic features' error.
    */
  def sigvqError(ops: Ops): (Boolean, Double) = {
    val source = WeightSource.open(
      ops,
      Fixtures.path("tiny/llada_connectors/model.safetensors")
    )
    val sigvq = new LladaSigVq(ops, source, "sigvq.")
    try
      Fixtures.withSafetensors("tiny/llada_connectors/expected.safetensors") {
        golden =>
          val image =
            ops.fromFloats(golden("image").shape, golden("image").decode())
          val codes = sigvq.tokens(image)
          (
            codes.toSeq == golden("tokens").decode().map(_.toInt).toSeq,
            relativeError(
              ops.toFloats(sigvq.semantic(codes)),
              golden("semantic").decode()
            )
          )
      }
    finally {
      sigvq.close()
      source.close()
    }
  }
}
