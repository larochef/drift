package drift.runner

import drift.runner.models.{Grn, QwenImage21Vae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny GRN (`fixtures/tiny_grn.py`, bytedance's own modules) on a
  * backend.
  */
object TinyGrnCase {

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** One refinement pass of a 5 × 6 grid of bits on 7 text tokens at progress
    * 0.3: the logits' error, relative to the largest.
    */
  def logitsError(ops: Ops): Double = {
    val model = Grn.open(ops, Fixtures.path("tiny/grn/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/grn/expected.safetensors") { golden =>
        val Seq(height, width) = golden("grid").decode().map(_.toInt).toSeq
        val bits = golden("bits").decode()
        val oneHot = new Array[Float](2 * bits.length)
        bits.indices.foreach(i => oneHot(2 * i + bits(i).toInt) = 1f)
        val shape = Shape.of(height.toLong * width, 2L * model.config.bits)
        val text = model.text(
          ops.fromFloats(golden("text").shape, golden("text").decode())
        )
        val out = ops.allocate(DType.F32, shape)
        model.logits(
          ops.fromFloats(shape, oneHot),
          text,
          golden("progress").decode().head,
          height,
          width,
          out
        )
        relativeError(ops.toFloats(out), golden("logits").decode())
      }
    finally model.close()
  }

  /** The HBQ tokenizer's decoder on a 3 × 2 latent: the image's error. */
  def tokenizerError(ops: Ops): Double = {
    val tokenizer = QwenImage21Vae.open(
      ops,
      Fixtures.path("tiny/grn_tokenizer/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/grn_tokenizer/expected.safetensors") {
        golden =>
          val latent =
            ops.fromFloats(golden("latent").shape, golden("latent").decode())
          relativeError(
            ops.toFloats(tokenizer.decode(latent)),
            golden("decoded").decode()
          )
      }
    finally tokenizer.close()
  }
}
