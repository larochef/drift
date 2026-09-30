package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** UMT5's encoder (transformers' `UMT5EncoderModel`; Wan's text encoder,
  * `umt5-xxl`), from transformers' names: pre-norm blocks of attention (no
  * score scaling, each block's own relative-position bias over 32 buckets up to
  * 128 apart, both directions) and a gated GELU (tanh) MLP, RMS norms without a
  * mean (T5's), a final norm. fp8 files (ComfyUI's) come as BF16 through
  * `WeightSource`. `encode` runs a prompt's tokens alone: a padded batch's mask
  * leaves the real tokens' outputs as they are.
  */
final class Umt5 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  private def dimensions(name: String) =
    source.shape(name).dimensions.map(_.toInt)

  private val Seq(vocabulary, hidden) = dimensions("shared.weight")
  private val blockCount = Iterator
    .from(0)
    .takeWhile(i => source.has(s"encoder.block.$i.layer.0.layer_norm.weight"))
    .size
  private val Seq(buckets, heads) =
    dimensions(
      "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    )
  private val inner = dimensions(
    "encoder.block.0.layer.0.SelfAttention.q.weight"
  ).head
  private val head = inner / heads
  private val intermediate = dimensions(
    "encoder.block.0.layer.1.DenseReluDense.wi_0.weight"
  ).head
  private val MaxDistance = 128
  private val Epsilon = 1e-6f

  /** The width of each token's output. */
  def width: Int = hidden

  private val weights = new HybridWeights(ops, source, gguf = false)

  final private class Block(i: Int) {
    private val prefix = s"encoder.block.$i.layer"
    val attentionNorm: Tensor =
      weights.floats(
        s"$prefix.0.layer_norm.weight",
        s"$prefix.0.layer_norm.weight",
        Shape.of(hidden)
      )
    val q: Tensor = source(s"$prefix.0.SelfAttention.q.weight")
    val k: Tensor = source(s"$prefix.0.SelfAttention.k.weight")
    val v: Tensor = source(s"$prefix.0.SelfAttention.v.weight")
    val o: Tensor = source(s"$prefix.0.SelfAttention.o.weight")
    val relative: Array[Float] =
      weights.hostFloats(
        s"$prefix.0.SelfAttention.relative_attention_bias.weight"
      )
    val mlpNorm: Tensor =
      weights.floats(
        s"$prefix.1.layer_norm.weight",
        s"$prefix.1.layer_norm.weight",
        Shape.of(hidden)
      )
    val gate: Tensor = source(s"$prefix.1.DenseReluDense.wi_0.weight")
    val up: Tensor = source(s"$prefix.1.DenseReluDense.wi_1.weight")
    val down: Tensor = source(s"$prefix.1.DenseReluDense.wo.weight")
  }

  private val embedding = source("shared.weight")
  private val blocks = (0 until blockCount).map(new Block(_))
  private val finalNorm =
    weights.floats(
      "encoder.final_layer_norm.weight",
      "encoder.final_layer_norm.weight",
      Shape.of(hidden)
    )

  /** T5's bucket of a key `distance` positions after its query (negative
    * before): half the buckets for each direction, exact up to a quarter of
    * them, logarithmic to `MaxDistance`.
    */
  private def bucket(distance: Int): Int = {
    val half = buckets / 2
    val exact = half / 2
    val side = if (distance > 0) half else 0
    val magnitude = math.abs(distance)
    side + (
      if (magnitude < exact) magnitude
      else
        math.min(
          half - 1,
          exact + (math.log(magnitude.toDouble / exact) / math.log(
            MaxDistance.toDouble / exact
          ) * (half - exact)).toInt
        )
    )
  }

  /** The prompt's `ids` through the encoder: `[L, hidden]`, the caller's to
    * release.
    */
  def encode(ids: Array[Int]): Tensor = {
    require(
      ids.forall(id => id >= 0 && id < vocabulary),
      "a token past UMT5's vocabulary"
    )
    val tokens = ids.length.toLong
    val held = mutable.ArrayBuffer.empty[Tensor]
    def allocate(columns: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(tokens, columns))
      held += tensor
      tensor
    }
    val x = ops.allocate(DType.F32, Shape.of(tokens, hidden.toLong))
    try {
      val idTensor = ops.fromInts(Shape.of(tokens), ids)
      held += idTensor
      ops.embedding(embedding, idTensor, x)
      val (normed, q, k, v, attended) =
        (
          allocate(hidden),
          allocate(inner),
          allocate(inner),
          allocate(inner),
          allocate(inner)
        )
      val (projected, gate, up) =
        (allocate(hidden), allocate(intermediate), allocate(intermediate))
      val buckets = Array.tabulate(ids.length * ids.length) { i =>
        bucket(i % ids.length - i / ids.length)
      }
      blocks.foreach { b =>
        ops.rmsNorm(x, b.attentionNorm, Epsilon, 0f, normed)
        ops.linears(normed, Seq(b.q, b.k, b.v), Seq(q, k, v))
        // [heads, query, key]
        val table = Array.tabulate(heads * ids.length * ids.length) { i =>
          val (h, pair) =
            (i / (ids.length * ids.length), i % (ids.length * ids.length))
          b.relative(buckets(pair) * heads + h)
        }
        val bias = ops.fromFloats(Shape.of(heads, tokens, tokens), table)
        try {
          def view(t: Tensor) = t.view(tokens, 1, heads, head)
          ops.shortAttention(
            view(q),
            view(k),
            view(v),
            1f,
            view(attended),
            Some(bias)
          )
        } finally ops.release(bias)
        ops.linear(attended, b.o, projected)
        ops.add(x, projected, x)
        ops.rmsNorm(x, b.mlpNorm, Epsilon, 0f, normed)
        ops.linears(normed, Seq(b.gate, b.up), Seq(gate, up))
        ops.gated(Activation.GeluTanh, gate, up, gate)
        ops.linear(gate, b.down, projected)
        ops.add(x, projected, x)
      }
      ops.rmsNorm(x, finalNorm, Epsilon, 0f, normed)
      ops.copy(normed, x)
      x
    } catch {
      case error: Throwable =>
        ops.release(x)
        throw error
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object Umt5 {

  def open(ops: Ops, path: Path): Umt5 = {
    val source = WeightSource.open(ops, path)
    try {
      if (!source.has("encoder.final_layer_norm.weight"))
        throw new FormatException(
          s"$path is no UMT5 encoder in transformers' names (no encoder.final_layer_norm.weight)"
        )
      new Umt5(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
