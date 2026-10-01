package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** MiniMax H3's audio encoder (diffusers' `AutoencoderKLMiniMaxH3Audio.encode`,
  * ComfyUI's `MiniMaxH3AudioVAE.encode`), from the audio VAE's `encoder`,
  * `pre_block` and `mean_proj`: mono 32 kHz samples, right-padded with zeros to
  * whole latents (800 samples), into normalized latents `[L, 32]` (the
  * posterior's mean; `logs_proj` is never read).
  *   - DAC's encoder: a convolution of 7 taps, then blocks of three residual
  *     units (Snake, a dilated convolution of 7 taps at dilations 1, 3 and 9,
  *     Snake, a 1-tap one; added back) and a Snake before a strided convolution
  *     of `2 × stride` taps doubling the channels (the strides from the taps),
  *     then Snake and a convolution of 3 taps; weight norms folded in the file.
  *     F32, as the vocoders.
  *   - `pre_block`: `proj(norm3(x)) + attention(norm1(x))`, then a GeGLU MLP
  *     residual. The attention is causal over 8 heads of the full width, keys
  *     without bias; its heads averaged and each 8 values of the result
  *     averaged down to 32, then `proj`: one linear of the attended rows, built
  *     at load.
  * A stereo track is two encodes, as for the decoder.
  */
final class MiniMaxH3AudioEncoder private (
    ops: Ops,
    source: WeightSource,
    heads: Int
) extends AutoCloseable {

  private val weights = new VaeWeights(ops, source)
  private val (mean, deviation) =
    (weights.values("latents_mean"), weights.values("latents_std"))

  /** The latent channels a frame holds. */
  val channels: Int = mean.length

  private def alpha(name: String): Tensor = weights.floats(name)

  final private case class ResidualUnit(
      first: Tensor,
      dilated: TimeConvolution,
      second: Tensor,
      pointwise: TimeConvolution
  )

  final private case class Block(
      units: Seq[ResidualUnit],
      snake: Tensor,
      stride: Int,
      down: Tensor,
      bias: Tensor,
      taps: Int
  )

  if (!weights.has("encoder.block.0.weight"))
    throw new FormatException("no MiniMax H3 audio encoder (encoder.block.0)")
  private val input = TimeConvolution.load(weights, "encoder.block.0")
  private val blocks = Iterator
    .from(1)
    .takeWhile(i => weights.has(s"encoder.block.$i.block.0.block.0.alpha"))
    .map { i =>
      val prefix = s"encoder.block.$i.block"
      val units = Seq(1, 3, 9).zipWithIndex.map { (dilation, u) =>
        ResidualUnit(
          alpha(s"$prefix.$u.block.0.alpha"),
          TimeConvolution.load(weights, s"$prefix.$u.block.1", dilation),
          alpha(s"$prefix.$u.block.2.alpha"),
          TimeConvolution.load(weights, s"$prefix.$u.block.3")
        )
      }
      val Seq(out, in, taps) = weights.shape(s"$prefix.4.weight").dimensions
      Block(
        units,
        alpha(s"$prefix.3.alpha"),
        (taps / 2).toInt,
        weights
          .f32(Shape.of(out, in * taps), weights.values(s"$prefix.4.weight")),
        weights.floats(s"$prefix.4.bias"),
        taps.toInt
      )
    }
    .toSeq
  private val outputSnake = alpha(s"encoder.block.${blocks.size + 1}.alpha")
  private val output =
    TimeConvolution.load(weights, s"encoder.block.${blocks.size + 2}")

  /** Samples a latent spans: the strides' product. */
  val hop: Int = blocks.map(_.stride).product

  private val width =
    weights.shape("pre_block.norm1.weight").dimensions.head.toInt
  private val headWidth = width / heads

  final private class Norm(prefix: String) {
    val weight: Tensor = weights.floats(s"$prefix.weight")
    val bias: Tensor = weights.floats(s"$prefix.bias")
  }
  private val norm1 = new Norm("pre_block.norm1")
  private val norm2 = new Norm("pre_block.norm2")
  private val norm3 = new Norm("pre_block.norm3")
  private val mlpNorm = new Norm("pre_block.mlp.norm")
  private def linear(name: String): (Tensor, Tensor) = {
    val Seq(out, in) = weights.shape(s"$name.weight").dimensions
    (
      weights.f32(Shape.of(out, in), weights.values(s"$name.weight")),
      weights.floats(s"$name.bias")
    )
  }
  private val qkv = {
    val all = weights.values("pre_block.attn.qkv.weight")
    Seq(0, 1, 2).map(part =>
      weights.f32(
        Shape.of(width, width),
        java.util.Arrays
          .copyOfRange(all, part * width * width, (part + 1) * width * width)
      )
    )
  }
  private val queryBias = weights.floats("pre_block.attn.q_bias")
  private val valueBias = weights.floats("pre_block.attn.v_bias")
  private val projection = linear("pre_block.proj")
  private val (w0, w1, w2) = (
    linear("pre_block.mlp.w0"),
    linear("pre_block.mlp.w1"),
    linear("pre_block.mlp.w2")
  )

  /** The attention's head mean, the average pool to `channels` and `proj`, as
    * one `[channels, width]` linear and its bias.
    */
  private val (pooledProjection, pooledBias) = {
    val proj =
      weights.values("pre_block.attn.proj.weight") // [channels, channels]
    val group = headWidth / channels
    val matrix = Array.tabulate(channels * width) { i =>
      val (o, column) = (i / width, i % width)
      val j = column % headWidth
      proj(o * channels + j / group) / (heads * group)
    }
    (
      weights.f32(Shape.of(channels, width), matrix),
      weights.floats("pre_block.attn.proj.bias")
    )
  }

  /** `(T, C)` of a `[T, C]` tensor. */
  private def sizes(x: Tensor): (Long, Long) = {
    val Seq(t, c) = x.shape.dimensions
    (t, c)
  }

  private def snake(x: Tensor, alpha: Tensor): Tensor = {
    val out = ops.allocate(DType.F32, x.shape)
    ops.snake(x, alpha, out)
    out
  }

  private def affine(x: Tensor, layer: (Tensor, Tensor)): Tensor = {
    val out = ops.allocate(
      DType.F32,
      Shape.of(sizes(x)._1, layer._1.shape.dimensions.head)
    )
    ops.linear(x, layer._1, out)
    ops.addRow(out, layer._2, out)
    out
  }

  private def layerNorm(x: Tensor, norm: Norm): Tensor = {
    val out = ops.allocate(DType.F32, x.shape)
    ops.layerNorm(x, Some(norm.weight), Some(norm.bias), 1e-5f, out)
    out
  }

  /** `x` (`[T, width]`) through `pre_block`: `[T, channels]`. */
  private def preBlock(x: Tensor): Tensor = {
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val length = sizes(x)._1
      val normed = keep(layerNorm(x, norm1))
      val Seq(q, k, v) = qkv.map { weight =>
        val out = keep(ops.allocate(DType.F32, Shape.of(length, width)))
        ops.linear(normed, weight, out)
        out
      }
      ops.addRow(q, queryBias, q)
      ops.addRow(v, valueBias, v)
      val causal = keep(
        ops.fromFloats(
          Shape.of(heads, length, length),
          Array.tabulate((heads * length * length).toInt) { i =>
            val pair = i % (length * length)
            if (pair % length > pair / length) -1e30f else 0f
          }
        )
      )
      val attended = keep(ops.allocate(DType.F32, Shape.of(length, width)))
      def headed(t: Tensor) = t.view(length, 1, heads, headWidth)
      ops.shortAttention(
        headed(q),
        headed(k),
        headed(v),
        (1 / math.sqrt(headWidth)).toFloat,
        headed(attended),
        Some(causal)
      )
      val h = affine(attended, (pooledProjection, pooledBias))
      val projected = keep(affine(keep(layerNorm(x, norm3)), projection))
      ops.add(h, projected, h)
      val m = keep(layerNorm(h, norm2))
      val inner = keep(layerNorm(m, mlpNorm))
      val (gate, value) = (keep(affine(inner, w0)), keep(affine(inner, w1)))
      ops.gated(Activation.GeluTanh, gate, value, gate)
      val mlp = keep(affine(gate, w2))
      ops.add(h, mlp, h)
      h
    } finally held.foreach(ops.release)
  }

  /** One channel's normalized latents `[⌈samples / hop⌉, channels]` (row-major)
    * from its samples.
    */
  def encode(samples: Array[Float]): Array[Float] = {
    val padded =
      java.util.Arrays.copyOf(samples, (samples.length + hop - 1) / hop * hop)
    var x = ops.fromFloats(Shape.of(padded.length.toLong, 1), padded)
    def step(next: Tensor => Tensor): Unit = {
      val made = next(x)
      ops.release(x)
      x = made
    }
    try {
      step(input(ops, _))
      blocks.foreach { block =>
        block.units.foreach { unit =>
          val first = snake(x, unit.first)
          val dilated =
            try unit.dilated(ops, first)
            finally ops.release(first)
          val second =
            try snake(dilated, unit.second)
            finally ops.release(dilated)
          val residual =
            try unit.pointwise(ops, second)
            finally ops.release(second)
          ops.add(x, residual, x)
          ops.release(residual)
        }
        step(snake(_, block.snake))
        step { y =>
          val (length, _) = sizes(y)
          val pad = (block.stride + 1) / 2
          val out = ops.allocate(
            DType.F32,
            Shape.of(
              (length + 2 * pad - block.taps) / block.stride + 1,
              block.down.shape.dimensions.head
            )
          )
          ops.conv1d(
            y,
            block.down,
            Some(block.bias),
            block.taps,
            1,
            block.stride,
            pad,
            pad,
            out
          )
          out
        }
      }
      step(snake(_, outputSnake))
      step(output(ops, _))
      step(preBlock)
      val latents = ops.toFloats(x)
      // mean_proj, then normalized, on the host (32 × 32 a frame)
      val meanWeight = weights.values("mean_proj.weight")
      val meanBias = weights.values("mean_proj.bias")
      Array.tabulate(latents.length) { i =>
        val (row, o) = (i / channels, i % channels)
        var sum = meanBias(o).toDouble
        (0 until channels).foreach(j =>
          sum += meanWeight(o * channels + j) * latents(row * channels + j)
        )
        ((sum - mean(o)) / deviation(o)).toFloat
      }
    } finally ops.release(x)
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object MiniMaxH3AudioEncoder {

  /** The encoder of the audio VAE `path`, its attention of `heads` (the file
    * does not record them; the released 8).
    */
  def open(ops: Ops, path: Path, heads: Int = 8): MiniMaxH3AudioEncoder = {
    val source = WeightSource.open(ops, path)
    try new MiniMaxH3AudioEncoder(ops, source, heads)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
