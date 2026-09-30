package drift.runner.models

import drift.runner.ops.Ops
import drift.runner.tensor.*

/** A 1-D convolution over time, "same" padded: its F32 weight `[out, in ×
  * taps]` and F32 patches (a vocoder moves its waveform by 10% for a 0.2%
  * change of its input, so it adds no rounding of its own; F32 takes the same
  * 2 s for LTX's 3.5 s of sound), its bias when it has one.
  */
final case class TimeConvolution(
    weight: Tensor,
    bias: Option[Tensor],
    taps: Int,
    dilation: Int
) {
  def outChannels: Long = weight.shape.dimensions.head

  /** `x` `[T, in]` into a new `[T, out]`. */
  def apply(ops: Ops, x: Tensor): Tensor = {
    val out =
      ops.allocate(DType.F32, Shape.of(x.shape.dimensions.head, outChannels))
    val padding = dilation * (taps - 1)
    ops.conv1d(
      x,
      weight,
      bias,
      taps,
      dilation,
      1,
      padding / 2,
      padding - padding / 2,
      out
    )
    out
  }
}

object TimeConvolution {

  /** The `Conv1d` under `prefix` (`[out, in, taps]`, weight norm folded). */
  def load(
      weights: VaeWeights,
      prefix: String,
      dilation: Int = 1
  ): TimeConvolution = {
    val Seq(out, in, taps) = weights.shape(s"$prefix.weight").dimensions
    TimeConvolution(
      weights.f32(Shape.of(out, in * taps), weights.values(s"$prefix.weight")),
      Option.when(weights.has(s"$prefix.bias"))(
        weights.floats(s"$prefix.bias")
      ),
      taps.toInt,
      dilation
    )
  }
}

/** A transposed 1-D convolution (`ConvTranspose1d`, padding `(taps − stride) /
  * 2`): the input times the weight regrouped as `[out × taps, in]`, then
  * overlapped and added.
  */
final case class TimeUpsampler(
    weight: Tensor,
    bias: Tensor,
    taps: Int,
    stride: Int
) {
  def outChannels: Long = bias.shape.elementCount

  /** `x` `[T, in]` into a new `[T × stride, out]`. */
  def apply(ops: Ops, x: Tensor): Tensor = {
    val length = x.shape.dimensions.head
    val columns = ops.allocate(DType.F32, Shape.of(length, outChannels * taps))
    try {
      ops.conv1d(x, weight, None, 1, 1, 1, 0, 0, columns)
      val pad = (taps - stride) / 2
      val out = ops.allocate(
        DType.F32,
        Shape.of((length - 1) * stride - 2 * pad + taps, outChannels)
      )
      ops.overlapAdd(columns, taps, stride, pad, bias, out)
      out
    } finally ops.release(columns)
  }
}

/** BigVGAN's anti-aliased SnakeBeta: `exp(α)` and `1 / (exp(β) + 10⁻⁹)` per
  * channel, and the stored Kaiser-sinc filters.
  */
final case class AntiAliasedSnake(
    frequency: Tensor,
    inverseMagnitude: Tensor,
    upFilter: Tensor,
    downFilter: Tensor
) {

  /** `x` `[T, C]` into a new `[T, C]`. */
  def apply(ops: Ops, x: Tensor): Tensor = {
    val out = ops.allocate(DType.F32, x.shape)
    ops.antiAliasedSnake(
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    out
  }
}

object AntiAliasedSnake {
  def load(weights: VaeWeights, prefix: String): AntiAliasedSnake =
    AntiAliasedSnake(
      weights.floats(
        weights.values(s"$prefix.act.alpha").map(a => math.exp(a).toFloat)
      ),
      weights.floats(
        weights
          .values(s"$prefix.act.beta")
          .map(b => (1 / (math.exp(b) + 1e-9)).toFloat)
      ),
      weights.floats(s"$prefix.upsample.filter"),
      weights.floats(s"$prefix.downsample.lowpass.filter")
    )
}

/** A BigVGAN generator (anti-aliased SnakeBeta activations, AMP residual
  * blocks), from mel or latent frames `[T, in]` to a waveform `[T × hop,
  * out]`: a 7-tap convolution, then per stage a transposed convolution
  * upsampling by its stride and the mean of its parallel AMP blocks (each,
  * per dilation, `x += conv(snake(conv_d(snake(x))))`), then a snake and a
  * 7-tap convolution. The weights under `prefix`, in either naming of the
  * original checkpoints: MiniMax H3's (`ups.i.0`, `resblocks.j.activations.2d`
  * and `2d + 1`, `activation_post`) or LTX 2's (`ups.i`, `resblocks.j.acts1.d`
  * and `acts2.d`, `act_post`). The files keep no config: `strides` are the
  * upsampling rates, the dilations BigVGAN's (1, 3, 5) in every block.
  */
final class BigVgan(ops: Ops, weights: VaeWeights, prefix: String, strides: Seq[Int]) {

  private val Dilations = Seq(1, 3, 5)

  private final case class AmpBlock(
      first: Seq[(AntiAliasedSnake, TimeConvolution)],
      second: Seq[(AntiAliasedSnake, TimeConvolution)]
  )

  private final case class Stage(upsampler: TimeUpsampler, blocks: Seq[AmpBlock])

  private val oldNames = weights.has(s"$prefix.ups.0.0.weight")
  require(
    weights.has(s"$prefix.ups.${strides.size - 1}.${if (oldNames) "0." else ""}weight") &&
      !weights.has(s"$prefix.ups.${strides.size}.${if (oldNames) "0." else ""}weight"),
    s"$prefix: ${strides.size} upsampling rates for another count of stages"
  )
  private val blocksPerStage =
    Iterator.from(0).takeWhile(j => weights.has(s"$prefix.resblocks.$j.convs1.0.weight")).size /
      strides.size

  private val convPre = TimeConvolution.load(weights, s"$prefix.conv_pre")

  private val stages = strides.zipWithIndex.map { (stride, i) =>
    val name = s"$prefix.ups.$i${if (oldNames) ".0" else ""}"
    val Seq(in, out, taps) = weights.shape(s"$name.weight").dimensions.map(_.toInt)
    val stored = weights.values(s"$name.weight")
    // [in][out][taps] → rows (o, k) of the inputs
    val regrouped = Array.tabulate(out * taps * in) { index =>
      val (row, c) = (index / in, index % in)
      stored((c * out + row / taps) * taps + row % taps)
    }
    val upsampler = TimeUpsampler(
      weights.f32(Shape.of(out.toLong * taps, in), regrouped),
      weights.floats(s"$name.bias"),
      taps,
      stride
    )
    val blocks = (0 until blocksPerStage).map { b =>
      val block = s"$prefix.resblocks.${i * blocksPerStage + b}"
      def snake(first: Boolean, d: Int) =
        AntiAliasedSnake.load(
          weights,
          if (weights.has(s"$block.acts1.0.act.alpha"))
            s"$block.${if (first) "acts1" else "acts2"}.$d"
          else s"$block.activations.${2 * d + (if (first) 0 else 1)}"
        )
      AmpBlock(
        Dilations.indices.map(d =>
          snake(first = true, d) -> TimeConvolution
            .load(weights, s"$block.convs1.$d", Dilations(d))
        ),
        Dilations.indices.map(d =>
          snake(first = false, d) -> TimeConvolution
            .load(weights, s"$block.convs2.$d")
        )
      )
    }
    Stage(upsampler, blocks)
  }

  private val snakePost = AntiAliasedSnake.load(
    weights,
    if (weights.has(s"$prefix.act_post.act.alpha")) s"$prefix.act_post"
    else s"$prefix.activation_post"
  )
  private val convPost = TimeConvolution.load(weights, s"$prefix.conv_post")

  /** The samples each input frame becomes. */
  val hop: Int = strides.product

  /** The waveform's channels. */
  val outChannels: Int = convPost.outChannels.toInt

  /** `snake` then `conv` on `x`, into a new tensor. */
  private def activated(
      x: Tensor,
      snake: AntiAliasedSnake,
      conv: TimeConvolution
  ): Tensor = {
    val active = snake(ops, x)
    try conv(ops, active)
    finally ops.release(active)
  }

  /** `x` `[T, in]` (kept) into a new waveform `[T × hop, out]`, unclamped. */
  def apply(x: Tensor): Tensor = {
    var h = convPre(ops, x)
    stages.foreach { stage =>
      val up = stage.upsampler(ops, h)
      ops.release(h)
      val sum = ops.allocate(DType.F32, up.shape)
      ops.zero(sum)
      stage.blocks.foreach { block =>
        val y = ops.allocate(DType.F32, up.shape)
        ops.copy(up, y)
        block.first.zip(block.second).foreach { case ((s1, c1), (s2, c2)) =>
          val inner = activated(y, s1, c1)
          val residual = activated(inner, s2, c2)
          ops.release(inner)
          ops.add(y, residual, y)
          ops.release(residual)
        }
        ops.add(sum, y, sum)
        ops.release(y)
      }
      ops.release(up)
      ops.scale(sum, 1f / stage.blocks.size, sum)
      h = sum
    }
    try activated(h, snakePost, convPost)
    finally ops.release(h)
  }
}
