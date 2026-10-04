package drift.runner.models

import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import scala.collection.mutable

/** A VAE's weights as its layers take them, from a checkpoint's tensors: 3×3
  * convolutions as `[out, in × 9]` BF16 (a 3-D kernel keeps its last temporal
  * slice), 1×1 ones as linears `[out, in]` BF16, small vectors as F32. Every
  * tensor made is released by `release`.
  */
final class VaeWeights(ops: Ops, source: WeightSource) {

  private val made = mutable.ArrayBuffer.empty[Tensor]

  def keep(tensor: Tensor): Tensor = {
    made += tensor
    tensor
  }

  def has(name: String): Boolean = source.has(name)

  def shape(name: String): Shape = source.shape(name)

  /** A tensor's values on the host, as floats. */
  def values(name: String): Array[Float] = {
    val tensor = source(name)
    val segment = tensor.storage match {
      case Storage.Host(s)          => s
      case Storage.Registered(h, _) => h
      case Storage.Device(_, _)     =>
        throw new IllegalStateException(s"$name is not on the host")
    }
    tensor.dtype.decode(
      segment.asSlice(tensor.byteOffset, tensor.byteSize),
      tensor.shape.elementCount.toInt
    )
  }

  /** Floats as BF16 (round to nearest even), uploaded. */
  def bf16(shape: Shape, floats: Array[Float]): Tensor = {
    val bytes = new Array[Byte](2 * floats.length)
    floats.indices.foreach { i =>
      val bits = java.lang.Float.floatToRawIntBits(floats(i))
      val rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16
      bytes(2 * i) = rounded.toByte
      bytes(2 * i + 1) = (rounded >>> 8).toByte
    }
    keep(ops.fromBytes(DType.BF16, shape, bytes))
  }

  /** Floats as F16 (round to nearest even), uploaded. */
  def f16(shape: Shape, floats: Array[Float]): Tensor = {
    val bytes = new Array[Byte](2 * floats.length)
    floats.indices.foreach { i =>
      val half = java.lang.Float.floatToFloat16(floats(i))
      bytes(2 * i) = half.toByte
      bytes(2 * i + 1) = (half >>> 8).toByte
    }
    keep(ops.fromBytes(DType.F16, shape, bytes))
  }

  /** Floats as they are, uploaded in `shape`. */
  def f32(shape: Shape, floats: Array[Float]): Tensor =
    keep(ops.fromFloats(shape, floats))

  def floats(values: Array[Float]): Tensor =
    keep(ops.fromFloats(Shape.of(values.length), values))

  def floats(name: String): Tensor = floats(values(name))

  /** A 3×3 convolution and its bias. */
  def conv3x3(prefix: String): Convolution = {
    val dimensions = source(s"$prefix.weight").shape.dimensions.map(_.toInt)
    val (out, in) = (dimensions(0), dimensions(1))
    val all = values(s"$prefix.weight")
    val spatial =
      if (dimensions.size == 5) {
        val depth = dimensions(2)
        Array.tabulate(out * in * 9) { i =>
          val (oi, k) = (i / 9, i % 9)
          all((oi * depth + depth - 1) * 9 + k)
        }
      } else all
    Convolution(bf16(Shape.of(out, in * 9L), spatial), floats(s"$prefix.bias"))
  }

  /** A causal 3×3×3 convolution as its three temporal slices (the oldest
    * frame's first), each a 3×3 one; the bias rides on the newest. Output row
    * `r` is the stored row `rows(r)`, input channel `i` the stored `inputs(i)`.
    * `half` keeps the weights F16 (the convolutions then take F16 patches and
    * F32 sums).
    */
  def conv3x3x3(
      prefix: String,
      rows: Int => Int = identity,
      inputs: Int => Int = identity,
      half: Boolean = false
  ): Seq[Convolution] = {
    val dimensions = source(s"$prefix.weight").shape.dimensions.map(_.toInt)
    val (out, in, depth) = (dimensions(0), dimensions(1), dimensions(2))
    val all = values(s"$prefix.weight")
    val storedBias = values(s"$prefix.bias")
    val bias = floats(Array.tabulate(out)(r => storedBias(rows(r))))
    val zero = floats(new Array[Float](out))
    (0 until depth).map { t =>
      val slice = Array.tabulate(out * in * 9) { i =>
        val (oi, k) = (i / 9, i % 9)
        all((rows(oi / in) * in + inputs(oi % in)) * depth * 9 + t * 9 + k)
      }
      Convolution(
        if (half) f16(Shape.of(out, in * 9L), slice)
        else bf16(Shape.of(out, in * 9L), slice),
        if (t == depth - 1) bias else zero
      )
    }
  }

  /** A causal 3×3×3 convolution's three temporal slices summed into one 3×3:
    * what it does to a frame whose two predecessors are the frame itself (a
    * still picture, or a stream's first frame when the first stands for the
    * ones before it).
    */
  def conv3x3x3Summed(prefix: String): Convolution = {
    val dimensions = source(s"$prefix.weight").shape.dimensions.map(_.toInt)
    val (out, in, depth) = (dimensions(0), dimensions(1), dimensions(2))
    val all = values(s"$prefix.weight")
    Convolution(
      bf16(
        Shape.of(out, in * 9L),
        Array.tabulate(out * in * 9) { i =>
          val (oi, k) = (i / 9, i % 9)
          (0 until depth).map(t => all(oi * depth * 9 + t * 9 + k)).sum
        }
      ),
      floats(s"$prefix.bias")
    )
  }

  /** A causal temporal convolution (`[out, in, 3, 1, 1]`) as its three 1×1
    * slices, the oldest frame's first; the bias rides on the newest.
    */
  def conv3x1x1(prefix: String): Seq[Convolution] = {
    val dimensions = source(s"$prefix.weight").shape.dimensions.map(_.toInt)
    val (out, in, depth) = (dimensions(0), dimensions(1), dimensions(2))
    val all = values(s"$prefix.weight")
    val bias = floats(s"$prefix.bias")
    val zero = floats(new Array[Float](out))
    (0 until depth).map { t =>
      Convolution(
        bf16(
          Shape.of(out, in),
          Array.tabulate(out * in)(i => all(i * depth + t))
        ),
        if (t == depth - 1) bias else zero
      )
    }
  }

  /** A 1×1 convolution (a linear) and its bias. */
  def conv1x1(prefix: String): Convolution = {
    val dimensions = source(s"$prefix.weight").shape.dimensions
    Convolution(
      bf16(Shape.of(dimensions(0), dimensions(1)), values(s"$prefix.weight")),
      floats(s"$prefix.bias")
    )
  }

  def release(): Unit = made.foreach(ops.release)
}

/** A convolution's BF16 weight and F32 bias. */
final case class Convolution(weight: Tensor, bias: Tensor) {
  def outChannels: Long = bias.shape.elementCount

  /** The first `count` output channels alone. */
  def take(count: Long): Convolution =
    Convolution(weight.rows(0, count), bias.rows(0, count))
}

/** A norm over each pixel's channels: Wan's RMS (`F.normalize × √C × γ`), or a
  * group norm of 32 groups (the LDM VAEs').
  */
enum ChannelNorm {
  case Rms(gamma: Tensor)
  case Group(weight: Tensor, bias: Tensor)

  /** `out = norm(x)`, both `[pixels, C]`. */
  def apply(ops: Ops, x: Tensor, out: Tensor): Unit = this match {
    // F.normalize's 1e-12 on the norm is ~0 here
    case Rms(gamma)          => ops.rmsNorm(x, gamma, 1e-24f, 0f, out)
    case Group(weight, bias) => ops.groupNorm(x, 32, weight, bias, 1e-6f, out)
  }
}

/** Norm, SiLU, 3×3 convolution, twice; plus the input (through a 1×1 when the
  * channels change).
  */
final case class ResidualBlock(
    norm1: ChannelNorm,
    conv1: Convolution,
    norm2: ChannelNorm,
    conv2: Convolution,
    shortcut: Option[Convolution]
)

/** Single-head attention over all the pixels, 1×1 projections. */
final case class PixelAttention(
    norm: ChannelNorm,
    q: Convolution,
    k: Convolution,
    v: Convolution,
    proj: Convolution
)

/** The Wan VAEs' layers by their original names (the ComfyUI repackages',
  * sd-cpp's): RMS channel norms (`gamma`), residual blocks (`residual.0`, `.2`,
  * `.3`, `.6`, and `shortcut` when the channels change), and the middle's
  * attention (`to_qkv` split into its thirds, `proj`).
  */
final class WanLayers(weights: VaeWeights) {

  def norm(prefix: String): ChannelNorm =
    ChannelNorm.Rms(weights.floats(s"$prefix.gamma"))

  def residual(prefix: String): ResidualBlock =
    ResidualBlock(
      norm(s"$prefix.residual.0"),
      weights.conv3x3(s"$prefix.residual.2"),
      norm(s"$prefix.residual.3"),
      weights.conv3x3(s"$prefix.residual.6"),
      Option.when(weights.has(s"$prefix.shortcut.weight"))(
        weights.conv1x1(s"$prefix.shortcut")
      )
    )

  def attention(prefix: String): PixelAttention = {
    val qkv = weights.conv1x1(s"$prefix.to_qkv")
    val channels = qkv.outChannels / 3
    def part(i: Int) = Convolution(
      qkv.weight.rows(i * channels, channels),
      qkv.bias.rows(i * channels, channels)
    )
    PixelAttention(
      norm(s"$prefix.norm"),
      part(0),
      part(1),
      part(2),
      weights.conv1x1(s"$prefix.proj")
    )
  }

  /** The middle: a residual block, the attention, a residual block. */
  def middle(prefix: String): (ResidualBlock, PixelAttention, ResidualBlock) =
    (
      residual(s"$prefix.middle.0"),
      attention(s"$prefix.middle.1"),
      residual(s"$prefix.middle.2")
    )
}

/** One image through a VAE's layers, channels-last `[H, W, C]`. Each step takes
  * its input, releases it and returns its output; `image` tensors not yet
  * handed back by `result` are released by `release`.
  */
final class VaeRun(ops: Ops) {

  private val live = mutable.ArrayBuffer.empty[Tensor]

  def image(height: Long, width: Long, channels: Long): Tensor = {
    val tensor = ops.allocate(DType.F32, Shape.of(height, width, channels))
    live += tensor
    tensor
  }

  def free(tensor: Tensor): Unit = {
    live -= tensor
    ops.release(tensor)
  }

  /** `tensor`, no longer released by `release`. */
  def result(tensor: Tensor): Tensor = {
    live -= tensor
    tensor
  }

  def pixels(x: Tensor): Tensor = {
    val Seq(height, width, c) = x.shape.dimensions
    x.view(height * width, c)
  }

  def dimensions(x: Tensor): (Long, Long, Long) = {
    val Seq(height, width, c) = x.shape.dimensions
    (height, width, c)
  }

  def conv3x3(x: Tensor, conv: Convolution, stride: Int = 1): Tensor = {
    val (height, width, _) = dimensions(x)
    val out = image(height / stride, width / stride, conv.outChannels)
    ops.conv3x3(x, conv.weight, conv.bias, out, stride)
    free(x)
    out
  }

  def conv1x1(x: Tensor, conv: Convolution): Tensor = {
    val (height, width, _) = dimensions(x)
    val out = image(height, width, conv.outChannels)
    ops.linear(pixels(x), conv.weight, pixels(out))
    ops.addRow(pixels(out), conv.bias, pixels(out))
    free(x)
    out
  }

  /** `norm(x)` then SiLU, into a new image; `x` is kept. */
  def normSilu(x: Tensor, norm: ChannelNorm): Tensor = {
    val (height, width, c) = dimensions(x)
    val normed = image(height, width, c)
    norm(ops, pixels(x), pixels(normed))
    ops.activation(Activation.Silu, normed, normed)
    normed
  }

  def residual(x: Tensor, block: ResidualBlock): Tensor = {
    val first = conv3x3(normSilu(x, block.norm1), block.conv1)
    val second = conv3x3(normSilu(first, block.norm2), block.conv2)
    free(first)
    block.shortcut match {
      case Some(conv) =>
        val (height, width, _) = dimensions(x)
        val shortcut = image(height, width, conv.outChannels)
        ops.linear(pixels(x), conv.weight, pixels(shortcut))
        ops.addRow(pixels(shortcut), conv.bias, pixels(shortcut))
        ops.add(second, shortcut, second)
        free(shortcut)
      case None => ops.add(second, x, second)
    }
    free(x)
    second
  }

  /** Nearest ×2 then a 3×3 convolution. */
  def upsample(x: Tensor, conv: Convolution): Tensor = {
    val (height, width, c) = dimensions(x)
    val up = image(2 * height, 2 * width, c)
    ops.upsample2x(x, up)
    free(x)
    conv3x3(up, conv)
  }

  /** A middle: residual block, attention, residual block. */
  def middle(
      x: Tensor,
      layers: (ResidualBlock, PixelAttention, ResidualBlock)
  ): Tensor =
    residual(attend(residual(x, layers._1), layers._2), layers._3)

  /** The attention, in place on `x`, in chunks of queries. */
  def attend(x: Tensor, block: PixelAttention): Tensor = {
    val (height, width, c) = dimensions(x)
    val count = height * width
    val flat = x.view(count, c)
    val temporaries = mutable.ArrayBuffer.empty[Tensor]
    def allocate(dtype: DType, rows: Long, cols: Long) = {
      val tensor = ops.allocate(dtype, Shape.of(rows, cols))
      temporaries += tensor
      tensor
    }
    try {
      val normed = allocate(DType.F32, count, c)
      block.norm(ops, flat, normed)
      val (q, k, v) =
        (
          allocate(DType.F32, count, c),
          allocate(DType.F32, count, c),
          allocate(DType.F32, count, c)
        )
      ops.linears(
        normed,
        Seq(block.q.weight, block.k.weight, block.v.weight),
        Seq(q, k, v)
      )
      Seq((q, block.q.bias), (k, block.k.bias), (v, block.v.bias))
        .foreach((t, bias) => ops.addRow(t, bias, t))
      // the keys, and the values transposed, as the BF16 "weights" of two GEMMs
      val keys = allocate(DType.BF16, count, c)
      ops.convert(k, keys)
      val transposed = allocate(DType.F32, c, count)
      ops.transpose(v, transposed)
      val values = allocate(DType.BF16, c, count)
      ops.convert(transposed, values)
      val chunk = math.max(1L, math.min(count, (128L << 20) / (4 * count)))
      val scores = allocate(DType.F32, chunk, count)
      (0L until count by chunk).foreach { first =>
        val n = math.min(chunk, count - first)
        val s = scores.rows(0, n)
        ops.linear(q.rows(first, n), keys, s)
        ops.softmax(s, (1 / math.sqrt(c.toDouble)).toFloat, s)
        ops.linear(s, values, normed.rows(first, n))
      }
      ops.linear(normed, block.proj.weight, q)
      ops.addRow(q, block.proj.bias, q)
      ops.add(flat, q, flat)
      x
    } finally temporaries.foreach(ops.release)
  }

  def release(): Unit = {
    live.foreach(ops.release)
    live.clear()
  }
}
