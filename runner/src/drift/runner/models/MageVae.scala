package drift.runner.models

import drift.runner.diffusion.Images
import drift.runner.formats.FormatException
import drift.runner.ops.{Activation, Ops}
import drift.runner.tensor.*

import java.nio.file.Path
import java.util.SplittableRandom
import scala.collection.mutable

/** Mage-Flow's VAE (microsoft/Mage's `MageVAE`, sd-cpp's `mage_vae.hpp`): 128
  * channels at 1/16, neither packed nor normalized, both ways from the released
  * names. Channels-last throughout; the sizes are read off the weights, the 1×1
  * convolutions run as BF16 linears, the activations F32.
  *   - A DiCo block, on rows of pixels: layer norm, a modulation, a 1×1, a
  *     depthwise 3×3, GELU, a channel attention (each channel times the sigmoid
  *     of a 1×1 of the image's mean), a 1×1, a gated residual; then layer norm,
  *     a modulation, a 1×1 to four times the width, GELU, a 1×1 back, a gated
  *     residual. The modulations come from a timestep that is always 0: they
  *     are constants, computed once here.
  *   - The encoder (`student.dconv_encoder`): the image in patches of 16
  *     through a linear, two DiCo blocks of 768 (affine norms, no modulation),
  *     a 1×1 to 384, fused with the projection of a zero latent (a constant),
  *     21 DiCo blocks, a layer norm and a 1×1 to the latent's mean and log
  *     variance.
  *   - The decoder (`pipeline`): the latent through `y_embedder.decoder` (a
  *     3×3, three residual blocks with two attentions between them, each over
  *     windows of 32 × 32 latents on their own, a 3×3) to a condition of 384 at
  *     1/16; that through 21 DiCo blocks; then every pixel on its own: 32
  *     features from the condition (`y_embedder_x`) plus its place in its patch
  *     (a DCT table through `x_embedder`), three small MLP blocks modulated per
  *     pixel from the blocks' output (`dec_net`), an RMS norm and a linear to
  *     RGB. The noisy image the denoiser was trained to take is zeros, so its
  *     patch embedding (`s_embedder.proj1`, no bias) drops out.
  * `pipeline.y_embedder.encoder` (a FLUX.2 encoder kept in the file) is not
  * read.
  */
final class MageVae private (ops: Ops, source: WeightSource, window: Int)
    extends AutoCloseable {

  private val Encoder = "student.dconv_encoder"
  private val Decoder = "pipeline"
  private val Epsilon = 1e-6f

  if (!source.has(s"$Encoder.patch_cond_embed.weight"))
    throw new FormatException(
      s"no Mage-Flow VAE ($Encoder.patch_cond_embed.weight)"
    )

  private val weights = new VaeWeights(ops, source)

  private def dimensions(name: String) =
    source.shape(name).dimensions.map(_.toInt)

  /** Image pixels per latent along each side. */
  val scale: Int = dimensions(s"$Encoder.patch_cond_embed.weight").last

  /** Latent channels. */
  val channels: Int =
    dimensions(s"$Decoder.y_embedder.decoder.conv_in.weight")(1)
  private val hidden =
    dimensions(s"$Decoder.y_embedder.decoder.conv_in.weight").head

  /** The per-pixel decoder's width. */
  private val pixelWidth = dimensions(s"$Decoder.final_layer.norm.weight").head
  private val area = scale * scale

  // ---- host arithmetic on small weights, done once ----------------------------------

  /** `weight · x + bias`, `weight` `[bias.length, x.length]` row-major. */
  private def affine(
      weight: Array[Float],
      bias: Array[Float],
      x: Array[Float]
  ): Array[Float] =
    Array.tabulate(bias.length) { o =>
      var sum = bias(o).toDouble
      var i = 0
      while (i < x.length) {
        sum += weight(o * x.length + i).toDouble * x(i)
        i += 1
      }
      sum.toFloat
    }

  private def silu(x: Array[Float]) =
    x.map(v => (v / (1 + math.exp(-v.toDouble))).toFloat)

  /** The timestep embedder's output at `t = 0`: a sinusoid of ones (the
    * cosines) then zeros through its MLP.
    */
  private def timeAtZero(prefix: String): Array[Float] = {
    val frequencies = dimensions(s"$prefix.mlp.0.weight").last
    val sinusoid =
      Array.tabulate(frequencies)(i => if (i < frequencies / 2) 1f else 0f)
    val inner = affine(
      weights.values(s"$prefix.mlp.0.weight"),
      weights.values(s"$prefix.mlp.0.bias"),
      sinusoid
    )
    affine(
      weights.values(s"$prefix.mlp.2.weight"),
      weights.values(s"$prefix.mlp.2.bias"),
      silu(inner)
    )
  }

  /** The columns `[first, first + count)` of a `[rows, columns]` weight. */
  private def columns(
      values: Array[Float],
      rows: Int,
      first: Int,
      count: Int
  ): Array[Float] = {
    val all = values.length / rows
    Array.tabulate(rows * count)(i =>
      values(i / count * all + first + i % count)
    )
  }

  // ---- layers ---------------------------------------------------------------------

  /** A DiCo block: affine norms (the encoder's head) or a modulation of six
    * rows (shift, scale and gate, twice).
    */
  final private case class DiCo(
      norm1: Option[(Tensor, Tensor)],
      norm2: Option[(Tensor, Tensor)],
      modulation: Option[Tensor],
      conv1: Convolution,
      depthwise: Convolution,
      attention: Convolution,
      conv3: Convolution,
      conv4: Convolution,
      conv5: Convolution
  )

  private def dico(prefix: String, time: Option[Array[Float]]): DiCo = {
    val width = dimensions(s"$prefix.conv1.weight").head
    def norm(name: String) =
      Option.when(weights.has(s"$prefix.$name.weight"))(
        (
          weights.floats(s"$prefix.$name.weight"),
          weights.floats(s"$prefix.$name.bias")
        )
      )
    DiCo(
      norm("norm1"),
      norm("norm2"),
      time.map(t =>
        weights.f32(
          Shape.of(6, width),
          affine(
            weights.values(s"$prefix.adaLN_modulation.1.weight"),
            weights.values(s"$prefix.adaLN_modulation.1.bias"),
            silu(t)
          )
        )
      ),
      weights.conv1x1(s"$prefix.conv1"),
      Convolution(
        weights
          .f32(Shape.of(width, 9), weights.values(s"$prefix.conv2.weight")),
        weights.floats(s"$prefix.conv2.bias")
      ),
      weights.conv1x1(s"$prefix.ca.1"),
      weights.conv1x1(s"$prefix.conv3"),
      weights.conv1x1(s"$prefix.conv4"),
      weights.conv1x1(s"$prefix.conv5")
    )
  }

  private def dicos(prefix: String, time: Option[Array[Float]]): Seq[DiCo] =
    Iterator
      .from(0)
      .map(i => s"$prefix.$i")
      .takeWhile(p => weights.has(s"$p.conv1.weight"))
      .map(dico(_, time))
      .toSeq

  private def groupNorm(prefix: String) =
    ChannelNorm.Group(
      weights.floats(s"$prefix.weight"),
      weights.floats(s"$prefix.bias")
    )

  /** Vectors of a width, made once: zeros, and minus ones. */
  private val constants = mutable.Map.empty[(Long, Float), Tensor]
  private def constant(width: Long, value: Float): Tensor =
    constants.getOrElseUpdate(
      (width, value),
      weights.floats(Array.fill(width.toInt)(value))
    )

  // the encoder
  private val patchEmbed = Convolution(
    weights.bf16(
      Shape.of(
        dimensions(s"$Encoder.patch_cond_embed.weight").head,
        3L * area
      ),
      weights.values(s"$Encoder.patch_cond_embed.weight")
    ),
    weights.floats(s"$Encoder.patch_cond_embed.bias")
  )
  private val headBlocks = dicos(s"$Encoder.head_blocks", None)
  private val projectDown = weights.conv1x1(s"$Encoder.proj_down")
  // fuse_proj of [the image's features ; z_proj of a zero latent]: its first
  // columns, the others times z_proj's bias joining the bias
  private val fuse = {
    val weight = weights.values(s"$Encoder.fuse_proj.weight")
    val zero = weights.values(s"$Encoder.z_proj.bias")
    Convolution(
      weights
        .bf16(Shape.of(hidden, hidden), columns(weight, hidden, 0, hidden)),
      weights.floats(
        affine(
          columns(weight, hidden, hidden, hidden),
          weights.values(s"$Encoder.fuse_proj.bias"),
          zero
        )
      )
    )
  }
  private val encoderBlocks =
    dicos(s"$Encoder.blocks", Some(timeAtZero(s"$Encoder.t_embedder")))
  private val encoderNorm = (
    weights.floats(s"$Encoder.norm_out.weight"),
    weights.floats(s"$Encoder.norm_out.bias")
  )
  private val encoderOut = weights.conv1x1(s"$Encoder.proj_out")
  require(
    encoderOut.outChannels == 2L * channels,
    s"the encoder gives ${encoderOut.outChannels} channels, not a mean and a log variance of $channels"
  )

  // the decoder: the latent to its condition
  private val conditioner = s"$Decoder.y_embedder.decoder"
  private val conditionIn = weights.conv3x3(s"$conditioner.conv_in")
  private val conditionBlocks: Seq[Either[ResidualBlock, PixelAttention]] =
    Iterator
      .from(0)
      .map(i => s"$conditioner.block.$i")
      .takeWhile(p =>
        weights.has(s"$p.conv1.weight") || weights.has(s"$p.q.weight")
      )
      .map(p =>
        if (weights.has(s"$p.q.weight"))
          Right(
            PixelAttention(
              groupNorm(s"$p.norm"),
              weights.conv1x1(s"$p.q"),
              weights.conv1x1(s"$p.k"),
              weights.conv1x1(s"$p.v"),
              weights.conv1x1(s"$p.proj_out")
            )
          )
        else
          Left(
            ResidualBlock(
              groupNorm(s"$p.norm1"),
              weights.conv3x3(s"$p.conv1"),
              groupNorm(s"$p.norm2"),
              weights.conv3x3(s"$p.conv2"),
              None
            )
          )
      )
      .toSeq
  private val conditionNorm = groupNorm(s"$conditioner.norm_out")
  private val conditionOut = weights.conv3x3(s"$conditioner.conv_out")

  // the decoder's stream: s_embedder.proj2 of [zeros ; the condition]
  private val streamIn = {
    val weight = weights.values(s"$Decoder.s_embedder.proj2.weight")
    val all = weight.length / hidden
    Convolution(
      weights.bf16(
        Shape.of(hidden, hidden),
        columns(weight, hidden, all - hidden, hidden)
      ),
      weights.floats(s"$Decoder.s_embedder.proj2.bias")
    )
  }
  private val decoderBlocks =
    dicos(s"$Decoder.blocks", Some(timeAtZero(s"$Decoder.t_embedder")))

  // the per-pixel decoder. A patch's pixel `p` and feature `f`: the released
  // `y_embedder_x` gives `f × area + p`, taken here as `p × width + f`.
  private val pixelFeatures = {
    val weight = weights.values(s"$Decoder.y_embedder_x.weight")
    val bias = weights.values(s"$Decoder.y_embedder_x.bias")
    def stored(row: Int) = row % pixelWidth * area + row / pixelWidth
    Convolution(
      weights.bf16(
        Shape.of(area.toLong * pixelWidth, hidden),
        Array.tabulate(weight.length)(i =>
          weight(stored(i / hidden) * hidden + i % hidden)
        )
      ),
      weights.floats(Array.tabulate(bias.length)(row => bias(stored(row))))
    )
  }
  // x_embedder of [3 zeros ; the features ; the place's DCT]: the features'
  // columns, the DCT's part a bias per place in the patch
  private val (pixelEmbed, placeBias) = {
    val weight = weights.values(s"$Decoder.x_embedder.embedder.0.weight")
    val bias = weights.values(s"$Decoder.x_embedder.embedder.0.bias")
    val inputs = weight.length / pixelWidth
    val frequencies = math.round(math.sqrt(inputs - 3 - pixelWidth)).toInt
    val places =
      columns(weight, pixelWidth, 3 + pixelWidth, inputs - 3 - pixelWidth)
    // NerfEmbedder.fetch_pos: positions and frequencies both linspaces
    def position(i: Int) = i.toDouble / (scale - 1)
    def frequency(i: Int) = i.toDouble * frequencies / (frequencies - 1)
    val table = (0 until area).flatMap { p =>
      val (py, px) = (p / scale, p % scale)
      val dct = Array.tabulate(frequencies * frequencies) { i =>
        val (fx, fy) = (frequency(i / frequencies), frequency(i % frequencies))
        (math.cos(position(px) * fx * math.Pi) *
          math.cos(position(py) * fy * math.Pi) / (1 + fx * fy)).toFloat
      }
      affine(places, bias, dct)
    }
    (
      weights.bf16(
        Shape.of(pixelWidth, pixelWidth),
        columns(weight, pixelWidth, 3, pixelWidth)
      ),
      weights.floats(table.toArray)
    )
  }
  private val pixelIn = weights.conv1x1(s"$Decoder.dec_net.input_proj")
  private val pixelCondition = weights.conv1x1(s"$Decoder.dec_net.cond_embed")

  final private case class PixelBlock(
      norm: (Tensor, Tensor),
      modulation: Convolution,
      first: Convolution,
      second: Convolution
  )
  private val pixelBlocks = Iterator
    .from(0)
    .map(i => s"$Decoder.dec_net.res_blocks.$i")
    .takeWhile(p => weights.has(s"$p.in_ln.weight"))
    .map(p =>
      PixelBlock(
        (weights.floats(s"$p.in_ln.weight"), weights.floats(s"$p.in_ln.bias")),
        weights.conv1x1(s"$p.adaLN_modulation.1"),
        weights.conv1x1(s"$p.mlp.0"),
        weights.conv1x1(s"$p.mlp.2")
      )
    )
    .toSeq
  private val pixelNorm = weights.floats(s"$Decoder.final_layer.norm.weight")
  private val pixelOut = weights.conv1x1(s"$Decoder.final_layer.linear")

  // ---- running --------------------------------------------------------------------

  /** `y = x · weightᵀ + bias` on rows. */
  private def linear(x: Tensor, layer: Convolution, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  /** A DiCo block, in place on `x`. */
  private def run(vae: VaeRun, x: Tensor, block: DiCo): Unit = {
    val (height, width, c) = vae.dimensions(x)
    val flat = vae.pixels(x)
    def part(i: Int) = block.modulation.map(_.rows(i, 1).view(c))
    def normed(norm: Option[(Tensor, Tensor)], shift: Int): Tensor = {
      val out = vae.image(height, width, c)
      val rows = vae.pixels(out)
      ops.layerNorm(flat, norm.map(_._1), norm.map(_._2), Epsilon, rows)
      for {
        by <- part(shift + 1)
        plus <- part(shift)
      } ops.modulate(rows, by, plus, rows)
      out
    }
    def residual(y: Tensor, gate: Int): Unit = {
      part(gate) match {
        case Some(g) => ops.gatedAdd(flat, vae.pixels(y), g)
        case None    => ops.add(flat, vae.pixels(y), flat)
      }
      vae.free(y)
    }
    val mixed = vae.conv1x1(normed(block.norm1, 0), block.conv1)
    val local = vae.image(height, width, c)
    ops.depthwiseConv3x3(
      mixed,
      block.depthwise.weight,
      block.depthwise.bias,
      local
    )
    vae.free(mixed)
    ops.activation(Activation.GeluErf, local, local)
    // each channel times the sigmoid of a 1×1 of the image's mean
    val (mean, weight) = (vae.image(1, 1, c), vae.image(1, 1, c))
    ops.columnMean(vae.pixels(local), mean.view(c))
    linear(mean.view(1, c), block.attention, weight.view(1, c))
    ops.activation(Activation.Sigmoid, weight, weight)
    ops.addRow(weight.view(1, c), constant(c, -1f), weight.view(1, c))
    ops.modulate(
      vae.pixels(local),
      weight.view(c),
      constant(c, 0f),
      vae.pixels(local)
    )
    vae.free(mean)
    vae.free(weight)
    residual(vae.conv1x1(local, block.conv3), 2)
    val wide = vae.conv1x1(normed(block.norm2, 3), block.conv4)
    ops.activation(Activation.GeluErf, wide, wide)
    residual(vae.conv1x1(wide, block.conv5), 5)
  }

  /** The mean and log variance of `image`'s latents (`[H, W, 3]`, values in
    * [−1, 1], H and W multiples of `scale`): `[H/scale, W/scale, 2 ×
    * channels]`, the means first.
    */
  def moments(image: Tensor): Tensor = {
    val Seq(height, width, rgb) = image.shape.dimensions
    require(
      rgb == 3 && height % scale == 0 && width % scale == 0,
      s"encode: image ${image.shape}"
    )
    val (gridHeight, gridWidth) = (height / scale, width / scale)
    val vae = new VaeRun(ops)
    try {
      val packed = vae.image(gridHeight, gridWidth, 3L * area)
      ops.packPatches(image, scale, vae.pixels(packed))
      var x = vae.conv1x1(packed, patchEmbed)
      headBlocks.foreach(run(vae, x, _))
      x = vae.conv1x1(vae.conv1x1(x, projectDown), fuse)
      encoderBlocks.foreach(run(vae, x, _))
      val normed = vae.image(gridHeight, gridWidth, hidden)
      ops.layerNorm(
        vae.pixels(x),
        Some(encoderNorm._1),
        Some(encoderNorm._2),
        Epsilon,
        vae.pixels(normed)
      )
      vae.free(x)
      vae.result(vae.conv1x1(normed, encoderOut))
    } finally vae.release()
  }

  /** `image`'s latents as the transformer takes them, `[H/scale × W/scale,
    * channels]` row-major: drawn from the posterior with `random` (`mean +
    * exp(log variance / 2) × noise`, the log variance held within [−20, 10], as
    * the released pipeline encodes), or its mean.
    */
  def encode(image: Tensor, random: Option[SplittableRandom]): Tensor = {
    val both = moments(image)
    try {
      val tokens = both.shape.elementCount / (2 * channels)
      val values = ops.toFloats(both)
      val latents = Array.tabulate(tokens.toInt * channels) { i =>
        val at = i / channels * 2 * channels + i % channels
        val mean = values(at)
        random.fold(mean) { r =>
          val logVariance =
            math.max(-20f, math.min(10f, values(at + channels)))
          mean + math.exp(0.5 * logVariance).toFloat * Images.gaussian(r)
        }
      }
      ops.fromFloats(Shape.of(tokens, channels), latents)
    } finally ops.release(both)
  }

  /** The attention over windows of `window × window` pixels, each on its own
    * (the right and bottom ones filled with the nearest edge pixel), in place
    * on `x`.
    */
  private def attendWindows(x: Tensor, block: PixelAttention): Tensor = {
    val Seq(height, width, c) = x.shape.dimensions.map(_.toInt)
    val count = height.toLong * width
    val flat = x.view(count, c)
    val (down, across) =
      ((height + window - 1) / window, (width + window - 1) / window)
    val inside = window * window
    val padded = down.toLong * across * inside
    // a window's rows from the pixels, and a pixel's row among the windows'
    val gather = Array.tabulate(padded.toInt) { row =>
      val (w, p) = (row / inside, row % inside)
      val y = math.min(w / across * window + p / window, height - 1)
      val xx = math.min(w % across * window + p % window, width - 1)
      y * width + xx
    }
    val scatter = Array.tabulate(count.toInt) { pixel =>
      val (y, xx) = (pixel / width, pixel % width)
      (y / window * across + xx / window) * inside +
        y % window * window + xx % window
    }
    val temporaries = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = {
      temporaries += tensor
      tensor
    }
    def allocate(dtype: DType, rows: Long, cols: Long) =
      keep(ops.allocate(dtype, Shape.of(rows, cols)))
    try {
      val normed = allocate(DType.F32, count, c)
      block.norm(ops, flat, normed)
      val projected = Seq(block.q, block.k, block.v).map { layer =>
        val all = allocate(DType.F32, count, c)
        linear(normed, layer, all)
        all
      }
      val rows = keep(ops.fromInts(Shape.of(padded), gather))
      val Seq(q, k, v) = projected.map { all =>
        val windows = allocate(DType.F32, padded, c)
        ops.embedding(all, rows, windows)
        windows
      }
      // the keys, and each window's values transposed, as the BF16 "weights"
      // of two GEMMs
      val keys = allocate(DType.BF16, padded, c)
      ops.convert(k, keys)
      val transposed = allocate(DType.F32, c, inside)
      val values = allocate(DType.BF16, c, inside)
      val scores = allocate(DType.F32, inside, inside)
      val attended = k // its rows are free once converted
      (0L until down.toLong * across).foreach { w =>
        val first = w * inside
        ops.transpose(v.rows(first, inside), transposed)
        ops.convert(transposed, values)
        ops.linear(q.rows(first, inside), keys.rows(first, inside), scores)
        ops.softmax(scores, (1 / math.sqrt(c.toDouble)).toFloat, scores)
        ops.linear(scores, values, attended.rows(first, inside))
      }
      val back = keep(ops.fromInts(Shape.of(count), scatter))
      ops.embedding(attended, back, normed)
      val out = projected.head
      linear(normed, block.proj, out)
      ops.add(flat, out, flat)
      x
    } finally temporaries.foreach(ops.release)
  }

  /** Patches the per-pixel decoder runs at once. */
  private def patchesAtOnce = math.max(1, (1 << 19) / area)

  /** The image of `latents` (`[H/scale, W/scale, channels]`): `[H, W, 3]`,
    * values about [−1, 1].
    */
  def decode(latents: Tensor): Tensor = {
    val Seq(gridHeight, gridWidth, deep) = latents.shape.dimensions
    require(deep == channels, s"decode: latents ${latents.shape}")
    val patches = gridHeight * gridWidth
    val vae = new VaeRun(ops)
    try {
      val copy = vae.image(gridHeight, gridWidth, channels)
      ops.copy(latents, copy)
      var x = vae.conv3x3(copy, conditionIn)
      conditionBlocks.foreach {
        case Left(block)  => x = vae.residual(x, block)
        case Right(block) => x = attendWindows(x, block)
      }
      val normed = vae.normSilu(x, conditionNorm)
      vae.free(x)
      val condition = vae.conv3x3(normed, conditionOut)
      val stream = vae.image(gridHeight, gridWidth, hidden)
      linear(vae.pixels(condition), streamIn, vae.pixels(stream))
      decoderBlocks.foreach(run(vae, stream, _))

      val rgb = vae.image(1, patches * area, 3)
      val chunk = math.min(patches, patchesAtOnce.toLong)
      val wide = area.toLong * pixelWidth
      val features = vae.image(1, chunk, wide)
      val modulators = vae.image(1, chunk, wide)
      val Seq(x1, x2, x3) =
        Seq.fill(3)(vae.image(1, chunk * area, pixelWidth))
      val table = vae.image(1, chunk * area, 3L * pixelWidth)
      (0L until patches by chunk).foreach { first =>
        val n = math.min(chunk, patches - first)
        val m = n * area
        def pixels(t: Tensor) = t.view(chunk * area, pixelWidth).rows(0, m)
        def perPatch(t: Tensor) = t.view(chunk, wide).rows(0, n)
        // a pixel's features from its patch's condition, and its place
        linear(
          vae.pixels(condition).rows(first, n),
          pixelFeatures,
          perPatch(features)
        )
        ops.linear(pixels(features), pixelEmbed, pixels(x1))
        ops.addRow(perPatch(x1), placeBias, perPatch(x1))
        linear(pixels(x1), pixelIn, pixels(x2))
        // its modulations from its patch's stream
        linear(
          vae.pixels(stream).rows(first, n),
          pixelCondition,
          perPatch(modulators)
        )
        ops.activation(Activation.Silu, modulators, modulators)
        val chunks = table.view(chunk * area, 3L * pixelWidth).rows(0, m)
        pixelBlocks.foreach { block =>
          linear(pixels(modulators), block.modulation, chunks)
          ops.layerNorm(
            pixels(x2),
            Some(block.norm._1),
            Some(block.norm._2),
            Epsilon,
            pixels(x1)
          )
          ops.modulateChunks(pixels(x1), chunks, 0, 1, pixels(x1))
          linear(pixels(x1), block.first, pixels(x3))
          ops.activation(Activation.Silu, pixels(x3), pixels(x3))
          linear(pixels(x3), block.second, pixels(x1))
          ops.gatedAddChunk(pixels(x2), pixels(x1), chunks, 2)
        }
        ops.rmsNorm(pixels(x2), pixelNorm, Epsilon, 0f, pixels(x2))
        linear(
          pixels(x2),
          pixelOut,
          rgb.view(patches * area, 3).rows(first * area, m)
        )
      }
      val image = vae.image(gridHeight * scale, gridWidth * scale, 3)
      ops.patchesToPixels(rgb.view(patches * area, 3), scale, image)
      vae.result(image)
    } finally vae.release()
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object MageVae {

  /** The side of the decoder's attention windows, in latents (the released
    * code's `AttnBlock(patch_size=32)`; it is no weight).
    */
  val Window = 32

  /** Whether the weights are a Mage-Flow VAE's. */
  def holds(source: WeightSource): Boolean =
    source.has("student.dconv_encoder.patch_cond_embed.weight")

  def open(ops: Ops, path: Path, window: Int = Window): MageVae = {
    val source = WeightSource.open(ops, path)
    try new MageVae(ops, source, window)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
