package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** PiD 1.5's sizes, read from the weights (the 1024→4096 decoders: 1536 wide,
  * 24 heads of 64, 14 patch blocks over 16×16 patches, 2 pixel blocks of 16
  * channels attending as 16 heads of 72, text 300 × 2304, a latent stack of
  * 1024 over 4 residual blocks, 7 injections, one every 2 blocks). The latent
  * stack's input tells the variant, as sd-cpp infers it: 32 channels for
  * FLUX.2's latent (128 packed features at 1/16, unpacked 2×2), 16 for FLUX.1's
  * and Qwen Image's (16 channels at 1/8).
  */
final case class PidConfig(
    hidden: Int,
    heads: Int,
    headDimension: Int,
    patch: Int,
    patchBlocks: Int,
    pixelBlocks: Int,
    pixelChannels: Int,
    pixelHeads: Int,
    pixelHead: Int,
    intermediate: Int,
    textWidth: Int,
    textLength: Int,
    latentHidden: Int,
    latentChannels: Int,
    residualBlocks: Int,
    injections: Int,
    timeChannels: Int,
    /** The latent's 2×2 packing, undone before the stack: 2 or 1. */
    latentUnpatchify: Int,
    /** Image pixels per latent row along each side: 16 or 8. */
    latentDown: Int
) {

  /** Features per latent row. */
  def latentFeatures: Int = latentUnpatchify * latentUnpatchify * latentChannels

  /** Patch blocks between two injections. */
  def interval: Int = (patchBlocks + injections - 1) / injections

  /** The pixel blocks' heads padded to whole WMMA steps of 16 (72 → 80). */
  def paddedPixelHead: Int = (pixelHead + 15) / 16 * 16
}

object PidConfig {

  def of(source: WeightSource): PidConfig = {
    def dimensions(name: String) =
      source(name).shape.dimensions.map(_.toInt)
    def count(prefix: String, part: String) =
      Iterator.from(0).takeWhile(i => source.has(s"$prefix.$i.$part")).size
    val Seq(hidden, patchInputs) = dimensions("s_embedder.proj.weight")
    val headDimension = dimensions("patch_blocks.0.attn.q_norm_x.weight").head
    val pixelHead = dimensions("pixel_blocks.0.attn.q_norm.weight").head
    val latent = dimensions("lq_proj.latent_proj.0.weight")
    PidConfig(
      hidden = hidden,
      heads = hidden / headDimension,
      headDimension = headDimension,
      patch = math.round(math.sqrt(patchInputs / 3.0)).toInt,
      patchBlocks = count("patch_blocks", "attn.qkv_x.weight"),
      pixelBlocks = count("pixel_blocks", "attn.qkv.weight"),
      pixelChannels = dimensions("pixel_embedder.proj.weight").head,
      pixelHeads =
        dimensions("pixel_blocks.0.compress_to_attn.weight").head / pixelHead,
      pixelHead = pixelHead,
      intermediate = dimensions("patch_blocks.0.mlp_x.w1.weight").head,
      textWidth = dimensions("y_embedder.proj.weight").last,
      textLength = dimensions("y_pos_embedding")(1),
      latentHidden = latent.head,
      latentChannels = latent(1),
      residualBlocks = Iterator
        .from(3)
        .takeWhile(i => source.has(s"lq_proj.latent_proj.$i.block.0.weight"))
        .size,
      injections = count("lq_proj.output_heads", "weight"),
      timeChannels = dimensions("t_embedder.mlp.0.weight").last,
      latentUnpatchify = if (latent(1) == 32) 2 else 1,
      latentDown = if (latent(1) == 32) 16 else 8
    )
  }
}

/** NVIDIA's PiD 1.5 (pixel diffusion decoder, `nv-tlabs/PiD`'s `PidNet` over
  * PixelDiT), from ComfyUI's single file (`net.`). It decodes a FLUX.2, FLUX.1
  * or Qwen Image latent (`config.latentUnpatchify`) straight to pixels at 4×
  * its image, conditioned on Gemma 2 text:
  *   - the latent (as its VAE gives it; FLUX.2's packed and unpacked here)
  *     nearest-upsampled to the patch grid and run once through a convolution
  *     stack (replicate padding, GroupNorm of 4) whose heads give one feature
  *     map per injection and one for the pixel blocks (`conditionOn`);
  *   - patch tokens (`patch`² pixels each) through MMDiT blocks with the text
  *     (joint attention over [text ; image], RMS norms `w · x̂`, adaLN from the
  *     timestep, SwiGLU), a gated injection of the latent's features before
  *     every `interval`-th block;
  *   - then pixel blocks: each pixel's channels modulated per pixel by its
  *     token, compressed per patch into attention over the patches, expanded
  *     back, and an MLP per pixel; a final norm and linear to RGB.
  * The network predicts the velocity `noise − image`. RoPE is 2-D over the
  * patch grid (positions `linspace(0, 16)`, NTK-scaled bases against a
  * `ropeReference`-pixel image) and 1-D over the text. The pixel blocks' heads
  * of 72 are padded to 80 at load (zero weights; the head norms' weights
  * rescaled so the RMS over 80 is the RMS over 72).
  */
final class Pid private (ops: Ops, source: WeightSource, ropeReference: Int)
    extends AutoCloseable {

  if (!source.has("s_embedder.proj.weight"))
    throw new FormatException("no PiD network (s_embedder.proj)")
  val config: PidConfig = PidConfig.of(source)
  private val c = config
  private val weights = new VaeWeights(ops, source)
  private val Epsilon = 1e-6f

  private def matrix(name: String): Tensor = source(name)
  private def vector(name: String): Tensor = weights.floats(name)
  private def scalar(name: String): Float = weights.values(name).head

  /** A linear layer and its bias. */
  final private case class Affine(weight: Tensor, bias: Tensor)
  private def affine(name: String) =
    Affine(matrix(s"$name.weight"), vector(s"$name.bias"))

  /** `out = x · weight + bias`. */
  private def project(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  // ---- weights ---------------------------------------------------------------------

  private val h = c.hidden.toLong

  /** One stream of a patch block (`x` the image, `y` the text). */
  final private class Stream(block: Int, s: String) {
    private val prefix = s"patch_blocks.$block"
    private val qkv = matrix(s"$prefix.attn.qkv_$s.weight")
    val (q, k, v) = (qkv.rows(0, h), qkv.rows(h, h), qkv.rows(2 * h, h))
    val qNorm: Tensor = vector(s"$prefix.attn.q_norm_$s.weight")
    val kNorm: Tensor = vector(s"$prefix.attn.k_norm_$s.weight")
    val proj: Affine = affine(s"$prefix.attn.proj_$s")
    val norm1: Tensor = vector(s"$prefix.norm_${s}1.weight")
    val norm2: Tensor = vector(s"$prefix.norm_${s}2.weight")
    val w1: Tensor = matrix(s"$prefix.mlp_$s.w1.weight")
    val w2: Tensor = matrix(s"$prefix.mlp_$s.w2.weight")
    val w3: Tensor = matrix(s"$prefix.mlp_$s.w3.weight")
    val modulation: Affine = affine(
      s"$prefix.adaLN_modulation_${if (s == "x") "img" else "txt"}.0"
    )
  }

  private val patchBlocks =
    (0 until c.patchBlocks).map(i => (new Stream(i, "x"), new Stream(i, "y")))

  /** A sigma-aware per-token gate: `s += sigmoid(proj([s ; f]) − e^α σ) f`. */
  final private case class Gate(content: Affine, alpha: Float)
  private def gate(prefix: String) =
    Gate(
      affine(s"$prefix.content_proj"),
      math.exp(scalar(s"$prefix.log_alpha")).toFloat
    )
  private val gates =
    (0 until c.injections).map(i => gate(s"lq_proj.gate_modules.$i"))
  private val pixelGate = gate("pit_lq_gate")

  /** A pixel block: its attention's heads padded to `paddedPixelHead`. */
  final private class PixelBlock(i: Int) {
    private val prefix = s"pixel_blocks.$i"
    private val (heads, head, padded) =
      (c.pixelHeads, c.pixelHead, c.paddedPixelHead)
    val norm1: Tensor = vector(s"$prefix.norm1.weight")
    val norm2: Tensor = vector(s"$prefix.norm2.weight")
    val modulation: Affine = affine(s"$prefix.adaLN_modulation.0")
    val compress: Affine = affine(s"$prefix.compress_to_attn")
    val expand: Affine = affine(s"$prefix.expand_from_attn")
    val fc1: Affine = affine(s"$prefix.mlp.fc1")
    val fc2: Affine = affine(s"$prefix.mlp.fc2")
    private val width = heads * head
    // q, k, v: each head's rows followed by zero rows
    val qkv: Tensor = {
      val values = weights.values(s"$prefix.attn.qkv.weight")
      val padding = new Array[Float](heads * 3 * padded * width)
      for {
        part <- 0 until 3
        n <- 0 until heads
        r <- 0 until head
      }
        System.arraycopy(
          values,
          ((part * heads + n) * head + r) * width,
          padding,
          ((part * heads + n) * padded + r) * width,
          width
        )
      weights.bf16(Shape.of(3L * heads * padded, width), padding)
    }
    val (q, k, v) = {
      val rows = heads.toLong * padded
      (qkv.rows(0, rows), qkv.rows(rows, rows), qkv.rows(2 * rows, rows))
    }
    // the RMS over `padded` values of which `head` are the head's
    private def headNorm(name: String) = {
      val values = weights.values(s"$prefix.attn.$name.weight")
      val factor = math.sqrt(head.toDouble / padded).toFloat
      weights.floats(
        Array.tabulate(padded)(i => if (i < head) values(i) * factor else 0f)
      )
    }
    val qNorm: Tensor = headNorm("q_norm")
    val kNorm: Tensor = headNorm("k_norm")
    // the output projection's columns padded alike
    val proj: Affine = {
      val values = weights.values(s"$prefix.attn.proj.weight")
      val out = values.length / width
      val padding = new Array[Float](out * heads * padded)
      for {
        o <- 0 until out
        n <- 0 until heads
      }
        System.arraycopy(
          values,
          o * width + n * head,
          padding,
          (o * heads + n) * padded,
          head
        )
      Affine(
        weights.bf16(Shape.of(out.toLong, heads.toLong * padded), padding),
        vector(s"$prefix.attn.proj.bias")
      )
    }
  }

  private val pixelBlocks = (0 until c.pixelBlocks).map(new PixelBlock(_))
  private val patchEmbedder = affine("s_embedder.proj")
  private val pixelEmbedder = affine("pixel_embedder.proj")
  private val textIn = affine("y_embedder.proj")
  private val textNorm = vector("y_embedder.norm.weight")
  private val textPositions = {
    val values = weights.values(s"y_pos_embedding")
    weights.keep(ops.fromFloats(Shape.of(c.textLength.toLong, h), values))
  }
  private val timeIn = affine("t_embedder.mlp.0")
  private val timeOut = affine("t_embedder.mlp.2")
  private val finalNorm = vector("final_layer.norm.weight")
  private val finalOut = affine("final_layer.linear")

  // the latent stack
  private val latentIn = weights.conv3x3(s"lq_proj.latent_proj.0")
  private val latentSecond = weights.conv3x3(s"lq_proj.latent_proj.2")
  final private case class LatentResidual(
      norm1: (Tensor, Tensor),
      conv1: Convolution,
      norm2: (Tensor, Tensor),
      conv2: Convolution
  )
  private val latentResiduals = (0 until c.residualBlocks).map { i =>
    val prefix = s"lq_proj.latent_proj.${3 + i}.block"
    LatentResidual(
      (vector(s"$prefix.0.weight"), vector(s"$prefix.0.bias")),
      weights.conv3x3(s"$prefix.2"),
      (vector(s"$prefix.3.weight"), vector(s"$prefix.3.bias")),
      weights.conv3x3(s"$prefix.5")
    )
  }
  private val outputHeads =
    (0 until c.injections).map(i => affine(s"lq_proj.output_heads.$i"))
  private val pixelHead = affine("lq_proj.pit_head")

  // ---- per image -------------------------------------------------------------------

  /** A patch grid and the tensors that depend on it and on the conditioning:
    * the latent's features, the RoPE tables, the pixels' positions, the
    * attention caches. Released by `release`.
    */
  final class Conditioning private[Pid] (
      val gridHeight: Int,
      val gridWidth: Int,
      val text: Tensor,
      val features: Seq[Tensor],
      val pixelFeatures: Tensor,
      val degradeSigma: Float,
      /** RoPE cosines and sines: the patch attention's image grid, the pixel
        * attention's, the text's.
        */
      private[Pid] val ropes: Seq[(Tensor, Tensor)],
      private[Pid] val held: Seq[Tensor],
      private[Pid] val caches: Seq[KvCache]
  ) {
    def tokens: Int = gridHeight * gridWidth
    def release(): Unit = {
      held.foreach(ops.release)
      caches.foreach(cache => {
        ops.release(cache.keys)
        ops.release(cache.values)
      })
    }
  }

  /** The conditioning of one decode: the text's encoder features (`[textLength,
    * textWidth]`, Gemma 2's last hidden state), the latent (`[h × w,
    * latentFeatures]` rows on an `h × w` grid, as its VAE's `encode` gives it)
    * and the latent's own noise level (`degradeSigma`, 0 for a clean latent).
    * With `u` the unpatchify factor, the image is `u patch h × u patch w` when
    * the latent is upsampled ×1, and larger by the power of two `gridHeight / u
    * h`.
    */
  def conditionOn(
      textFeatures: Tensor,
      latent: Tensor,
      latentHeight: Int,
      gridHeight: Int,
      gridWidth: Int,
      degradeSigma: Float
  ): Conditioning = {
    val held = mutable.ArrayBuffer.empty[Tensor]
    def allocate(shape: Shape) = {
      val tensor = ops.allocate(DType.F32, shape)
      held += tensor
      tensor
    }
    val latentWidth =
      (latent.shape.dimensions.head / latentHeight).toInt
    val unpatchify = c.latentUnpatchify
    val factor = gridHeight / (unpatchify * latentHeight)
    require(
      Integer.bitCount(factor) == 1 &&
        gridHeight == unpatchify * latentHeight * factor &&
        gridWidth == unpatchify * latentWidth * factor &&
        latent.shape.last == c.latentFeatures,
      s"conditionOn: latent ${latent.shape} on $latentHeight rows for a $gridHeight × $gridWidth grid"
    )
    // text: projected, normed, plus the learned positions
    val length = textFeatures.shape.dimensions.head
    require(
      length <= c.textLength && textFeatures.shape.last == c.textWidth,
      s"conditionOn: text ${textFeatures.shape}"
    )
    val text = allocate(Shape.of(length, h))
    project(textFeatures, textIn, text)
    ops.rmsNorm(text, textNorm, Epsilon, 0f, text)
    ops.add(text, textPositions.rows(0, length), text)
    // the latent: unpacked, upsampled to the grid, the convolution stack
    val run = new VaeRun(ops)
    val tokens = gridHeight.toLong * gridWidth
    val (features, pixelFeatures) =
      try {
        var z = run.image(
          unpatchify.toLong * latentHeight,
          unpatchify.toLong * latentWidth,
          c.latentChannels
        )
        ops.unpackPatches(latent, latentHeight, unpatchify, z)
        var scale = 1
        while (scale < factor) {
          val (height, width, channels) = run.dimensions(z)
          val up = run.image(2 * height, 2 * width, channels)
          ops.upsample2x(z, up)
          run.free(z)
          z = up
          scale *= 2
        }
        def conv(x: Tensor, convolution: Convolution) = {
          val (height, width, _) = run.dimensions(x)
          val out = run.image(height, width, convolution.outChannels)
          ops.conv3x3(
            x,
            convolution.weight,
            convolution.bias,
            out,
            replicate = true
          )
          run.free(x)
          out
        }
        def normSilu(x: Tensor, norm: (Tensor, Tensor)) = {
          val (height, width, channels) = run.dimensions(x)
          val out = run.image(height, width, channels)
          ops.groupNorm(
            run.pixels(x),
            4,
            norm._1,
            norm._2,
            1e-5f,
            run.pixels(out)
          )
          ops.activation(Activation.Silu, out, out)
          out
        }
        var x = conv(z, latentIn)
        ops.activation(Activation.Silu, x, x)
        x = conv(x, latentSecond)
        latentResiduals.foreach { block =>
          val inner = conv(normSilu(x, block.norm1), block.conv1)
          val outer = conv(normSilu(inner, block.norm2), block.conv2)
          ops.add(outer, x, outer)
          run.free(x)
          x = outer
        }
        val rows = run.pixels(x)
        val heads = outputHeads.map { head =>
          val out = allocate(Shape.of(tokens, h))
          project(rows, head, out)
          out
        }
        val pixel = allocate(Shape.of(tokens, h))
        project(rows, pixelHead, pixel)
        (heads, pixel)
      } finally run.release()
    // RoPE tables: the image grid for both attentions, the text
    def table(pairs: Array[Array[Double]]) = {
      val (cosines, sines) =
        (
          pairs.flatten.map(a => math.cos(a).toFloat),
          pairs.flatten.map(a => math.sin(a).toFloat)
        )
      val shape = Shape.of(pairs.length.toLong, pairs.head.length)
      val made = (ops.fromFloats(shape, cosines), ops.fromFloats(shape, sines))
      held += made._1
      held += made._2
      made
    }
    val imageRope = table(
      Pid.imageAngles(
        c.headDimension,
        gridHeight,
        gridWidth,
        ropeReference / c.patch
      )
    )
    val pixelRope = table(
      Pid.imageAngles(
        c.pixelHead,
        gridHeight,
        gridWidth,
        ropeReference / c.patch
      )
    )
    val textRope = table(Array.tabulate(length.toInt) { t =>
      Array.tabulate(c.headDimension / 2)(k =>
        t * math.pow(10000, -2.0 * k / c.headDimension)
      )
    })
    val caches = Seq(
      ops.allocateCache(
        1,
        ((length + tokens + 15) / 16 * 16).toInt,
        c.heads,
        c.headDimension
      ),
      ops.allocateCache(
        1,
        ((tokens + 15) / 16 * 16).toInt,
        c.pixelHeads,
        c.paddedPixelHead
      )
    )
    new Conditioning(
      gridHeight,
      gridWidth,
      text,
      features,
      pixelFeatures,
      degradeSigma,
      Seq(imageRope, pixelRope, textRope),
      held.toSeq,
      caches
    )
  }

  /** The pixels' fixed positions per grid, patch-ordered `[pixels, C]`: the
    * official 2-D sin-cos table (per pixel, `C/4` sines then cosines of the
    * column, then of the row, frequencies `10000^(−k / (C/4))`).
    */
  private val positions = mutable.Map.empty[(Int, Int), Tensor]

  private def positionsFor(gridHeight: Int, gridWidth: Int): Tensor =
    positions.getOrElseUpdate(
      (gridHeight, gridWidth), {
        val (height, width) = (gridHeight * c.patch, gridWidth * c.patch)
        val channels = c.pixelChannels
        val quarter = channels / 4
        def encode(n: Int) = Array.tabulate(n, 2 * quarter) { (i, k) =>
          val angle = i * math.pow(10000, -(k % quarter).toDouble / quarter)
          (if (k < quarter) math.sin(angle) else math.cos(angle)).toFloat
        }
        val (columns, rows) = (encode(width), encode(height))
        val values = new Array[Float](height * width * channels)
        for {
          y <- 0 until height
          x <- 0 until width
        } {
          val at = (y * width + x) * channels
          System.arraycopy(columns(x), 0, values, at, 2 * quarter)
          System.arraycopy(rows(y), 0, values, at + 2 * quarter, 2 * quarter)
        }
        val image =
          ops.fromFloats(
            Shape.of(height.toLong, width.toLong, channels),
            values
          )
        try {
          val ordered = ops.allocate(
            DType.F32,
            Shape.of(height.toLong * width, channels)
          )
          ops.pixelsToPatches(image, c.patch, ordered)
          ordered
        } finally ops.release(image)
      }
    )

  // ---- per step --------------------------------------------------------------------

  /** The attention's values go through an F16 cache (largest 65504), which
    * PiD's pixel values outgrow: its activations reach 1e8 (the official net
    * runs in BF16). Attention is linear in the values, so values beyond this
    * are scaled down by a power of two before the cache and the output back.
    */
  private val ValueLimit = 8192f

  /** The power of two `values` are divided by (1 when they fit), applied. */
  private def fitValues(values: Tensor): Float = {
    val largest = ops.maxAbs(values)
    if (largest <= ValueLimit) 1f
    else {
      val factor =
        math
          .pow(2, math.ceil(math.log(largest / ValueLimit) / math.log(2)))
          .toFloat
      ops.scale(values, 1 / factor, values)
      factor
    }
  }

  private val pageTable = weights.keep(ops.fromInts(Shape.of(1), Array(0)))

  /** The velocity at `sigma` of the noisy image `x` (`[patch gridHeight, patch
    * gridWidth, 3]`, channels-last) under `conditioning`, into `out` (like
    * `x`). Its intermediates are released but for `out`.
    */
  def velocity(
      x: Tensor,
      sigma: Float,
      conditioning: Conditioning,
      out: Tensor,
      /** Sees each stage's output, for debugging: `(stage, tensor)`. */
      inspect: (String, Tensor) => Unit = (_, _) => ()
  ): Unit = {
    val (gridHeight, gridWidth) =
      (conditioning.gridHeight, conditioning.gridWidth)
    val tokens = conditioning.tokens.toLong
    val area = c.patch.toLong * c.patch
    require(
      x.shape == Shape
        .of(gridHeight.toLong * c.patch, gridWidth.toLong * c.patch, 3) &&
        out.shape == x.shape,
      s"velocity: x ${x.shape}, out ${out.shape} on a $gridHeight × $gridWidth grid"
    )
    val Seq((imageCos, imageSin), (pixelCos, pixelSin), (textCos, textSin)) =
      conditioning.ropes
    val live = mutable.ArrayBuffer.empty[Tensor]
    def allocate(rows: Long, cols: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(rows, cols))
      live += tensor
      tensor
    }
    try {
      // the timestep: σ × 1000, a sinusoid of max period 10 (cosines first)
      val half = c.timeChannels / 2
      val t = sigma * 1000f
      val sinusoid = Array.tabulate(2 * half) { i =>
        val angle = t * math.exp(-math.log(10) * (i % half) / half)
        (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
      }
      val embedded = ops.fromFloats(Shape.of(1, 2L * half), sinusoid)
      live += embedded
      val (inner, time, condition) =
        (allocate(1, h), allocate(1, h), allocate(1, h))
      project(embedded, timeIn, inner)
      ops.activation(Activation.Silu, inner, inner)
      project(inner, timeOut, time)
      ops.activation(Activation.Silu, time, condition)
      inspect("time", time)

      // the patch tokens, and the text beside them
      val patches = allocate(tokens, 3 * area)
      ops.packPatches(x, c.patch, patches)
      val s = allocate(tokens, h)
      project(patches, patchEmbedder, s)
      inspect("patches", s)
      val length = conditioning.text.shape.dimensions.head
      val y = allocate(length, h)
      ops.copy(conditioning.text, y)

      val all = length + tokens
      val (q, k, v, attended) =
        (allocate(all, h), allocate(all, h), allocate(all, h), allocate(all, h))
      val (rotatedQueries, rotatedKeys) = (allocate(all, h), allocate(all, h))
      // the widest stream: the image's tokens, or a longer text
      val widest = math.max(tokens, length)
      val normed = allocate(widest, h)
      val projected = allocate(widest, h)
      val (w1, w3) =
        (allocate(widest, c.intermediate), allocate(widest, c.intermediate))
      val joined = allocate(tokens, 2 * h)
      val gateValue = allocate(tokens, 1)
      val modulation = allocate(1, 6 * h)
      val d = c.headDimension.toLong
      val heads = c.heads.toLong
      val patchCache = conditioning.caches.head

      def inject(target: Tensor, feature: Tensor, g: Gate): Unit = {
        ops.concatColumns(Seq(target, feature), joined)
        project(joined, g.content, gateValue)
        if (conditioning.degradeSigma != 0f)
          ops.addRow(
            gateValue, {
              val shift = ops.fromFloats(
                Shape.of(1),
                Array(-g.alpha * conditioning.degradeSigma)
              )
              live += shift
              shift
            },
            gateValue
          )
        ops.activation(Activation.Sigmoid, gateValue, gateValue)
        ops.rowGatedAdd(target, feature, gateValue)
      }
      def chunk(i: Int) = modulation.view(6, h).rows(i, 1).view(h)
      def headNorm(t: Tensor, weight: Tensor) = {
        val rows = t.shape.dimensions.head * heads
        ops.rmsNorm(t.view(rows, d), weight, Epsilon, 0f, t.view(rows, d))
      }

      patchBlocks.zipWithIndex.foreach { case ((image, text), i) =>
        if (i % c.interval == 0) {
          inject(
            s,
            conditioning.features(i / c.interval),
            gates(i / c.interval)
          )
          inspect(s"injected $i", s)
        }
        val last = i == c.patchBlocks - 1
        // each stream normed, modulated and projected into [text ; image]
        val streams = Seq((text, y, 0L, length), (image, s, length, tokens))
        streams.foreach { (stream, z, first, n) =>
          def rows(t: Tensor) = t.rows(first, n)
          project(condition, stream.modulation, modulation)
          val zn = normed.rows(0, n)
          ops.rmsNorm(z, stream.norm1, Epsilon, 0f, zn)
          ops.modulate(zn, chunk(1), chunk(0), zn)
          ops.linears(
            zn,
            Seq(stream.q, stream.k, stream.v),
            Seq(rows(q), rows(k), rows(v))
          )
          headNorm(rows(q), stream.qNorm)
          headNorm(rows(k), stream.kNorm)
        }
        def rope(t: Tensor, out: Tensor) = {
          ops.ropeTable(
            t.rows(0, length).view(length, heads, d),
            textCos,
            textSin,
            out.rows(0, length).view(length, heads, d)
          )
          ops.ropeTable(
            t.rows(length, tokens).view(tokens, heads, d),
            imageCos,
            imageSin,
            out.rows(length, tokens).view(tokens, heads, d)
          )
        }
        inspect(s"queries $i", q)
        rope(q, rotatedQueries)
        rope(k, rotatedKeys)
        inspect(s"rotated queries $i", rotatedQueries)
        val valueScale = fitValues(v)
        ops.cacheWrite(
          rotatedKeys.view(all, heads, d),
          v.view(all, heads, d),
          patchCache,
          pageTable,
          0
        )
        ops.attention(
          rotatedQueries.view(all, heads, d),
          patchCache,
          pageTable,
          0,
          all.toInt,
          Attention(
            (1 / math.sqrt(d.toDouble)).toFloat,
            causal = false,
            None,
            None,
            None
          ),
          attended.view(all, heads, d)
        )
        if (valueScale != 1f) ops.scale(attended, valueScale, attended)
        inspect(s"attended $i", attended)
        // the text's last outputs are never read
        streams.foreach { (stream, z, first, n) =>
          if (!(last && (z eq y))) {
            project(condition, stream.modulation, modulation)
            val out = projected.rows(0, n)
            project(attended.rows(first, n), stream.proj, out)
            ops.gatedAdd(z, out, chunk(2))
            val zn = normed.rows(0, n)
            ops.rmsNorm(z, stream.norm2, Epsilon, 0f, zn)
            ops.modulate(zn, chunk(4), chunk(3), zn)
            val (a, b) = (w1.rows(0, n), w3.rows(0, n))
            ops.linears(zn, Seq(stream.w1, stream.w3), Seq(a, b))
            ops.gated(Activation.Silu, a, b, a)
            ops.linear(a, stream.w2, out)
            ops.gatedAdd(z, out, chunk(5))
          }
        }
      }

      // the patch tokens as the pixels' conditioning
      ops.addRow(s, time.view(h), s)
      ops.activation(Activation.Silu, s, s)
      inject(s, conditioning.pixelFeatures, pixelGate)
      inspect("pixel conditioning", s)

      // the pixels, patch-ordered, embedded, plus their fixed positions
      val pixels = tokens * area
      val channels = c.pixelChannels.toLong
      val rgb = allocate(pixels, 3)
      ops.pixelsToPatches(x, c.patch, rgb)
      val stream = allocate(pixels, channels)
      project(rgb, pixelEmbedder, stream)
      ops.add(stream, positionsFor(gridHeight, gridWidth), stream)
      inspect("pixels", stream)

      val table = allocate(tokens, 6 * area * channels)
      val pixelNormed = allocate(pixels, channels)
      val (pixelHeads, padded) = (c.pixelHeads.toLong, c.paddedPixelHead.toLong)
      val compressed = allocate(tokens, pixelHeads * c.pixelHead)
      val (pq, pk, pv, pAttended) = (
        allocate(tokens, pixelHeads * padded),
        allocate(tokens, pixelHeads * padded),
        allocate(tokens, pixelHeads * padded),
        allocate(tokens, pixelHeads * padded)
      )
      val (pRotatedQueries, pRotatedKeys) =
        (
          allocate(tokens, pixelHeads * padded),
          allocate(tokens, pixelHeads * padded)
        )
      val expanded = allocate(tokens, area * channels)
      val hidden =
        allocate(pixels, pixelBlocks.head.fc1.weight.shape.dimensions.head)
      val pixelCache = conditioning.caches(1)
      val rows = table.view(pixels, 6 * channels)
      pixelBlocks.foreach { block =>
        project(s, block.modulation, table)
        ops.rmsNorm(stream, block.norm1, Epsilon, 0f, pixelNormed)
        ops.modulateChunks(pixelNormed, rows, 0, 1, pixelNormed)
        project(
          pixelNormed.view(tokens, area * channels),
          block.compress,
          compressed
        )
        ops.linears(compressed, Seq(block.q, block.k, block.v), Seq(pq, pk, pv))
        Seq((pq, block.qNorm), (pk, block.kNorm)).foreach { (t, weight) =>
          ops.rmsNorm(
            t.view(tokens * pixelHeads, padded),
            weight,
            Epsilon,
            0f,
            t.view(tokens * pixelHeads, padded)
          )
        }
        ops.ropeTable(
          pq.view(tokens, pixelHeads, padded),
          pixelCos,
          pixelSin,
          pRotatedQueries.view(tokens, pixelHeads, padded)
        )
        ops.ropeTable(
          pk.view(tokens, pixelHeads, padded),
          pixelCos,
          pixelSin,
          pRotatedKeys.view(tokens, pixelHeads, padded)
        )
        val valueScale = fitValues(pv)
        ops.cacheWrite(
          pRotatedKeys.view(tokens, pixelHeads, padded),
          pv.view(tokens, pixelHeads, padded),
          pixelCache,
          pageTable,
          0
        )
        ops.attention(
          pRotatedQueries.view(tokens, pixelHeads, padded),
          pixelCache,
          pageTable,
          0,
          tokens.toInt,
          Attention(
            (1 / math.sqrt(c.pixelHead.toDouble)).toFloat,
            causal = false,
            None,
            None,
            None
          ),
          pAttended.view(tokens, pixelHeads, padded)
        )
        if (valueScale != 1f) ops.scale(pAttended, valueScale, pAttended)
        project(pAttended, block.proj, compressed)
        project(compressed, block.expand, expanded)
        ops.gatedAddChunk(stream, expanded.view(pixels, channels), rows, 2)
        ops.rmsNorm(stream, block.norm2, Epsilon, 0f, pixelNormed)
        ops.modulateChunks(pixelNormed, rows, 3, 4, pixelNormed)
        project(pixelNormed, block.fc1, hidden)
        ops.activation(Activation.GeluErf, hidden, hidden)
        project(hidden, block.fc2, pixelNormed)
        ops.gatedAddChunk(stream, pixelNormed, rows, 5)
        inspect("pixel block", stream)
      }
      ops.rmsNorm(stream, finalNorm, Epsilon, 0f, pixelNormed)
      project(pixelNormed, finalOut, rgb)
      ops.patchesToPixels(rgb, c.patch, out)
    } finally live.foreach(ops.release)
  }

  def close(): Unit = {
    positions.foreach((_, tensor) => ops.release(tensor))
    weights.release()
    source.close()
  }
}

object Pid {

  /** The official image RoPE's angles (`precompute_freqs_cis_2d_ntk`): per
    * token of a `height × width` grid, `head / 2` pairs alternating column and
    * row, positions `linspace(0, 16)` on each axis, each axis's base
    * `10000 × (grid / reference)^(a / (a − 2))` with `a = head / 2`.
    */
  def imageAngles(
      head: Int,
      height: Int,
      width: Int,
      reference: Int
  ): Array[Array[Double]] = {
    val axis = head / 2
    def theta(size: Int) =
      10000 * (if (axis > 2)
                 math.pow(size.toDouble / reference, axis.toDouble / (axis - 2))
               else 1.0)
    val (thetaH, thetaW) = (theta(height), theta(width))
    def linspace(n: Int, i: Int) = if (n == 1) 0.0 else 16.0 * i / (n - 1)
    Array.tabulate(height * width) { token =>
      val (x, y) =
        (linspace(width, token % width), linspace(height, token / width))
      Array.tabulate(head / 2) { m =>
        val k = m / 2
        if (m % 2 == 0) x * math.pow(thetaW, -4.0 * k / head)
        else y * math.pow(thetaH, -4.0 * k / head)
      }
    }
  }

  def open(ops: Ops, path: Path, ropeReference: Int = 2048): Pid = {
    val source = WeightSource.open(ops, path)
    try new Pid(ops, source, ropeReference)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
