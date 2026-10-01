package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** MiniMax H3's video VAE decoder (diffusers' `AutoencoderKLMiniMaxH3`,
  * sd-cpp's `minimax_h3_vae.hpp`): a ViT over every latent voxel plus learned
  * register tokens and a zero token, full attention with head RMS norms (no
  * weights) and a rotate-half RoPE over the three axes normalized to [−1, 1),
  * layer-scaled residuals, a SwiGLU MLP (`w1` holds [gate; value]); each token
  * expands to 4 frames of 16 × 16 pixels. The released recipe decodes in
  * temporal chunks of 7 latent frames (5 new, 2 of overlap, their frames
  * cross-faded) and spatial tiles of `tilePixels` overlapping by at least
  * `tileOverlap`, blended; the frames are ImageNet-normalized RGB, brought back
  * to [0, 1].
  *   - The latents come normalized: their statistics, `post_quant_conv` and the
  *     embedding fold into one affine at load, its inputs padded to 32 channels
  *     for the kernels.
  *   - The fused qkv holds each head's q, k and v in turn; they are regrouped
  *     at load.
  */
final class MiniMaxH3Vae private (
    ops: Ops,
    source: WeightSource,
    tilePixels: Int,
    tileOverlap: Int
) extends AutoCloseable {

  private val weights = new HybridWeights(ops, source, gguf = true)
  private def dimensions(name: String) =
    source.shape(name).dimensions.map(_.toInt)

  private val Seq(width, latentChannels) = dimensions(
    "decoder.x_embedder.weight"
  )
  private val HeadDimension = 64
  private val heads = width / HeadDimension
  private val blockCount = Iterator
    .from(0)
    .takeWhile(i => source.has(s"decoder.transformer_blocks.$i.norm1.weight"))
    .size
  private val intermediate =
    dimensions("decoder.transformer_blocks.0.ff.w1.weight").head / 2
  private val registers = dimensions("decoder.register_tokens")(1)
  private val outputs = dimensions("decoder.proj_out.weight").head
  private val Epsilon = 1e-5f

  /** The pixels a token becomes: 3 channels × 4 frames × 16 × 16. */
  val Frames = 4
  val Patch = 16
  require(outputs == 3 * Frames * Patch * Patch, s"proj_out gives $outputs")

  /** Latent channels as the embedding reads them: whole blocks of 32. */
  private val paddedChannels = (latentChannels + 31) / 32 * 32

  /** The embedding of normalized latents: `x_embedder(post_quant_conv(z × std +
    * mean))` as one affine, `[width, paddedChannels]` and `[width]`.
    */
  private val (embedding, embeddingBias) = {
    val c = latentChannels
    val mean = weights.hostFloats("latents_mean")
    val std = weights.hostFloats("latents_std")
    val post = weights.hostFloats("post_quant_conv.weight") // [c, c]
    val postBias = weights.hostFloats("post_quant_conv.bias")
    val embed = weights.hostFloats("decoder.x_embedder.weight") // [width, c]
    val embedBias = weights.hostFloats("decoder.x_embedder.bias")
    // A = E · P, then z' = z × std + mean: A · diag(std), bias A · mean + E · b + e
    val composed = Array.tabulate(width * c) { i =>
      val (row, column) = (i / c, i % c)
      (0 until c)
        .map(j => embed(row * c + j).toDouble * post(j * c + column))
        .sum
    }
    val matrix = new Array[Float](width * paddedChannels)
    val bias = new Array[Float](width)
    (0 until width).foreach { row =>
      var sum = embedBias(row).toDouble
      (0 until c).foreach { j =>
        sum += embed(row * c + j).toDouble * postBias(j)
        sum += composed(row * c + j) * mean(j)
        matrix(row * paddedChannels + j) =
          (composed(row * c + j) * std(j)).toFloat
      }
      bias(row) = sum.toFloat
    }
    (
      weights.keep(ops.fromFloats(Shape.of(width, paddedChannels), matrix)),
      weights.keep(ops.fromFloats(Shape.of(width), bias))
    )
  }

  final private class Block(prefix: String) {
    val norm1: Tensor = floats(s"$prefix.norm1.weight", width)
    val norm2: Tensor = floats(s"$prefix.norm2.weight", width)
    val scale1: Tensor = floats(s"$prefix.scale1", width)
    val scale2: Tensor = floats(s"$prefix.scale2", width)
    // per head [q; k; v] → [q of every head; k …; v …]
    private val qkv = {
      val name = s"$prefix.attn.to_qkv.weight"
      val stored = source(name)
      val rowBytes = stored.dtype.byteSize(width.toLong).toInt
      val bytes = weights.hostBytes(name)
      val regrouped = new Array[Byte](bytes.length)
      for {
        part <- 0 until 3
        head <- 0 until heads
        i <- 0 until HeadDimension
      } System.arraycopy(
        bytes,
        ((head * 3 + part) * HeadDimension + i) * rowBytes,
        regrouped,
        ((part * heads + head) * HeadDimension + i) * rowBytes,
        rowBytes
      )
      weights.upload(stored.dtype, stored.shape, regrouped)
    }
    private val qkvBias = {
      val stored = weights.hostFloats(s"$prefix.attn.to_qkv.bias")
      val regrouped = new Array[Float](stored.length)
      for {
        part <- 0 until 3
        head <- 0 until heads
        i <- 0 until HeadDimension
      } regrouped((part * heads + head) * HeadDimension + i) = stored(
        (head * 3 + part) * HeadDimension + i
      )
      weights.keep(ops.fromFloats(Shape.of(3L * width), regrouped))
    }
    val q: Tensor = qkv.rows(0, width)
    val k: Tensor = qkv.rows(width, width)
    val v: Tensor = qkv.rows(2L * width, width)
    val qBias: Tensor = qkvBias.view(3, width).rows(0, 1).view(width)
    val kBias: Tensor = qkvBias.view(3, width).rows(1, 1).view(width)
    val vBias: Tensor = qkvBias.view(3, width).rows(2, 1).view(width)
    val o: Tensor = source(s"$prefix.attn.to_out.weight")
    val oBias: Tensor = floats(s"$prefix.attn.to_out.bias", width)
    private val w1 = source(s"$prefix.ff.w1.weight")
    private val w1Bias = floats(s"$prefix.ff.w1.bias", 2L * intermediate)
    val gate: Tensor = w1.rows(0, intermediate)
    val value: Tensor = w1.rows(intermediate, intermediate)
    val gateBias: Tensor =
      w1Bias.view(2, intermediate).rows(0, 1).view(intermediate)
    val valueBias: Tensor =
      w1Bias.view(2, intermediate).rows(1, 1).view(intermediate)
    val down: Tensor = source(s"$prefix.ff.w2.weight")
    val downBias: Tensor = floats(s"$prefix.ff.w2.bias", width)
  }

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  private val blocks =
    (0 until blockCount).map(i => new Block(s"decoder.transformer_blocks.$i"))
  private val registerTokens = weights.floats(
    "decoder.register_tokens",
    "decoder.register_tokens",
    Shape.of(registers, width)
  )
  private val finalNormWeight = floats("decoder.norm_out.weight", width)
  private val finalNormBias = floats("decoder.norm_out.bias", width)
  private val out = source("decoder.proj_out.weight")
  private val outBias = floats("decoder.proj_out.bias", outputs)
  private val ones =
    weights.keep(
      ops.fromFloats(Shape.of(HeadDimension), Array.fill(HeadDimension)(1f))
    )

  /** RoPE: 8 frequencies per axis over 48 of each head's 64 values. */
  private val RopeFrequencies = HeadDimension * 3 / 4 / 6
  private val inverseFrequencies = Array.tabulate(RopeFrequencies)(i =>
    1 / math.pow(100, i * 6.0 / (HeadDimension * 3 / 4))
  )
  private val attention = Attention(
    (1 / math.sqrt(HeadDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  /** Buffers for one tile of `frames × height × width` latents. */
  final private class Buffers(
      val frames: Int,
      val height: Int,
      val widthLatents: Int
  ) {
    val patches: Int = frames * height * widthLatents
    val tokens: Int = patches + registers + 1
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(elements: Long): Tensor = {
      val tensor = ops.allocate(DType.F32, Shape.of(elements))
      held += tensor
      tensor
    }
    private val rows = tokens.toLong
    val input: Tensor = flat(patches.toLong * paddedChannels)
    val x: Tensor = flat(rows * width)
    val normed: Tensor = flat(rows * width)
    val q: Tensor = flat(rows * width)
    val k: Tensor = flat(rows * width)
    val v: Tensor = flat(rows * width)
    val rotatedQueries: Tensor = flat(rows * width)
    val rotatedKeys: Tensor = flat(rows * width)
    val attended: Tensor = flat(rows * width)
    val projected: Tensor = flat(rows * width)
    val gate: Tensor = flat(rows * intermediate)
    val value: Tensor = flat(rows * intermediate)
    val pixels: Tensor = flat(patches.toLong * outputs)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, heads, HeadDimension)
    val pageTable: Tensor = {
      val tensor = ops.fromInts(Shape.of(1), Array(0))
      held += tensor
      tensor
    }
    // the rotary angles: the voxels' (t, h, w), each in [−1, 1), then the
    // registers and the zero token at 0
    private val pairs = 3 * RopeFrequencies
    private val angles = Array.tabulate(tokens * pairs) { i =>
      val (token, pair) = (i / pairs, i % pairs)
      if (token >= patches) 0.0
      else {
        val (axis, frequency) = (pair / RopeFrequencies, pair % RopeFrequencies)
        val (index, size) = axis match {
          case 0 => (token / (height * widthLatents), frames)
          case 1 => (token / widthLatents % height, height)
          case _ => (token % widthLatents, widthLatents)
        }
        val position = 2 * ((index + 0.5) / size) - 1
        2 * math.Pi * position * inverseFrequencies(frequency)
      }
    }
    val cosines: Tensor = {
      val tensor = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.cos(a).toFloat)
      )
      held += tensor
      tensor
    }
    val sines: Tensor = {
      val tensor = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.sin(a).toFloat)
      )
      held += tensor
      tensor
    }
    def release(): Unit = {
      held.foreach(ops.release)
      ops.release(cache.keys)
      ops.release(cache.values)
    }
  }

  private var buffers = Option.empty[Buffers]
  private def buffersFor(frames: Int, height: Int, widthLatents: Int): Buffers =
    buffers
      .filter(b =>
        b.frames == frames && b.height == height && b.widthLatents == widthLatents
      )
      .getOrElse {
        buffers.foreach(_.release())
        val created = new Buffers(frames, height, widthLatents)
        buffers = Some(created)
        created
      }

  private def affine(x: Tensor, weight: Tensor, bias: Tensor, out: Tensor) = {
    ops.linear(x, weight, out)
    ops.addRow(out, bias, out)
  }

  /** One tile's pixels: `latents` (`[frames × h × w, latentChannels]`,
    * normalized) through the ViT, `[frames × 4, h × 16, w × 16, 3]` in the
    * decoder's space (ImageNet-normalized).
    */
  private def decodeTile(
      latents: Array[Float],
      frames: Int,
      height: Int,
      widthLatents: Int
  ): Array[Float] = {
    val w = buffersFor(frames, height, widthLatents)
    val (tokens, patches) = (w.tokens.toLong, w.patches)
    val padded = new Array[Float](patches * paddedChannels)
    (0 until patches).foreach(p =>
      System.arraycopy(
        latents,
        p * latentChannels,
        padded,
        p * paddedChannels,
        latentChannels
      )
    )
    val uploaded = ops.fromFloats(Shape.of(patches, paddedChannels), padded)
    val x = w.x.prefix(tokens, width)
    try affine(uploaded, embedding, embeddingBias, x.rows(0, patches))
    finally ops.release(uploaded)
    ops.copy(registerTokens, x.rows(patches, registers))
    ops.zero(x.rows(patches.toLong + registers, 1))
    val (h, d) = (heads.toLong, HeadDimension.toLong)
    val normed = w.normed.prefix(tokens, width)
    blocks.foreach { b =>
      ops.rmsNorm(x, b.norm1, Epsilon, 0f, normed)
      val (q, k, v) = (
        w.q.prefix(tokens, width),
        w.k.prefix(tokens, width),
        w.v.prefix(tokens, width)
      )
      ops.linears(normed, Seq(b.q, b.k, b.v), Seq(q, k, v))
      ops.addRow(q, b.qBias, q)
      ops.addRow(k, b.kBias, k)
      ops.addRow(v, b.vBias, v)
      ops.rmsNorm(
        q.view(tokens * h, d),
        ones,
        Epsilon,
        0f,
        q.view(tokens * h, d)
      )
      ops.rmsNorm(
        k.view(tokens * h, d),
        ones,
        Epsilon,
        0f,
        k.view(tokens * h, d)
      )
      val (rq, rk) = (
        w.rotatedQueries.prefix(tokens, h, d),
        w.rotatedKeys.prefix(tokens, h, d)
      )
      ops.ropeTable(q.view(tokens, h, d), w.cosines, w.sines, rq, halves = true)
      ops.ropeTable(k.view(tokens, h, d), w.cosines, w.sines, rk, halves = true)
      ops.cacheWrite(rk, v.view(tokens, h, d), w.cache, w.pageTable, 0)
      val attended = w.attended.prefix(tokens, h, d)
      ops.attention(
        rq,
        w.cache,
        w.pageTable,
        0,
        tokens.toInt,
        attention,
        attended
      )
      val projected = w.projected.prefix(tokens, width)
      affine(attended.view(tokens, width), b.o, b.oBias, projected)
      ops.gatedAdd(x, projected, b.scale1)
      ops.rmsNorm(x, b.norm2, Epsilon, 0f, normed)
      val (gate, value) = (
        w.gate.prefix(tokens, intermediate),
        w.value.prefix(tokens, intermediate)
      )
      ops.linears(normed, Seq(b.gate, b.value), Seq(gate, value))
      ops.addRow(gate, b.gateBias, gate)
      ops.addRow(value, b.valueBias, value)
      ops.gated(Activation.Silu, gate, value, gate)
      affine(gate, b.down, b.downBias, projected)
      ops.gatedAdd(x, projected, b.scale2)
    }
    val voxels = normed.rows(0, patches)
    ops.layerNorm(
      x.rows(0, patches),
      Some(finalNormWeight),
      Some(finalNormBias),
      Epsilon,
      voxels
    )
    val pixels = w.pixels.prefix(patches, outputs)
    affine(voxels, out, outBias, pixels)
    val values = ops.toFloats(pixels)
    // token (t, y, x), feature (c, pt, py, px) → pixel (t × 4 + pt, y × 16 +
    // py, x × 16 + px, c)
    val (pixelHeight, pixelWidth) = (height * Patch, widthLatents * Patch)
    val result =
      new Array[Float](frames * Frames * pixelHeight * pixelWidth * 3)
    var token = 0
    while (token < patches) {
      val (t, y, x0) = (
        token / (height * widthLatents),
        token / widthLatents % height,
        token % widthLatents
      )
      var feature = 0
      while (feature < outputs) {
        val channel = feature / (Frames * Patch * Patch)
        val pt = feature / (Patch * Patch) % Frames
        val py = feature / Patch % Patch
        val px = feature % Patch
        val frame = t * Frames + pt
        val at =
          ((frame * pixelHeight + y * Patch + py) * pixelWidth + x0 * Patch + px) * 3 + channel
        result(at) = values(token * outputs + feature)
        feature += 1
      }
      token += 1
    }
    result
  }

  /** One temporal clip (`[frames, h, w, C]` normalized latents), spatially
    * tiled and blended (diffusers' `_decode_clip` and `_stitch_tiles`): `[4 ×
    * frames, 16h, 16w, 3]` in the decoder's space.
    */
  private def decodeClip(
      latents: Array[Float],
      frames: Int,
      height: Int,
      widthLatents: Int
  ): Array[Float] = {
    val (pixelHeight, pixelWidth) = (height * Patch, widthLatents * Patch)
    val (rowStarts, tileHeight, rowOverlaps) =
      MiniMaxH3Vae.split(pixelHeight, tilePixels, tileOverlap)
    val (columnStarts, tileWidth, columnOverlaps) =
      MiniMaxH3Vae.split(pixelWidth, tilePixels, tileOverlap)
    val (th, tw) = (tileHeight / Patch, tileWidth / Patch)
    val tiles = rowStarts.map { top =>
      columnStarts.map { left =>
        val (y0, x0) = (top / Patch, left / Patch)
        val tile = new Array[Float](frames * th * tw * latentChannels)
        for {
          t <- 0 until frames
          y <- 0 until th
        } System.arraycopy(
          latents,
          ((t * height + y0 + y) * widthLatents + x0) * latentChannels,
          tile,
          ((t * th + y) * tw) * latentChannels,
          tw * latentChannels
        )
        decodeTile(tile, frames, th, tw)
      }
    }
    MiniMaxH3Vae.stitch(
      tiles,
      frames * Frames,
      tileHeight,
      tileWidth,
      rowOverlaps,
      columnOverlaps,
      pixelHeight,
      pixelWidth,
      3
    )
  }

  /** Decodes normalized `latents` (`[frames, h, w, C]`) into video frames, each
    * `[16h, 16w, 3]` RGB in [0, 1], handed to `frame` in order (diffusers'
    * `_decode`: chunks of 5 latent frames read with the next 2, the 3 frames
    * before each chunk's first dropped, its last 5 cross-faded with the next
    * chunk's first; the tail padded by repeating the last latent frame, the
    * frames that makes cut off). `progress(done, chunks)` after each chunk.
    */
  def decode(
      latents: Array[Float],
      frames: Int,
      height: Int,
      widthLatents: Int,
      frame: Array[Float] => Unit,
      progress: (Int, Int) => Unit = (_, _) => ()
  ): Unit = {
    require(
      latents.length == frames * height * widthLatents * latentChannels,
      s"${latents.length} latent values for $frames × $height × $widthLatents × $latentChannels"
    )
    val chunkLatents = 5
    val tokenDrop = 3
    val tokenOverlap = 2
    val framePrePadding = 3
    val frameOverlap = 5
    val chunkFrames = chunkLatents * Frames
    val tokens = frames + tokenDrop
    val pad = (chunkLatents - tokens % chunkLatents) % chunkLatents
    val chunks = (tokens + pad) / chunkLatents - 1
    require(chunks >= 1, s"$frames latent frames are too few to decode")
    val voxel = height * widthLatents * latentChannels
    val padded = latents ++ Array.tabulate(pad * voxel)(i =>
      latents((frames - 1) * voxel + i % voxel)
    )
    val (pixelHeight, pixelWidth) = (height * Patch, widthLatents * Patch)
    val frameValues = pixelHeight * pixelWidth * 3
    // the frames the padding made, cut off at the end
    val tail = {
      val intraTail = 17 % Frames
      (0 until pad).map { k =>
        if (intraTail != 0 && (frames + k) % chunkLatents == 0) intraTail
        else Frames
      }.sum
    }
    val total = {
      val clipFrames = chunkFrames - framePrePadding
      chunks * clipFrames + frameOverlap - tail
    }
    var emitted = 0
    def emit(values: Array[Float], index: Int): Unit =
      if (emitted < total) {
        val rgb = new Array[Float](frameValues)
        val base = index * frameValues
        var i = 0
        while (i < frameValues) {
          val c = i % 3
          val value = values(base + i) * MiniMaxH3Vae.PixelStd(c) +
            MiniMaxH3Vae.PixelMean(c)
          rgb(i) = math.max(0f, math.min(1f, value))
          i += 1
        }
        frame(rgb)
        emitted += 1
      }
    var overlap = Option.empty[Array[Float]]
    (0 until chunks).foreach { i =>
      val start = i * chunkLatents
      val count = chunkLatents + tokenOverlap
      val clip = decodeClip(
        java.util.Arrays
          .copyOfRange(padded, start * voxel, (start + count) * voxel),
        count,
        height,
        widthLatents
      )
      // frames [3, 20) of the clip, the first 5 cross-faded with the overlap
      val first = java.util.Arrays.copyOfRange(
        clip,
        framePrePadding * frameValues,
        chunkFrames * frameValues
      )
      overlap.foreach { previous =>
        (0 until frameOverlap).foreach { f =>
          val weight = f.toFloat / frameOverlap
          var v = 0
          while (v < frameValues) {
            val at = f * frameValues + v
            first(at) = previous(at) * (1 - weight) + first(at) * weight
            v += 1
          }
        }
      }
      (0 until chunkFrames - framePrePadding).foreach(emit(first, _))
      // frames [23, 28): the overlap with the next chunk
      overlap = Some(
        java.util.Arrays.copyOfRange(
          clip,
          (chunkFrames + framePrePadding) * frameValues,
          (chunkFrames + framePrePadding + frameOverlap) * frameValues
        )
      )
      progress(i + 1, chunks)
    }
    overlap.foreach(last => (0 until frameOverlap).foreach(emit(last, _)))
  }

  /** How many frames `decode` gives for `frames` latent frames. */
  def frameCount(frames: Int): Int = {
    val tokens = frames + 3
    val pad = (5 - tokens % 5) % 5
    val chunks = (tokens + pad) / 5 - 1
    val tail = (0 until pad).map { k =>
      if ((frames + k) % 5 == 0) 17 % Frames else Frames
    }.sum
    chunks * 17 + 5 - tail
  }

  def close(): Unit = {
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object MiniMaxH3Vae {

  /** ImageNet's normalization, which the decoder's pixels carry. */
  val PixelMean: Array[Float] = Array(0.485f, 0.456f, 0.406f)
  val PixelStd: Array[Float] = Array(0.229f, 0.224f, 0.225f)

  /** Tiles over `length` pixels (diffusers' `_split_tiles`): starts, the tile
    * length, and the overlaps, every boundary on the 16-pixel latent grid.
    */
  def split(
      length: Int,
      tilePixels: Int,
      tileOverlap: Int
  ): (Seq[Int], Int, Seq[Int]) =
    if (tilePixels >= length) (Seq(0), length, Nil)
    else {
      val step = 16
      var count = (length + tilePixels - 1) / tilePixels
      while (tilePixels * count - tileOverlap * (count - 1) - length < 0)
        count += 1
      val overlaps = Array.fill(count - 1)(tileOverlap)
      val remaining = tilePixels * count - overlaps.sum - length
      (0 until remaining / step).foreach(i => overlaps(i % (count - 1)) += step)
      val starts =
        overlaps.scanLeft(0)((start, overlap) => start + tilePixels - overlap)
      (starts.toSeq, tilePixels, overlaps.toSeq)
    }

  /** Tiles (each `[frames, tileHeight, tileWidth, channels]`) into one
    * `[frames, height, width, channels]` (diffusers' `_stitch_tiles`): each
    * blended with the one above and the one to its left (both as made), over
    * the overlap between them, then cut by its own overlaps below and to the
    * right.
    */
  def stitch(
      tiles: Seq[Seq[Array[Float]]],
      frames: Int,
      tileHeight: Int,
      tileWidth: Int,
      rowOverlaps: Seq[Int],
      columnOverlaps: Seq[Int],
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    val result = new Array[Float](frames * height * width * channels)
    var top = 0
    tiles.indices.foreach { i =>
      var left = 0
      val keptHeight =
        if (i < tiles.size - 1) tileHeight - rowOverlaps(i) else tileHeight
      tiles(i).indices.foreach { j =>
        val keptWidth =
          if (j < tiles(i).size - 1) tileWidth - columnOverlaps(j)
          else tileWidth
        val tile = tiles(i)(j).clone()
        def at(frame: Int, y: Int, x: Int) =
          ((frame * tileHeight + y) * tileWidth + x) * channels
        if (i > 0) {
          val above = tiles(i - 1)(j)
          val extent = math.min(rowOverlaps(i - 1), tileHeight)
          for {
            f <- 0 until frames
            y <- 0 until extent
            x <- 0 until tileWidth
            c <- 0 until channels
          } {
            val weight = y.toFloat / extent
            tile(at(f, y, x) + c) =
              above(at(f, tileHeight - extent + y, x) + c) *
                (1 - weight) + tile(at(f, y, x) + c) * weight
          }
        }
        if (j > 0) {
          val leftTile = tiles(i)(j - 1)
          val extent = math.min(columnOverlaps(j - 1), tileWidth)
          for {
            f <- 0 until frames
            y <- 0 until tileHeight
            x <- 0 until extent
            c <- 0 until channels
          } {
            val weight = x.toFloat / extent
            tile(at(f, y, x) + c) =
              leftTile(at(f, y, tileWidth - extent + x) + c) *
                (1 - weight) + tile(at(f, y, x) + c) * weight
          }
        }
        for {
          f <- 0 until frames
          y <- 0 until keptHeight
        } System.arraycopy(
          tile,
          at(f, y, 0),
          result,
          ((f * height + top + y) * width + left) * channels,
          keptWidth * channels
        )
        left += keptWidth
      }
      top += keptHeight
    }
    result
  }

  def holds(source: WeightSource): Boolean =
    source.has("decoder.x_embedder.weight") && source.has(
      "decoder.register_tokens"
    )

  /** The decoder of `path`, tiled as released (256-pixel tiles overlapping by
    * at least 64) unless told otherwise.
    */
  def open(
      ops: Ops,
      path: Path,
      tilePixels: Int = 256,
      tileOverlap: Int = 64
  ): MiniMaxH3Vae = {
    val source = WeightSource.open(ops, path)
    try {
      if (!holds(source))
        throw new FormatException(s"$path is no MiniMax H3 video VAE")
      new MiniMaxH3Vae(ops, source, tilePixels, tileOverlap)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
