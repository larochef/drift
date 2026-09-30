package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** A Wan transformer's sizes, read off its weights: `hidden` wide (heads of
  * 128, the width every Wan has, which the files do not record), `blocks` deep,
  * a GELU MLP of `intermediate`, `inChannels` latent channels in (16 for text
  * to video, 36 for Wan 2.2 I2V: the noise, then a mask of 4 and the
  * conditioning latents of 16), `outChannels` out, text of `textWidth`.
  */
final case class WanConfig(
    hidden: Int,
    blocks: Int,
    intermediate: Int,
    inChannels: Int,
    outChannels: Int,
    textWidth: Int,
    frequencies: Int
) {
  val headDimension: Int = 128
  def heads: Int = hidden / headDimension

  /** Each patch's input features (channel-major 2 × 2), and as the kernels take
    * them (whole blocks of 32).
    */
  def patchWidth: Int = 4 * inChannels
  def paddedPatchWidth: Int = (patchWidth + 31) / 32 * 32

  /** RoPE's pairs per axis (frame, row, column): sd-cpp's and diffusers' `d −
    * 4⌊d/6⌋`, `2⌊d/6⌋`, `2⌊d/6⌋` values, halved.
    */
  def ropePairs: Seq[Int] = {
    val side = 2 * (headDimension / 6)
    Seq((headDimension - 2 * side) / 2, side / 2, side / 2)
  }
}

object WanConfig {
  def holds(source: WeightSource): Boolean =
    source.has("patch_embedding.weight") && source.has(
      "blocks.0.cross_attn.q.weight"
    )

  def of(source: WeightSource): WanConfig = {
    def dimensions(name: String) = source.shape(name).dimensions.map(_.toInt)
    WanConfig(
      hidden = dimensions("patch_embedding.weight").head,
      blocks = Iterator
        .from(0)
        .takeWhile(i => source.has(s"blocks.$i.self_attn.q.weight"))
        .size,
      intermediate = dimensions("blocks.0.ffn.0.weight").head,
      inChannels = dimensions("patch_embedding.weight")(1),
      outChannels = dimensions("head.head.weight").head / 4,
      textWidth = dimensions("text_embedding.0.weight").last,
      frequencies = dimensions("time_embedding.0.weight").last
    )
  }
}

/** A prompt as one Wan transformer reads it: each block's cross-attention keys
  * and values over the projected text. Made by `Wan.text`, released by
  * `release`.
  */
final class WanText private[models] (
    val tokens: Int,
    val caches: Seq[KvCache],
    ops: Ops
) {
  def release(): Unit = caches.foreach { cache =>
    ops.release(cache.keys)
    ops.release(cache.values)
  }
}

/** Wan's video transformer (diffusers' `WanTransformer3DModel`, sd-cpp's
  * `wan.hpp`; Wan 2.1 and 2.2, text to video and I2V), from the original names
  * (the GGUFs', ComfyUI's). The latents come as rows of 2 × 2 patches, each
  * channel-major (`packPatches`' layout), frame by frame; the I2V checkpoints'
  * conditioning rows (mask then latents) follow the noise's in each row.
  *   - Blocks: self-attention with RMS norms across the heads and an
  *     interleaved 3-axis RoPE (frame, row, column), cross-attention to the
  *     text (its keys and values made once per prompt, `text`), a GELU (tanh)
  *     MLP; the self-attention and the MLP modulated by the timestep's six
  *     vectors plus the block's table, the cross-attention's input normed with
  *     weights.
  *   - The head modulated by the time embedding itself plus its table; its rows
  *     permuted at load so the output rows are channel-major too.
  * Timesteps are σ × 1000; the velocity is `noise − data`.
  */
final class Wan private (ops: Ops, source: WeightSource) extends AutoCloseable {

  val config: WanConfig = WanConfig.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val Epsilon = 1e-6f

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  final private case class Affine(weight: Tensor, bias: Tensor)
  private def affineLayer(prefix: String, outputs: Long): Affine =
    Affine(source(s"$prefix.weight"), floats(s"$prefix.bias", outputs))

  final private class Attention(prefix: String) {
    val q: Affine = affineLayer(s"$prefix.q", c.hidden)
    val k: Affine = affineLayer(s"$prefix.k", c.hidden)
    val v: Affine = affineLayer(s"$prefix.v", c.hidden)
    val o: Affine = affineLayer(s"$prefix.o", c.hidden)
    val normQ: Tensor = floats(s"$prefix.norm_q.weight", c.hidden)
    val normK: Tensor = floats(s"$prefix.norm_k.weight", c.hidden)
  }

  final private class Block(i: Int) {
    val self = new Attention(s"blocks.$i.self_attn")
    val cross = new Attention(s"blocks.$i.cross_attn")
    val crossNormWeight: Tensor = floats(s"blocks.$i.norm3.weight", c.hidden)
    val crossNormBias: Tensor = floats(s"blocks.$i.norm3.bias", c.hidden)
    val up: Affine = affineLayer(s"blocks.$i.ffn.0", c.intermediate)
    val down: Affine = affineLayer(s"blocks.$i.ffn.2", c.hidden)
    val modulation: Tensor = weights.floats(
      s"blocks.$i.modulation",
      s"blocks.$i.modulation",
      Shape.of(6, c.hidden)
    )
  }

  /** The patch embedding as a linear over padded patch rows. */
  private val patchIn: Affine = {
    val stored = weights.hostFloats("patch_embedding.weight")
    val (width, padded) = (c.patchWidth, c.paddedPatchWidth)
    val matrix = new Array[Float](c.hidden * padded)
    (0 until c.hidden).foreach(row =>
      System.arraycopy(stored, row * width, matrix, row * padded, width)
    )
    Affine(
      weights.keep(ops.fromFloats(Shape.of(c.hidden, padded), matrix)),
      floats("patch_embedding.bias", c.hidden)
    )
  }
  private val textIn = affineLayer("text_embedding.0", c.hidden)
  private val textOut = affineLayer("text_embedding.2", c.hidden)
  private val timeIn = affineLayer("time_embedding.0", c.hidden)
  private val timeOut = affineLayer("time_embedding.2", c.hidden)
  private val timeProjection = affineLayer("time_projection.1", 6L * c.hidden)
  private val blocks = (0 until c.blocks).map(new Block(_))
  private val headModulation =
    weights.floats("head.modulation", "head.modulation", Shape.of(2, c.hidden))

  /** The head, its outputs reordered from (row, column, channel) to channel
    * major.
    */
  private val head: Affine = {
    val out = 4 * c.outChannels
    val stored = weights.hostFloats("head.head.weight")
    val bias = weights.hostFloats("head.head.bias")
    def from(row: Int) = (row % 4) * c.outChannels + row / 4
    Affine(
      weights.keep(
        ops.fromFloats(
          Shape.of(out, c.hidden),
          Array.tabulate(out * c.hidden)(i =>
            stored(from(i / c.hidden) * c.hidden + i % c.hidden)
          )
        )
      ),
      weights.keep(
        ops.fromFloats(Shape.of(out), Array.tabulate(out)(i => bias(from(i))))
      )
    )
  }

  private val attention = drift.runner.ops.Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  /** `prompt` (the text encoder's `[L, textWidth]`, zero rows included as the
    * pipeline pads it) projected, and each block's cross-attention keys and
    * values over it.
    */
  def text(prompt: Tensor): WanText = {
    val tokens = prompt.shape.dimensions.head
    require(
      prompt.shape == Shape.of(tokens, c.textWidth),
      s"text: ${prompt.shape}"
    )
    val held = mutable.ArrayBuffer.empty[Tensor]
    def allocate() = {
      val tensor = ops.allocate(DType.F32, Shape.of(tokens, c.hidden.toLong))
      held += tensor
      tensor
    }
    try {
      val (inner, projected, k, v) =
        (allocate(), allocate(), allocate(), allocate())
      affine(prompt, textIn, inner)
      ops.activation(Activation.GeluTanh, inner, inner)
      affine(inner, textOut, projected)
      val pageTable = ops.fromInts(Shape.of(1), Array(0))
      held += pageTable
      val caches = blocks.map { b =>
        affine(projected, b.cross.k, k)
        ops.rmsNorm(k, b.cross.normK, Epsilon, 0f, k)
        affine(projected, b.cross.v, v)
        val cache = ops.allocateCache(
          1,
          (tokens.toInt + 15) / 16 * 16,
          c.heads,
          c.headDimension
        )
        ops.cacheWrite(
          k.view(tokens, c.heads, c.headDimension),
          v.view(tokens, c.heads, c.headDimension),
          cache,
          pageTable,
          0
        )
        cache
      }
      new WanText(tokens.toInt, caches, ops)
    } finally held.foreach(ops.release)
  }

  /** Buffers for `tokens` rows, grown when too small. */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(elements: Long): Tensor = {
      val tensor = ops.allocate(DType.F32, Shape.of(elements))
      held += tensor
      tensor
    }
    private val rows = tokens.toLong
    val input: Tensor = flat(rows * c.paddedPatchWidth)
    val x: Tensor = flat(rows * c.hidden)
    val normed: Tensor = flat(rows * c.hidden)
    val q: Tensor = flat(rows * c.hidden)
    val k: Tensor = flat(rows * c.hidden)
    val v: Tensor = flat(rows * c.hidden)
    val rotatedQueries: Tensor = flat(rows * c.hidden)
    val rotatedKeys: Tensor = flat(rows * c.hidden)
    val attended: Tensor = flat(rows * c.hidden)
    val projected: Tensor = flat(rows * c.hidden)
    val up: Tensor = flat(rows * c.intermediate)
    val cosines: Tensor = flat(rows * c.headDimension / 2)
    val sines: Tensor = flat(rows * c.headDimension / 2)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, c.heads, c.headDimension)
    val pageTable: Tensor = {
      val tensor = ops.fromInts(Shape.of(1), Array(0))
      held += tensor
      tensor
    }
    var grid: (Int, Int, Int) = (-1, -1, -1)
    def release(): Unit = {
      held.foreach(ops.release)
      ops.release(cache.keys)
      ops.release(cache.values)
    }
  }

  private var buffers = Option.empty[Buffers]
  private def buffersFor(tokens: Int): Buffers =
    buffers.filter(_.tokens >= tokens).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens)
      buffers = Some(created)
      created
    }

  /** The rotary angles of a `frames × rows × columns` grid, into the buffers'
    * tables (kept while the grid is the same).
    */
  private def angles(w: Buffers, frames: Int, rows: Int, columns: Int): Unit =
    if (w.grid != (frames, rows, columns)) {
      val tokens = frames * rows * columns
      val pairs = c.headDimension / 2
      val axes = c.ropePairs
      val theta = 10000.0
      val table = Array.tabulate(tokens * pairs) { i =>
        val (token, pair) = (i / pairs, i % pairs)
        val (axis, index) =
          if (pair < axes(0)) (0, pair)
          else if (pair < axes(0) + axes(1)) (1, pair - axes(0))
          else (2, pair - axes(0) - axes(1))
        val position = axis match {
          case 0 => token / (rows * columns)
          case 1 => token / columns % rows
          case _ => token % columns
        }
        position / math.pow(theta, 2.0 * index / (2 * axes(axis)))
      }
      val (cos, sin) = (
        ops.fromFloats(
          Shape.of(tokens, pairs),
          table.map(a => math.cos(a).toFloat)
        ),
        ops.fromFloats(
          Shape.of(tokens, pairs),
          table.map(a => math.sin(a).toFloat)
        )
      )
      try {
        ops.copy(cos, w.cosines.prefix(tokens, pairs))
        ops.copy(sin, w.sines.prefix(tokens, pairs))
      } finally {
        ops.release(cos)
        ops.release(sin)
      }
      w.grid = (frames, rows, columns)
    }

  /** The velocity at `timestep` (σ × 1000) of `latents` (`[frames × rows ×
    * columns, 4 × outChannels]` patch rows), with the I2V checkpoints'
    * `condition` rows (`[same, 4 × (inChannels − outChannels)]`), given
    * `prompt`, into `out` (like `latents`).
    */
  def velocity(
      latents: Tensor,
      condition: Option[Tensor],
      prompt: WanText,
      timestep: Float,
      frames: Int,
      rows: Int,
      columns: Int,
      out: Tensor
  ): Unit = {
    val tokens = frames * rows * columns
    val f = c.hidden.toLong
    val latentWidth = 4L * c.outChannels
    require(
      latents.shape == Shape
        .of(tokens, latentWidth) && out.shape == latents.shape &&
        condition.forall(
          _.shape == Shape.of(tokens, c.patchWidth - latentWidth)
        ) &&
        condition.isDefined == (c.inChannels != c.outChannels),
      s"velocity: latents ${latents.shape}, condition ${condition.map(_.shape)} for $c"
    )
    val w = buffersFor(tokens)
    val small = mutable.ArrayBuffer.empty[Tensor]
    def vector(elements: Long) = {
      val tensor = ops.allocate(DType.F32, Shape.of(1, elements))
      small += tensor
      tensor
    }
    try {
      // the patch rows, padded, embedded
      val input = w.input.prefix(tokens, c.paddedPatchWidth)
      val padding = c.paddedPatchWidth - c.patchWidth
      val zeros = Option.when(padding > 0) {
        val tensor = ops.allocate(DType.F32, Shape.of(tokens, padding))
        small += tensor
        ops.zero(tensor)
        tensor
      }
      ops.concatColumns(Seq(latents) ++ condition ++ zeros, input)
      val x = w.x.prefix(tokens, f)
      affine(input, patchIn, x)
      // the timestep: a sinusoid (cosines first) through the time MLP
      val half = c.frequencies / 2
      val sinusoid = Array.tabulate(2 * half) { i =>
        val frequency = math.exp(-math.log(10000) * (i % half) / half)
        val angle = timestep * frequency
        (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
      }
      val embedded = ops.fromFloats(Shape.of(1, 2L * half), sinusoid)
      small += embedded
      val (inner, time, activated) = (vector(f), vector(f), vector(f))
      val projection = vector(6 * f)
      affine(embedded, timeIn, inner)
      ops.activation(Activation.Silu, inner, inner)
      affine(inner, timeOut, time)
      ops.activation(Activation.Silu, time, activated)
      affine(activated, timeProjection, projection)
      angles(w, frames, rows, columns)
      val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
      val (cosines, sines) =
        (w.cosines.prefix(tokens, d / 2), w.sines.prefix(tokens, d / 2))
      val modulation = vector(6 * f)
      def chunk(i: Int) = modulation.view(6, f).rows(i, 1).view(f)
      val normed = w.normed.prefix(tokens, f)
      val (q, k, v) =
        (w.q.prefix(tokens, f), w.k.prefix(tokens, f), w.v.prefix(tokens, f))
      val attended = w.attended.prefix(tokens, heads, d)
      val projected = w.projected.prefix(tokens, f)
      blocks.zip(prompt.caches).foreach { (b, textCache) =>
        ops.add(projection, b.modulation.view(1, 6 * f), modulation)
        // self-attention
        ops.layerNorm(x, None, None, Epsilon, normed)
        ops.modulate(normed, chunk(1), chunk(0), normed)
        affine(normed, b.self.q, q)
        affine(normed, b.self.k, k)
        affine(normed, b.self.v, v)
        ops.rmsNorm(q, b.self.normQ, Epsilon, 0f, q)
        ops.rmsNorm(k, b.self.normK, Epsilon, 0f, k)
        val (rq, rk) = (
          w.rotatedQueries.prefix(tokens, heads, d),
          w.rotatedKeys.prefix(tokens, heads, d)
        )
        ops.ropeTable(q.view(tokens, heads, d), cosines, sines, rq)
        ops.ropeTable(k.view(tokens, heads, d), cosines, sines, rk)
        ops.cacheWrite(rk, v.view(tokens, heads, d), w.cache, w.pageTable, 0)
        ops.attention(rq, w.cache, w.pageTable, 0, tokens, attention, attended)
        affine(attended.view(tokens, f), b.self.o, projected)
        ops.gatedAdd(x, projected, chunk(2))
        // cross-attention
        ops.layerNorm(
          x,
          Some(b.crossNormWeight),
          Some(b.crossNormBias),
          Epsilon,
          normed
        )
        affine(normed, b.cross.q, q)
        ops.rmsNorm(q, b.cross.normQ, Epsilon, 0f, q)
        ops.attention(
          q.view(tokens, heads, d),
          textCache,
          w.pageTable,
          0,
          prompt.tokens,
          attention,
          attended
        )
        affine(attended.view(tokens, f), b.cross.o, projected)
        ops.add(x, projected, x)
        // MLP
        ops.layerNorm(x, None, None, Epsilon, normed)
        ops.modulate(normed, chunk(4), chunk(3), normed)
        val up = w.up.prefix(tokens, c.intermediate)
        affine(normed, b.up, up)
        ops.activation(Activation.GeluTanh, up, up)
        affine(up, b.down, projected)
        ops.gatedAdd(x, projected, chunk(5))
      }
      // the head: shift and scale from the time embedding and the table
      val shiftScale = vector(2 * f)
      ops.addRow(headModulation.view(2, f), time.view(f), shiftScale.view(2, f))
      val both = shiftScale.view(2, f)
      ops.layerNorm(x, None, None, Epsilon, normed)
      ops.modulate(
        normed,
        both.rows(1, 1).view(f),
        both.rows(0, 1).view(f),
        normed
      )
      affine(normed, head, out)
    } finally small.foreach(ops.release)
  }

  def close(): Unit = {
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object Wan {

  def open(ops: Ops, path: Path): Wan = {
    val source = WeightSource.open(ops, path)
    try {
      if (!WanConfig.holds(source))
        throw new FormatException(
          s"$path is no Wan transformer (no patch_embedding.weight)"
        )
      new Wan(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
