package drift.runner.models

import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** MiniMax H3's transformer, read off its weights: `hidden` wide, `blocks`
  * deep, `heads` of `headDimension` (wider together than `hidden`), a SwiGLU of
  * `intermediate`, video rows of `videoWidth` (2 × 2 patches of the latents),
  * audio rows of `audioWidth`, text of `textWidth`, and RoPE over
  * `ropeFrequencies` per axis. The timestep reaches the modulations either
  * through the time MLP (`curveGrid` empty) or, in the pruned files, through a
  * table of `curveGrid` points of `timeWidth` (sd-cpp's "AdaLN curves").
  */
final case class MiniMaxH3Config(
    hidden: Int,
    blocks: Int,
    refinerBlocks: Int,
    heads: Int,
    headDimension: Int,
    intermediate: Int,
    videoWidth: Int,
    audioWidth: Int,
    textWidth: Int,
    ropeFrequencies: Int,
    timeWidth: Int,
    curveGrid: Option[Int]
) {

  /** The rotated pairs of each head: `ropeFrequencies` per axis. */
  def ropePairs: Int = 3 * ropeFrequencies
}

object MiniMaxH3Config {
  val Epsilon = 1e-5f

  /** The modalities the modulation tables hold a row for, in their order. */
  val Video = 0
  val Text = 1
  val Audio = 2
  val Modalities = 3

  def holds(source: WeightSource): Boolean =
    source.has("video_patch_proj.weight") && source.has(
      "audio_patch_proj.weight"
    )

  def of(source: WeightSource): MiniMaxH3Config = {
    def dimensions(name: String) = source.shape(name).dimensions.map(_.toInt)
    def count(prefix: String) =
      Iterator
        .from(0)
        .takeWhile(i => source.has(s"$prefix.$i.norm1.weight"))
        .size
    val Seq(hidden, videoWidth) = dimensions("video_patch_proj.weight")
    val headDimension = dimensions("blocks.0.attn.q_norm.weight").head
    val curve = Option.when(source.has("adaln_t_table"))(
      dimensions("adaln_t_table")
    )
    MiniMaxH3Config(
      hidden = hidden,
      blocks = count("blocks"),
      refinerBlocks = count("token_refiner.blocks"),
      heads =
        dimensions("blocks.0.attn.qkv_proj.weight").head / (3 * headDimension),
      headDimension = headDimension,
      intermediate = dimensions("blocks.0.mlp.fc1.weight").head / 2,
      videoWidth = videoWidth,
      audioWidth = dimensions("audio_patch_proj.weight").last,
      textWidth = dimensions("condition_proj.weight").last,
      ropeFrequencies = dimensions("rope.inv_freq").head,
      timeWidth = curve.fold(dimensions("time_embedder.proj_out.weight").head)(
        _.last
      ),
      curveGrid = curve.map(_.head)
    )
  }
}

/** Where the rows of a t2va request sit in MiniMax H3's packed sequence
  * (diffusers' `MiniMaxH3PrepareLayoutStep`): `[text | audio | video]`, the
  * audio channel-major over its two stereo channels, the video frame-major in 2
  * × 2 patches. `positions` gives each row's (t, h, w) rotary coordinates: text
  * on the time axis at its index, the media continuing from there (a latent
  * frame spans 5/3 × (1, 4, 4, 4, 4)); the spatial axes centred and scaled to
  * the canvas' aspect; audio pinned to the width axis' two ends.
  */
final case class MiniMaxH3Layout(
    textTokens: Int,
    frames: Int,
    latentHeight: Int,
    latentWidth: Int,
    audioLatents: Int
) {
  val audioRows: Int = MiniMaxH3Layout.AudioChannels * audioLatents
  val rowsPerFrame: Int = (latentHeight / 2) * (latentWidth / 2)
  val videoRows: Int = frames * rowsPerFrame
  val audioStart: Int = textTokens
  val videoStart: Int = textTokens + audioRows
  val rows: Int = videoStart + videoRows

  /** `[rows × 3]`: (t, h, w) of each row. */
  def positions: Array[Double] = {
    val out = new Array[Double](3 * rows)
    def put(row: Int, t: Double, h: Double, w: Double): Unit = {
      out(3 * row) = t
      out(3 * row + 1) = h
      out(3 * row + 2) = w
    }
    (0 until textTokens).foreach(i => put(i, i, 0, 0))
    val sqrtArea = math.sqrt(latentHeight.toDouble * latentWidth)
    val heights = MiniMaxH3Layout.spatialAxis(latentHeight, sqrtArea)
    val widths = MiniMaxH3Layout.spatialAxis(latentWidth, sqrtArea)
    (0 until audioRows).foreach { i =>
      val (channel, latent) = (i / audioLatents, i % audioLatents)
      put(
        audioStart + i,
        textTokens + latent,
        0,
        if (channel == 0) widths.head else widths.last
      )
    }
    var time = textTokens.toDouble
    (0 until frames).foreach { frame =>
      for {
        (h, y) <- heights.zipWithIndex
        (w, x) <- widths.zipWithIndex
      } put(
        videoStart + frame * rowsPerFrame + y * widths.length + x,
        time,
        h,
        w
      )
      time += MiniMaxH3Layout.FrameRescale * MiniMaxH3Layout.FrameSpans(
        frame % MiniMaxH3Layout.FrameSpans.length
      )
    }
    out
  }
}

object MiniMaxH3Layout {
  val AudioChannels = 2
  val FrameRescale: Double = 5.0 / 3.0
  val FrameSpans: Seq[Int] = Seq(1, 4, 4, 4, 4)

  /** One aspect-normalized spatial axis: `dim / 2` coordinates centred on the
    * unit interval, × 32 (numpy's `linspace(endpoint=False)`).
    */
  def spatialAxis(dim: Int, sqrtArea: Double): Seq[Double] = {
    val ratio = dim / sqrtArea
    val left = (1 - ratio) / 2
    val count = dim / 2
    (0 until count).map(i => (left + i * (ratio / count)) * 32)
  }
}

/** MiniMax H3's joint video and audio transformer (diffusers'
  * `MiniMaxH3Transformer3DModel`, sd-cpp's `minimax_h3.hpp`), from the original
  * names (the GGUFs', ComfyUI's).
  *   - The text: projected to the model's width and refined by plain blocks (no
  *     modulation, no RoPE). It depends on the prompt only: `encodeText` once
  *     per video.
  *   - One stack of blocks over the whole packed sequence (`MiniMaxH3Layout`),
  *     full attention with q/k RMS norms and a rotate-half RoPE over the heads'
  *     first `2 × ropePairs` values, a SwiGLU MLP (`fc1` holds [gate; value]).
  *     Every row is modulated by its (timestep, modality) row of each block's
  *     AdaLN table: the video and the text at the video's timestep, the audio
  *     at its own.
  *   - Timesteps are `t = 1 − σ` in [0, 1]; the velocity points towards the
  *     data (`x₀ = x + σ v`).
  * The quantized products run in BF16 (`Ops.wideProducts`), whose range the MLP
  * needs.
  */
final class MiniMaxH3 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: MiniMaxH3Config = MiniMaxH3Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val epsilon = MiniMaxH3Config.Epsilon

  private def floats(name: String, count: Long): Tensor =
    weights.floats(name, name, Shape.of(count))

  final private case class Affine(weight: Tensor, bias: Tensor)

  private def affineLayer(prefix: String, outputs: Long): Affine =
    Affine(source(s"$prefix.weight"), floats(s"$prefix.bias", outputs))

  final private class Block(prefix: String) {
    private val inner = c.heads.toLong * c.headDimension
    val norm1: Tensor = floats(s"$prefix.norm1.weight", c.hidden)
    val norm2: Tensor = floats(s"$prefix.norm2.weight", c.hidden)
    private val qkv = source(s"$prefix.attn.qkv_proj.weight")
    val q: Tensor = qkv.rows(0, inner)
    val k: Tensor = qkv.rows(inner, inner)
    val v: Tensor = qkv.rows(2 * inner, inner)
    val qNorm: Tensor = floats(s"$prefix.attn.q_norm.weight", c.headDimension)
    val kNorm: Tensor = floats(s"$prefix.attn.k_norm.weight", c.headDimension)
    val o: Tensor = source(s"$prefix.attn.out_proj.weight")
    private val fc1 = source(s"$prefix.mlp.fc1.weight")
    val gate: Tensor = fc1.rows(0, c.intermediate)
    val value: Tensor = fc1.rows(c.intermediate, c.intermediate)
    val down: Tensor = source(s"$prefix.mlp.fc2.weight")
  }

  /** A modulation projection from the time embedding: `rows` of `outputs` per
    * timestep. The full files' through the GPU from the time MLP's output
    * (after a SiLU); the pruned files' (K = the table's width, too narrow for
    * the kernels) on the host from the table's point.
    */
  final private class Modulation(prefix: String, outputs: Long) {
    private val layer = affineLayer(prefix, outputs)
    private lazy val host: (Array[Float], Array[Float]) =
      (
        weights.hostFloats(s"$prefix.weight"),
        weights.hostFloats(s"$prefix.bias")
      )

    /** `[timesteps, outputs]` for `embedded` (`[timesteps, timeWidth]`). */
    def apply(embedded: TimeEmbedding, out: Tensor): Unit = embedded match {
      case TimeEmbedding.Device(values) =>
        ops.linear(values, layer.weight, out)
        ops.addRow(out, layer.bias, out)
      case TimeEmbedding.Host(values) =>
        val (weight, bias) = host
        val width = c.timeWidth
        val result = Array.tabulate(values.length * outputs.toInt) { i =>
          val (row, column) = (i / outputs.toInt, i % outputs.toInt)
          var sum = bias(column).toDouble
          var j = 0
          while (j < width) {
            sum += weight(column * width + j).toDouble * values(row)(j)
            j += 1
          }
          sum.toFloat
        }
        val uploaded = ops.fromFloats(out.shape, result)
        try ops.copy(uploaded, out)
        finally ops.release(uploaded)
    }
  }

  private enum TimeEmbedding {
    case Device(values: Tensor)
    case Host(values: Seq[Array[Float]])
  }

  private val videoIn = affineLayer("video_patch_proj", c.hidden)
  private val audioIn = affineLayer("audio_patch_proj", c.hidden)
  private val textIn = affineLayer("condition_proj", c.hidden)
  private val refiner =
    (0 until c.refinerBlocks).map(i => new Block(s"token_refiner.blocks.$i"))
  private val refinerNorm = floats("token_refiner.final_norm.weight", c.hidden)
  private val blocks = (0 until c.blocks).map(i =>
    (
      new Block(s"blocks.$i"),
      new Modulation(
        s"blocks.$i.adaln_proj.linear",
        6L * c.hidden * MiniMaxH3Config.Modalities
      )
    )
  )
  private val finalNorm = floats("final_layer.norm.weight", c.hidden)
  private val finalModulation =
    new Modulation("final_layer.adaln_proj.linear", 2L * c.hidden)
  private val videoHead = affineLayer("final_layer.video_out", c.videoWidth)
  private val audioHead = affineLayer("final_layer.audio_out", c.audioWidth)
  private val timeIn = Option.when(c.curveGrid.isEmpty)(
    affineLayer("time_embedder.proj_in", c.hidden)
  )
  private val timeOut = Option.when(c.curveGrid.isEmpty)(
    affineLayer("time_embedder.proj_out", c.timeWidth)
  )
  private lazy val curveTable: Array[Float] =
    weights.hostFloats("adaln_t_table")
  private val inverseFrequencies: Array[Float] =
    weights.hostFloats("rope.inv_freq")

  private val attention = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  /** Flat buffers for `tokens` rows, grown when too small. */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(dtype: DType, elements: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(elements))
      held += tensor
      tensor
    }
    private val inner = c.heads.toLong * c.headDimension
    private val rows = tokens.toLong
    val x: Tensor = flat(DType.F32, rows * c.hidden)
    val normed: Tensor = flat(DType.F32, rows * math.max(c.hidden, c.textWidth))
    val q: Tensor = flat(DType.F32, rows * inner)
    val k: Tensor = flat(DType.F32, rows * inner)
    val v: Tensor = flat(DType.F32, rows * inner)
    val rotatedQueries: Tensor = flat(DType.F32, rows * inner)
    val rotatedKeys: Tensor = flat(DType.F32, rows * inner)
    val attended: Tensor = flat(DType.F32, rows * inner)
    val projected: Tensor = flat(DType.F32, rows * c.hidden)
    val gate: Tensor = flat(DType.F32, rows * c.intermediate)
    val value: Tensor = flat(DType.F32, rows * c.intermediate)
    val cosines: Tensor = flat(DType.F32, rows * c.ropePairs)
    val sines: Tensor = flat(DType.F32, rows * c.ropePairs)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, c.heads, c.headDimension)
    val pageTable: Tensor = {
      val tensor = ops.fromInts(Shape.of(1), Array(0))
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
  private def buffersFor(tokens: Int): Buffers =
    buffers.filter(_.tokens >= tokens).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens)
      buffers = Some(created)
      created
    }

  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  /** A modulated span of rows: `[start, start + count)` at its table row. */
  final private case class Span(start: Int, count: Int, row: Int)

  /** A block on `x` (`[rows, hidden]`, in place). With `modulation` (the
    * block's projection, `[timesteps × modalities, 6 × hidden]`) each span is
    * modulated and gated by its row; `rotate` turns the queries and keys.
    */
  private def block(
      b: Block,
      x: Tensor,
      modulation: Option[(Tensor, Seq[Span])],
      rotate: Boolean,
      w: Buffers
  ): Unit = {
    val rows = x.shape.dimensions.head
    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    val f = c.hidden.toLong
    def chunk(row: Int, index: Int)(table: Tensor) =
      table
        .view(table.shape.elementCount / (6 * f), 6 * f)
        .rows(row, 1)
        .view(6, f)
        .rows(index, 1)
        .view(f)
    def modulate(normed: Tensor, shift: Int, scale: Int): Unit =
      modulation.foreach { (table, spans) =>
        spans.foreach { span =>
          val part = normed.rows(span.start, span.count)
          ops.modulate(
            part,
            chunk(span.row, scale)(table),
            chunk(span.row, shift)(table),
            part
          )
        }
      }
    def residual(y: Tensor, gate: Int): Unit = modulation match {
      case Some((table, spans)) =>
        spans.foreach(span =>
          ops.gatedAdd(
            x.rows(span.start, span.count),
            y.rows(span.start, span.count),
            chunk(span.row, gate)(table)
          )
        )
      case None => ops.add(x, y, x)
    }
    val normed = w.normed.prefix(rows, f)
    ops.rmsNorm(x, b.norm1, epsilon, 0f, normed)
    modulate(normed, 0, 1)
    val inner = heads * d
    val (q, k, v) =
      (
        w.q.prefix(rows, inner),
        w.k.prefix(rows, inner),
        w.v.prefix(rows, inner)
      )
    ops.linears(normed, Seq(b.q, b.k, b.v), Seq(q, k, v))
    ops.rmsNorm(
      q.view(rows * heads, d),
      b.qNorm,
      epsilon,
      0f,
      q.view(rows * heads, d)
    )
    ops.rmsNorm(
      k.view(rows * heads, d),
      b.kNorm,
      epsilon,
      0f,
      k.view(rows * heads, d)
    )
    val (queries, keys) =
      if (rotate) {
        val (rq, rk) = (
          w.rotatedQueries.prefix(rows, heads, d),
          w.rotatedKeys.prefix(rows, heads, d)
        )
        val (cosines, sines) = (
          w.cosines.prefix(rows, c.ropePairs),
          w.sines.prefix(rows, c.ropePairs)
        )
        ops.ropeTable(q.view(rows, heads, d), cosines, sines, rq, halves = true)
        ops.ropeTable(k.view(rows, heads, d), cosines, sines, rk, halves = true)
        (rq, rk)
      } else (q.view(rows, heads, d), k.view(rows, heads, d))
    ops.cacheWrite(keys, v.view(rows, heads, d), w.cache, w.pageTable, 0)
    val attended = w.attended.prefix(rows, heads, d)
    ops.attention(
      queries,
      w.cache,
      w.pageTable,
      0,
      rows.toInt,
      attention,
      attended
    )
    val projected = w.projected.prefix(rows, f)
    ops.linear(attended.view(rows, inner), b.o, projected)
    residual(projected, 2)
    ops.rmsNorm(x, b.norm2, epsilon, 0f, normed)
    modulate(normed, 3, 4)
    val (gate, value) = (
      w.gate.prefix(rows, c.intermediate),
      w.value.prefix(rows, c.intermediate)
    )
    ops.linears(normed, Seq(b.gate, b.value), Seq(gate, value))
    ops.gated(Activation.Silu, gate, value, gate)
    ops.linear(gate, b.down, projected)
    residual(projected, 5)
  }

  /** The prompt's text for every step: the encoder's `[L, textWidth]` projected
    * and refined into `out` (`[L, hidden]`).
    */
  def encodeText(text: Tensor, out: Tensor): Unit = {
    val tokens = text.shape.dimensions.head
    require(
      text.shape == Shape.of(tokens, c.textWidth) &&
        out.shape == Shape.of(tokens, c.hidden),
      s"encodeText: text ${text.shape}, out ${out.shape}"
    )
    val w = buffersFor(tokens.toInt)
    affine(text, textIn, out)
    refiner.foreach(block(_, out, None, rotate = false, w))
    val normed = w.normed.prefix(tokens, c.hidden)
    ops.rmsNorm(out, refinerNorm, epsilon, 0f, normed)
    ops.copy(normed, out)
  }

  /** The time embeddings of `timesteps` (each in [0, 1]). */
  private def embed(timesteps: Seq[Float], small: mutable.Buffer[Tensor]) =
    c.curveGrid match {
      case Some(grid) =>
        val width = c.timeWidth
        TimeEmbedding.Host(timesteps.map { t =>
          val position = math.min(math.max(t, 0f), 1f) * (grid - 1)
          val index = math.min(math.floor(position).toInt, grid - 2)
          val fraction = position - index
          Array.tabulate(width) { j =>
            val lower = curveTable(index * width + j)
            val upper = curveTable((index + 1) * width + j)
            lower + (upper - lower) * fraction
          }
        })
      case None =>
        // cosines first, over frequencies exp(−ln(10⁴) i / half), t unscaled
        val half = timeIn.get.weight.shape.last.toInt / 2
        val sinusoid = timesteps.flatMap { t =>
          (0 until 2 * half).map { i =>
            val frequency = math.exp(-math.log(10000) * (i % half) / half)
            val angle = t * frequency
            (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
          }
        }.toArray
        val rows = timesteps.size.toLong
        def keep(tensor: Tensor) = { small += tensor; tensor }
        val embedded = keep(ops.fromFloats(Shape.of(rows, 2L * half), sinusoid))
        val inner = keep(
          ops.allocate(DType.F32, Shape.of(rows, c.hidden.toLong))
        )
        val time = keep(
          ops.allocate(DType.F32, Shape.of(rows, c.timeWidth.toLong))
        )
        affine(embedded, timeIn.get, inner)
        ops.activation(Activation.Silu, inner, inner)
        affine(inner, timeOut.get, time)
        // every modulation reads it through a SiLU
        ops.activation(Activation.Silu, time, time)
        TimeEmbedding.Device(time)
    }

  /** The velocities of a t2va sequence laid out as `layout`: `videoRows`
    * (`[videoRows, videoWidth]`, the patchified latents) at `videoTime`,
    * `audioRows` (`[audioRows, audioWidth]`, channel-major) at `audioTime`,
    * given the prompt's `text` (`[L, hidden]` from `encodeText`), into
    * `videoOut` and `audioOut` (like their inputs).
    */
  def velocity(
      layout: MiniMaxH3Layout,
      text: Tensor,
      videoRows: Tensor,
      audioRows: Tensor,
      videoTime: Float,
      audioTime: Float,
      videoOut: Tensor,
      audioOut: Tensor
  ): Unit = {
    val (length, audio, video) =
      (layout.textTokens, layout.audioRows, layout.videoRows)
    require(
      text.shape == Shape.of(length, c.hidden) &&
        videoRows.shape == Shape.of(video, c.videoWidth) &&
        audioRows.shape == Shape.of(audio, c.audioWidth) &&
        videoOut.shape == videoRows.shape && audioOut.shape == audioRows.shape,
      s"velocity: text ${text.shape}, video ${videoRows.shape}, audio ${audioRows.shape} for $layout"
    )
    val tokens = layout.rows
    val w = buffersFor(tokens)
    val f = c.hidden.toLong
    val small = mutable.ArrayBuffer.empty[Tensor]
    try {
      val timesteps = Seq(videoTime, audioTime).distinct
      val (videoRow, audioRow) =
        (timesteps.indexOf(videoTime), timesteps.indexOf(audioTime))
      def row(timestep: Int, modality: Int) =
        timestep * MiniMaxH3Config.Modalities + modality
      val spans = Seq(
        Span(0, length, row(videoRow, MiniMaxH3Config.Text)),
        Span(layout.audioStart, audio, row(audioRow, MiniMaxH3Config.Audio)),
        Span(layout.videoStart, video, row(videoRow, MiniMaxH3Config.Video))
      )
      val embedded = embed(timesteps, small)
      // the packed sequence
      val stream = w.x.prefix(tokens, f)
      ops.copy(text, stream.rows(0, length))
      affine(audioRows, audioIn, stream.rows(layout.audioStart, audio))
      affine(videoRows, videoIn, stream.rows(layout.videoStart, video))
      // the rotary angles: (t, h, w) × each axis' frequencies
      val positions = layout.positions
      val pairs = c.ropePairs
      val angles = Array.tabulate(tokens * pairs) { i =>
        val (token, pair) = (i / pairs, i % pairs)
        val (axis, frequency) =
          (pair / c.ropeFrequencies, pair % c.ropeFrequencies)
        positions(3 * token + axis) * inverseFrequencies(frequency).toDouble
      }
      val cosines = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.cos(a).toFloat)
      )
      small += cosines
      val sines = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.sin(a).toFloat)
      )
      small += sines
      ops.copy(cosines, w.cosines.prefix(tokens, pairs))
      ops.copy(sines, w.sines.prefix(tokens, pairs))
      val table = ops.allocate(
        DType.F32,
        Shape.of(timesteps.size.toLong, 6 * f * MiniMaxH3Config.Modalities)
      )
      small += table
      blocks.foreach { (b, modulation) =>
        modulation(embedded, table)
        block(b, stream, Some(table -> spans), rotate = true, w)
      }
      // the final norm, shift and scale per timestep, then each head
      val last = ops.allocate(DType.F32, Shape.of(timesteps.size.toLong, 2 * f))
      small += last
      finalModulation(embedded, last)
      val normed = w.normed.prefix(tokens, f)
      ops.rmsNorm(stream, finalNorm, epsilon, 0f, normed)
      def head(
          start: Int,
          count: Int,
          timestep: Int,
          layer: Affine,
          out: Tensor
      ) = {
        val part = normed.rows(start, count)
        val row = last.rows(timestep, 1).view(2, f)
        ops.modulate(part, row.rows(1, 1).view(f), row.rows(0, 1).view(f), part)
        affine(part, layer, out)
      }
      head(layout.videoStart, video, videoRow, videoHead, videoOut)
      head(layout.audioStart, audio, audioRow, audioHead, audioOut)
    } finally small.foreach(ops.release)
  }

  def close(): Unit = {
    buffers.foreach(_.release())
    weights.release()
    source.close()
  }
}

object MiniMaxH3 {

  def open(ops: Ops, path: Path): MiniMaxH3 = {
    val source = WeightSource.open(ops, path)
    try {
      if (!MiniMaxH3Config.holds(source))
        throw new FormatException(
          s"$path is no MiniMax H3 checkpoint (no video_patch_proj.weight)"
        )
      new MiniMaxH3(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
