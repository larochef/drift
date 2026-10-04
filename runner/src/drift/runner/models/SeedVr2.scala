package drift.runner.models

import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** SeedVR2's transformer shape, read from the checkpoint. */
final case class SeedVr2Config(
    hidden: Int,
    layers: Int,
    /** The first layers keep video and text weights apart; the rest share. */
    separateLayers: Int,
    heads: Int,
    headDimension: Int,
    intermediate: Int,
    /** The 3B's generation of the network (SwiGLU MLPs, a RoPE over video and
      * text, a normed and modulated output, a last layer that leaves the text
      * alone) or the 7B's (plain MLPs with biases, a RoPE over the video only,
      * a bare output).
      */
    swiglu: Boolean,
    textWidth: Int,
    inputChannels: Int,
    outputChannels: Int
)

object SeedVr2Config {
  val Epsilon = 1e-5f
  val RopeTheta = 10000f

  /** A token is 1 × 2 × 2 latents. */
  val Patch = 2

  def of(source: WeightSource): SeedVr2Config = {
    def dimensions(name: String) = source.shape(name).dimensions.map(_.toInt)
    def separate(i: Int) = source.has(s"blocks.$i.attn.proj_qkv.vid.weight")
    def tag(i: Int) = if (separate(i)) "vid" else "all"
    val layers = Iterator
      .from(0)
      .takeWhile(i => source.has(s"blocks.$i.attn.proj_qkv.${tag(i)}.weight"))
      .size
    val Seq(hidden, packedInput) = dimensions("vid_in.proj.weight")
    val headDimension = dimensions(
      s"blocks.0.attn.norm_q.${tag(0)}.weight"
    ).head
    SeedVr2Config(
      hidden = hidden,
      layers = layers,
      separateLayers = (0 until layers).count(separate),
      heads = hidden / headDimension,
      headDimension = headDimension,
      intermediate = dimensions(s"blocks.0.mlp.${tag(0)}.proj_in.weight").head,
      swiglu = source.has(s"blocks.0.mlp.${tag(0)}.proj_in_gate.weight"),
      textWidth = dimensions("txt_in.weight").last,
      inputChannels = packedInput / (Patch * Patch),
      outputChannels = dimensions("vid_out.proj.weight").head / (Patch * Patch)
    )
  }
}

/** A window of a token grid: frames `[t0, t1)`, rows `[h0, h1)`, columns
  * `[w0, w1)`.
  */
final case class SeedVr2Window(
    t0: Int,
    t1: Int,
    h0: Int,
    h1: Int,
    w0: Int,
    w1: Int
) {
  def frames: Int = t1 - t0
  def height: Int = h1 - h0
  def width: Int = w1 - w0
  def tokens: Int = frames * height * width
}

/** The windows SeedVR2's attentions work in (the reference's `window.py`,
  * `720pwin_by_size_bysize` and its shifted twin, for windows of `(4, 3, 3)`):
  * the grid is scaled to 720p's area (45 × 80 tokens) and cut in 3 × 3 there,
  * `min(frames, 30)` in 4, and windows of that size are laid over the real grid
  * — whole ones from the origin, or shifted by half a window on every axis that
  * has more than one.
  */
object SeedVr2Windows {

  private def ceilDiv(a: Int, b: Int) = (a + b - 1) / b

  /** The window's size in frames, rows and columns. */
  private def size(frames: Int, height: Int, width: Int): (Int, Int, Int) = {
    val scale = math.sqrt(45.0 * 80 / (height.toLong * width))
    // Python's round: halves to the even
    val (scaledHeight, scaledWidth) =
      (math.rint(height * scale).toInt, math.rint(width * scale).toInt)
    (
      ceilDiv(math.min(frames, 30), 4),
      ceilDiv(scaledHeight, 3),
      ceilDiv(scaledWidth, 3)
    )
  }

  /** Width outermost, then height, then time, as the reference lists them. */
  private def laid(
      times: Seq[(Int, Int)],
      heights: Seq[(Int, Int)],
      widths: Seq[(Int, Int)]
  ): Seq[SeedVr2Window] =
    for {
      (w0, w1) <- widths if w1 > w0
      (h0, h1) <- heights if h1 > h0
      (t0, t1) <- times if t1 > t0
    } yield SeedVr2Window(t0, t1, h0, h1, w0, w1)

  def plain(frames: Int, height: Int, width: Int): Seq[SeedVr2Window] = {
    val (wt, wh, ww) = size(frames, height, width)
    def axis(length: Int, window: Int) =
      (0 until ceilDiv(length, window)).map(i =>
        (i * window, math.min((i + 1) * window, length))
      )
    laid(axis(frames, wt), axis(height, wh), axis(width, ww))
  }

  def shifted(frames: Int, height: Int, width: Int): Seq[SeedVr2Window] = {
    val (wt, wh, ww) = size(frames, height, width)
    def axis(length: Int, window: Int) = {
      val shift = if (window < length) 0.5 else 0.0
      val count =
        if (shift > 0) math.ceil((length - shift) / window).toInt + 1 else 1
      (0 until count).map(i =>
        (
          math.max(((i - shift) * window).toInt, 0),
          math.min(((i - shift + 1) * window).toInt, length)
        )
      )
    }
    laid(axis(frames, wt), axis(height, wh), axis(width, ww))
  }
}

/** SeedVR2's transformer (ByteDance's `NaDiT`), the 3B and the 7B, from the
  * released names. The 3B is described; the 7B is the older shape, told by its
  * plain MLPs (a linear, GELU, a linear, with biases): every layer holds
  * separate video and text weights and treats the text in full, the RoPE turns
  * the video alone, by angles (10 pairs an axis, frequencies from π to 128 π,
  * each axis of a window running from −1 to 1), and the output is the linear
  * alone. A multimodal DiT over video tokens (1 × 2 × 2 latents: 16 channels of
  * noise, 16 of the low-quality latent, a mask) and the tokens of a fixed text
  * embedding:
  *   - the first layers hold separate video and text weights (`vid`, `txt`),
  *     the rest share them (`all`); the last layer leaves the text as it is
  *     past its attention;
  *   - a layer is weightless RMS norm, modulation, attention, gate, residual,
  *     then the same around a SwiGLU MLP. The modulation is AdaLN-single: the
  *     timestep's embedding, six vectors (shift, scale and gate for the
  *     attention, then for the MLP, value `d × 6 + n` of the embedding), each
  *     added to the layer's own;
  *   - the attention is **windowed** (`SeedVr2Windows`), plain windows on even
  *     layers and shifted ones on odd layers. Every window attends over its own
  *     video tokens and all the text tokens, whose outputs are averaged over
  *     the windows. Queries and keys take an RMS norm per head and a three-axis
  *     RoPE counted inside the window: 21 pairs each for the frame, the row and
  *     the column (the last 2 values of 128 do not turn), the frame counted
  *     from the text's length on, and text token `i` at `(i, i, i)`;
  *   - the output is a weighted RMS norm, a modulation by the *attention's*
  *     shift and scale of the timestep (the reference reads them from its cache
  *     under the attention's key, and was trained so) plus its own, and a
  *     linear back to patches.
  * The RoPE frequencies are computed, not read: an fp8 file rounds the ones it
  * stores. Tokens come and go as `Ops.packPatches` packs them (feature `c × 4 +
  * py × 2 + px`), the patch projections being reordered at load.
  */
final class SeedVr2 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: SeedVr2Config = SeedVr2Config.of(source)
  private val c = config
  private val epsilon = SeedVr2Config.Epsilon
  private val f = c.hidden.toLong
  private val patch = SeedVr2Config.Patch * SeedVr2Config.Patch

  private val made = mutable.ArrayBuffer.empty[Tensor]
  private def keep(tensor: Tensor): Tensor = { made += tensor; tensor }

  /** A stored vector or matrix as F32, on the host. */
  private def hostFloats(name: String): Array[Float] = {
    val stored = source(name)
    if (stored.dtype == DType.F32) ops.toFloats(stored)
    else {
      val count = stored.shape.elementCount
      val floats = ops.allocate(DType.F32, Shape.of(1, count))
      try {
        ops.convert(stored.view(1, count), floats)
        ops.toFloats(floats)
      } finally ops.release(floats)
    }
  }

  private def floats(name: String): Tensor = {
    val values = hostFloats(name)
    keep(ops.fromFloats(Shape.of(values.length), values))
  }

  private def bf16(shape: Shape, values: Array[Float]): Tensor = {
    val uploaded = ops.fromFloats(shape, values)
    try {
      val out = keep(ops.allocate(DType.BF16, shape))
      ops.convert(uploaded, out)
      out
    } finally ops.release(uploaded)
  }

  final private case class Affine(weight: Tensor, bias: Tensor)

  private def affineLayer(prefix: String): Affine =
    Affine(source(s"$prefix.weight"), floats(s"$prefix.bias"))

  /** `out = x · weight + bias`. */
  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    ops.addRow(out, layer.bias, out)
  }

  /** One modality's weights of a layer (`vid`, `txt`, or `all` for both). */
  final private class Branch(prefix: String, tag: String) {
    private val qkv = source(s"$prefix.attn.proj_qkv.$tag.weight")
    val q: Tensor = qkv.rows(0, f)
    val k: Tensor = qkv.rows(f, f)
    val v: Tensor = qkv.rows(2 * f, f)
    val out: Affine = affineLayer(s"$prefix.attn.proj_out.$tag")
    val qNorm: Tensor = floats(s"$prefix.attn.norm_q.$tag.weight")
    val kNorm: Tensor = floats(s"$prefix.attn.norm_k.$tag.weight")
    val mlpGate: Option[Tensor] =
      Option.when(c.swiglu)(source(s"$prefix.mlp.$tag.proj_in_gate.weight"))
    val mlpUp: Tensor = source(s"$prefix.mlp.$tag.proj_in.weight")
    val mlpDown: Tensor = source(s"$prefix.mlp.$tag.proj_out.weight")
    // the plain MLP's biases
    val mlpUpBias: Option[Tensor] =
      Option.when(!c.swiglu)(floats(s"$prefix.mlp.$tag.proj_in.bias"))
    val mlpDownBias: Option[Tensor] =
      Option.when(!c.swiglu)(floats(s"$prefix.mlp.$tag.proj_out.bias"))

    /** The layer's own modulation: shift, scale and gate of the attention, then
      * of the MLP.
      */
    val modulation: Seq[Array[Float]] =
      for {
        layer <- Seq("attn", "mlp")
        part <- Seq("shift", "scale", "gate")
      } yield hostFloats(s"$prefix.ada.$tag.${layer}_$part")
  }

  final private class Block(index: Int) {
    private val prefix = s"blocks.$index"
    val last: Boolean = index == c.layers - 1
    val shifted: Boolean = index % 2 == 1
    val (video, text) =
      if (index < c.separateLayers)
        (new Branch(prefix, "vid"), new Branch(prefix, "txt"))
      else {
        val all = new Branch(prefix, "all")
        (all, all)
      }
  }

  // tokens as `packPatches` packs them: feature `c × 4 + p`, stored `p × C + c`
  private val videoIn: Affine = {
    val channels = c.inputChannels
    val stored = hostFloats("vid_in.proj.weight")
    val columns = channels * patch
    Affine(
      bf16(
        Shape.of(f, columns.toLong),
        Array.tabulate(c.hidden * columns) { i =>
          val (row, column) = (i / columns, i % columns)
          stored(row * columns + (column % patch) * channels + column / patch)
        }
      ),
      floats("vid_in.proj.bias")
    )
  }
  private val videoOut: Affine = {
    val channels = c.outputChannels
    val (weight, bias) =
      (hostFloats("vid_out.proj.weight"), hostFloats("vid_out.proj.bias"))
    val rows = channels * patch
    def stored(row: Int) = (row % patch) * channels + row / patch
    Affine(
      bf16(
        Shape.of(rows.toLong, f),
        Array.tabulate(rows * c.hidden)(i =>
          weight(stored(i / c.hidden) * c.hidden + i % c.hidden)
        )
      ),
      keep(
        ops.fromFloats(
          Shape.of(rows),
          Array.tabulate(rows)(r => bias(stored(r)))
        )
      )
    )
  }
  private val textIn = affineLayer("txt_in")
  private val timeIn = affineLayer("emb_in.proj_in")
  private val timeHidden = affineLayer("emb_in.proj_hid")
  private val timeOut = affineLayer("emb_in.proj_out")
  private val blocks = (0 until c.layers).map(new Block(_))
  private val outNorm = Option.when(c.swiglu)(floats("vid_out_norm.weight"))
  private val outShift =
    if (c.swiglu) hostFloats("vid_out_ada.out_shift") else Array.empty[Float]
  private val outScale =
    if (c.swiglu) hostFloats("vid_out_ada.out_scale") else Array.empty[Float]

  /** No weight: a weightless RMS norm takes it with an offset of 1. */
  private val noWeight = keep(
    ops.fromFloats(Shape.of(f), new Array[Float](c.hidden))
  )

  private val rope = Rope(
    SeedVr2Config.RopeTheta,
    c.headDimension / 6 * 6,
    RopeLayout.Interleaved,
    RopeSections.Axes(Seq.fill(3)(c.headDimension / 6))
  )

  /** The 7B's RoPE (rotary-embedding-torch's "pixel" kind, `max_freq` 256): a
    * third of half a head, in pairs, from π to 128 π.
    */
  private val pixelFrequencies: Seq[Float] = {
    val count = c.headDimension / 2 / 3 / 2
    (0 until count).map(i => ((1 + 127.0 * i / (count - 1)) * math.Pi).toFloat)
  }
  private val attention = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  /** The fixed text embedding (`[L, textWidth]`) as the layers take it, into
    * `out` (`[L, hidden]`): once for every picture and clip.
    */
  def encodeText(embedding: Tensor, out: Tensor): Unit =
    affine(embedding, textIn, out)

  /** A grid's windows as one attention runs them: the tokens in window order
    * (`order(i)` is the grid's token at window position `i`), the way back,
    * each token's RoPE position, and each window's first position and length.
    */
  final private class Layout(
      windows: Seq[SeedVr2Window],
      gridHeight: Int,
      gridWidth: Int,
      textLength: Int,
      keeping: Tensor => Tensor
  ) {
    // a token of the grid, its window, and its place in it
    private val cells = for {
      window <- windows
      t <- window.t0 until window.t1
      h <- window.h0 until window.h1
      w <- window.w0 until window.w1
    } yield (
      (t * gridHeight + h) * gridWidth + w,
      window,
      Seq(t - window.t0, h - window.h0, w - window.w0)
    )
    val tokens: Int = cells.size
    val order: Tensor =
      keeping(ops.fromInts(Shape.of(tokens), cells.map(_._1).toArray))
    val back: Tensor = {
      val inverse = new Array[Int](tokens)
      cells.zipWithIndex.foreach { case ((token, _, _), i) =>
        inverse(token) = i
      }
      keeping(ops.fromInts(Shape.of(tokens), inverse))
    }

    /** The 3B's positions: the frame counted from the text's length on. */
    lazy val positions: Tensor = keeping(
      ops.fromInts(
        Shape.of(3, tokens),
        (cells.map(textLength + _._3(0)) ++ cells.map(_._3(1)) ++
          cells.map(_._3(2))).toArray
      )
    )

    /** The 7B's angles, as cosines and sines `[tokens, 3 × frequencies]`: each
      * axis of the window runs from −1 to 1 (−1 alone when it is one long).
      */
    lazy val turns: (Tensor, Tensor) = {
      val angles = cells.flatMap { (_, window, place) =>
        Seq(window.frames, window.height, window.width).zip(place).flatMap {
          (size, at) =>
            val position = if (size == 1) -1f else -1f + 2f * at / (size - 1)
            pixelFrequencies.map(frequency => (position * frequency).toDouble)
        }
      }.toArray
      val shape = Shape.of(tokens, 3L * pixelFrequencies.size)
      (
        keeping(ops.fromFloats(shape, angles.map(math.cos(_).toFloat))),
        keeping(ops.fromFloats(shape, angles.map(math.sin(_).toFloat)))
      )
    }
    val spans: Seq[(Int, Int)] =
      windows.map(_.tokens).scanLeft(0)(_ + _).zip(windows.map(_.tokens))
    val largest: Int = windows.map(_.tokens).max
  }

  /** The velocity at `timestep` (0 to 1000) of `tokens` (`[frames × gridHeight
    * × gridWidth, 4 × 33]`: the packed patches of the latent frames, frame
    * after frame, row-major), given `text` (`[L, hidden]` from `encodeText`),
    * into `out` (`[frames × gridHeight × gridWidth, 4 × 16]`, packed alike).
    */
  def velocity(
      tokens: Tensor,
      text: Tensor,
      timestep: Float,
      frames: Int,
      gridHeight: Int,
      gridWidth: Int,
      out: Tensor
  ): Unit = {
    val count = frames * gridHeight * gridWidth
    val length = text.shape.dimensions.head.toInt
    require(
      tokens.shape == Shape.of(count, c.inputChannels.toLong * patch) &&
        out.shape == Shape.of(count, c.outputChannels.toLong * patch) &&
        text.shape == Shape.of(length, f),
      s"velocity: tokens ${tokens.shape}, text ${text.shape}, out ${out.shape} for a $frames × $gridHeight × $gridWidth grid"
    )
    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    val live = mutable.ArrayBuffer.empty[Tensor]
    def kept(tensor: Tensor): Tensor = { live += tensor; tensor }
    def rows(n: Long, width: Long) =
      kept(ops.allocate(DType.F32, Shape.of(n, width)))
    var cache = Option.empty[KvCache]
    try {
      // ---- the timestep: sines then cosines, through its MLP ----
      val half = timeIn.weight.shape.last.toInt / 2
      val sinusoid = Array.tabulate(2 * half) { i =>
        val angle =
          timestep * math.exp(-math.log(10000) * (i % half) / half).toFloat
        if (i < half) math.sin(angle).toFloat else math.cos(angle).toFloat
      }
      val embedded = kept(ops.fromFloats(Shape.of(1, 2L * half), sinusoid))
      val (inner, hidden, modulation) = (rows(1, f), rows(1, f), rows(1, 6 * f))
      affine(embedded, timeIn, inner)
      ops.activation(Activation.Silu, inner, inner)
      affine(inner, timeHidden, hidden)
      ops.activation(Activation.Silu, hidden, hidden)
      affine(hidden, timeOut, modulation)
      val time = ops.toFloats(modulation)
      // the timestep's vector `part` (shift, scale, gate of the attention, then
      // of the MLP) plus a layer's own; `one` takes 1 off, for `modulate`
      def vector(own: Array[Float], part: Int, one: Boolean): Array[Float] =
        Array.tabulate(c.hidden)(i =>
          time(i * 6 + part) + own(i) - (if (one) 1f else 0f)
        )
      // per layer: the video's six vectors, then the text's
      val table = kept(
        ops.fromFloats(
          Shape.of(c.layers * 12L + 2, f),
          (blocks.flatMap(block =>
            Seq(block.video, block.text).flatMap(branch =>
              (0 until 6).flatMap(part =>
                vector(branch.modulation(part), part, one = part % 3 == 1)
              )
            )
          ) ++ (if (c.swiglu)
                  vector(outShift, 0, one = false) ++
                    vector(outScale, 1, one = true)
                else new Array[Float](2 * c.hidden))).toArray
        )
      )
      def modulationOf(layer: Int, text: Boolean, part: Int): Tensor =
        table.rows(layer * 12L + (if (text) 6 else 0) + part, 1).view(f)

      // ---- the windows ----
      val layouts =
        Seq(SeedVr2Windows.plain, SeedVr2Windows.shifted).map(windows =>
          new Layout(
            windows(frames, gridHeight, gridWidth),
            gridHeight,
            gridWidth,
            length,
            kept
          )
        )
      require(
        layouts.forall(_.tokens == count),
        s"velocity: windows over ${layouts.map(_.tokens)} of $count tokens"
      )
      val textPositions = kept(
        ops.fromInts(
          Shape.of(3, length),
          Array.tabulate(3 * length)(_ % length)
        )
      )
      val pageSize = (layouts.map(_.largest).max + length + 15) / 16 * 16
      val kv = ops.allocateCache(1, pageSize, c.heads, c.headDimension)
      cache = Some(kv)
      val pageTable = kept(ops.fromInts(Shape.of(1), Array(0)))

      // ---- the streams and their buffers ----
      val video = rows(count, f)
      val words = rows(length, f)
      affine(tokens, videoIn, video)
      ops.copy(text, words)

      final class Buffers(n: Long) {
        val normed: Tensor = rows(n, f)
        val q: Tensor = rows(n, f)
        val k: Tensor = rows(n, f)
        val v: Tensor = rows(n, f)
        val rotatedQueries: Tensor = rows(n, f)
        val rotatedKeys: Tensor = rows(n, f)
        val attended: Tensor = rows(n, f)
        val projected: Tensor = rows(n, f)
        val mlpUp: Tensor = rows(n, c.intermediate)
        val mlpGate: Tensor = if (c.swiglu) rows(n, c.intermediate) else mlpUp
      }
      val (onVideo, onText) = (new Buffers(count), new Buffers(length))
      val gathered = rows(count, f)
      val textPart = rows(length, f)

      def normModulate(
          x: Tensor,
          w: Buffers,
          modulate: Option[(Tensor, Tensor)]
      ): Unit = {
        ops.rmsNorm(x, noWeight, epsilon, 1f, w.normed)
        modulate.foreach((shift, scale) =>
          ops.modulate(w.normed, scale, shift, w.normed)
        )
      }

      /** Queries and keys normed per head and turned, the values beside. */
      def project(
          input: Tensor,
          branch: Branch,
          w: Buffers,
          turn: (Tensor, Tensor) => Unit
      ): Unit = {
        val n = input.shape.dimensions.head
        ops.linears(
          input,
          Seq(branch.q, branch.k, branch.v),
          Seq(w.q, w.k, w.v)
        )
        Seq(w.q -> branch.qNorm, w.k -> branch.kNorm).foreach((x, weight) =>
          ops.rmsNorm(
            x.view(n * heads, d),
            weight,
            epsilon,
            0f,
            x.view(n * heads, d)
          )
        )
        turn(w.q.view(n, heads, d), w.rotatedQueries.view(n, heads, d))
        turn(w.k.view(n, heads, d), w.rotatedKeys.view(n, heads, d))
      }
      // the 3B turns video and text by positions, the 7B the video alone, by
      // each window's own angles
      def turned(positions: Tensor)(x: Tensor, out: Tensor): Unit =
        ops.rope(x, positions, rope, out)
      def turnVideo(layout: Layout): (Tensor, Tensor) => Unit =
        if (c.swiglu) turned(layout.positions)
        else (x, out) => ops.ropeTable(x, layout.turns._1, layout.turns._2, out)
      val turnText: (Tensor, Tensor) => Unit =
        if (c.swiglu) turned(textPositions) else ops.copy

      def mlp(x: Tensor, branch: Branch, w: Buffers, gate: Tensor): Unit = {
        branch.mlpGate match {
          case Some(gateWeight) =>
            ops.linears(
              w.normed,
              Seq(gateWeight, branch.mlpUp),
              Seq(w.mlpGate, w.mlpUp)
            )
            ops.gated(Activation.Silu, w.mlpGate, w.mlpUp, w.mlpGate)
          case None =>
            ops.linear(w.normed, branch.mlpUp, w.mlpUp)
            branch.mlpUpBias.foreach(ops.addRow(w.mlpUp, _, w.mlpUp))
            ops.activation(Activation.GeluTanh, w.mlpUp, w.mlpUp)
        }
        ops.linear(w.mlpGate, branch.mlpDown, w.projected)
        branch.mlpDownBias.foreach(ops.addRow(w.projected, _, w.projected))
        ops.gatedAdd(x, w.projected, gate)
      }

      blocks.zipWithIndex.foreach { (block, index) =>
        val layout = layouts(if (block.shifted) 1 else 0)
        def part(text: Boolean, n: Int) = modulationOf(index, text, n)
        // the 3B's last layer leaves the text unmodulated
        val textual = !(block.last && c.swiglu)

        // ---- attention ----
        normModulate(video, onVideo, Some((part(false, 0), part(false, 1))))
        ops.embedding(onVideo.normed, layout.order, gathered)
        project(gathered, block.video, onVideo, turnVideo(layout))
        normModulate(
          words,
          onText,
          Option.when(textual)((part(true, 0), part(true, 1)))
        )
        project(onText.normed, block.text, onText, turnText)

        def heads3(x: Tensor, from: Long, n: Long) =
          x.view(x.shape.dimensions.head, heads, d).rows(from, n)
        ops.zero(onText.attended)
        layout.spans.foreach { (first, size) =>
          ops.cacheWrite(
            heads3(onVideo.rotatedKeys, first, size),
            heads3(onVideo.v, first, size),
            kv,
            pageTable,
            0
          )
          ops.cacheWrite(
            heads3(onText.rotatedKeys, 0, length),
            heads3(onText.v, 0, length),
            kv,
            pageTable,
            size
          )
          ops.attention(
            heads3(onVideo.rotatedQueries, first, size),
            kv,
            pageTable,
            0,
            size + length,
            attention,
            heads3(onVideo.attended, first, size)
          )
          ops.attention(
            heads3(onText.rotatedQueries, 0, length),
            kv,
            pageTable,
            size,
            size + length,
            attention,
            heads3(textPart, 0, length)
          )
          ops.add(onText.attended, textPart, onText.attended)
        }
        affine(onVideo.attended, block.video.out, gathered)
        ops.embedding(gathered, layout.back, onVideo.projected)
        ops.gatedAdd(video, onVideo.projected, part(false, 2))
        if (textual) {
          // the text's outputs, averaged over the windows
          ops.scale(onText.attended, 1f / layout.spans.size, onText.attended)
          affine(onText.attended, block.text.out, onText.projected)
          ops.gatedAdd(words, onText.projected, part(true, 2))
        }

        // ---- MLP ----
        normModulate(video, onVideo, Some((part(false, 3), part(false, 4))))
        mlp(video, block.video, onVideo, part(false, 5))
        if (textual) {
          normModulate(words, onText, Some((part(true, 3), part(true, 4))))
          mlp(words, block.text, onText, part(true, 5))
        }
      }

      // ---- the output ----
      outNorm match {
        case Some(weight) =>
          ops.rmsNorm(video, weight, epsilon, 0f, onVideo.normed)
          ops.modulate(
            onVideo.normed,
            table.rows(c.layers * 12L + 1, 1).view(f),
            table.rows(c.layers * 12L, 1).view(f),
            onVideo.normed
          )
          affine(onVideo.normed, videoOut, out)
        case None => affine(video, videoOut, out)
      }
    } finally {
      live.foreach(ops.release)
      cache.foreach { kv =>
        ops.release(kv.keys)
        ops.release(kv.values)
      }
    }
  }

  def close(): Unit = {
    made.foreach(ops.release)
    source.close()
  }
}

object SeedVr2 {

  def open(ops: Ops, path: Path): SeedVr2 = {
    val source = WeightSource.open(ops, path)
    try new SeedVr2(ops, source)
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
