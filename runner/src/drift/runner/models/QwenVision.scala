package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*
import drift.runner.vision.PreparedImage

import java.lang.foreign.MemorySegment
import java.nio.file.Path
import scala.collection.mutable

/** The vision tower of Qwen3-VL, Qwen 3.5/3.6 and Qwen 3.8 (llama.cpp's mmproj
  * `qwen3vl_merger`, transformers' `model.visual`): a ViT over 16-pixel patches
  * with a learned position table and 2-D rotary positions, then a merger that
  * joins each 2 × 2 window of patches into one token of the language model's
  * width. Qwen3-VL also taps some blocks' outputs through mergers of their own
  * (deepstack), which its language model adds to its first layers' outputs.
  */
final case class QwenVisionConfig(
    hidden: Int,
    heads: Int,
    intermediate: Int,
    depth: Int,
    patch: Int,
    merge: Int,
    /** The learned position table's side (`√ num_position_embeddings`). */
    positionSide: Int,
    /** The language model's width, the merger's output. */
    outputWidth: Int,
    epsilon: Float,
    /** The blocks whose outputs deepstack taps, in order. */
    deepstack: Seq[Int]
) {
  def headDimension: Int = hidden / heads

  /** Pixels per merged token's side: the image's sides are multiples of it. */
  def factor: Int = patch * merge
}

object QwenVisionConfig {

  /** An mmproj GGUF's `clip.vision.*` keys. */
  def fromGguf(file: drift.runner.formats.GgufFile): QwenVisionConfig = {
    val projector = file.string("clip.projector_type")
    if (projector != "qwen3vl_merger")
      throw new FormatException(
        s"${file.source} is a '$projector' projector; the runner reads qwen3vl_merger"
      )
    def int(key: String) = file.long(s"clip.vision.$key").toInt
    val taps = file.metadata.get("clip.vision.is_deepstack_layers") match {
      case Some(GgufValue.Array(values)) =>
        values.zipWithIndex.collect { case (GgufValue.Bool(true), i) => i }
      case _ => Vector.empty
    }
    QwenVisionConfig(
      hidden = int("embedding_length"),
      heads = int("attention.head_count"),
      intermediate = int("feed_forward_length"),
      depth = int("block_count"),
      patch = int("patch_size"),
      merge = int("spatial_merge_size"),
      positionSide = 0, // from the table's rows, at load
      outputWidth = int("projection_dim"),
      epsilon = file.double("clip.vision.attention.layer_norm_epsilon").toFloat,
      deepstack = taps
    )
  }

  /** Qwen3-VL's taps by the tower's depth (its configs'
    * `deepstack_visual_indexes`: 4B's 24 blocks, 8B's and 32B's 27).
    */
  private val DeepstackByDepth = Map(24 -> Seq(5, 11, 17), 27 -> Seq(8, 16, 24))

  /** Qwen3-VL's tower from transformers' names under `prefix` alone, no
    * configuration (MiniMax H3's text encoder, a GGUF with no metadata): the
    * shapes give the widths, the depth and the patch; Qwen3-VL's every size has
    * 16 heads, layer norms' ε 10⁻⁶ and the taps of its depth.
    */
  def fromWeights(
      source: WeightSource,
      prefix: String,
      deepstack: Option[Seq[Int]]
  ): QwenVisionConfig = {
    def dimensions(name: String) =
      source.shape(s"$prefix$name").dimensions.map(_.toInt)
    val hidden = dimensions("patch_embed.proj.bias").head
    val depth = Iterator
      .from(0)
      .takeWhile(i => source.has(s"${prefix}blocks.$i.norm1.weight"))
      .size
    val joined = dimensions("merger.linear_fc1.weight").last
    QwenVisionConfig(
      hidden = hidden,
      heads = 16,
      intermediate = dimensions("blocks.0.mlp.linear_fc1.weight").head,
      depth = depth,
      patch = dimensions("patch_embed.proj.weight").last,
      merge = math.round(math.sqrt(joined / hidden)).toInt,
      positionSide = 0, // from the table's rows, at load
      outputWidth = dimensions("merger.linear_fc2.weight").head,
      epsilon = 1e-6f,
      deepstack =
        if (!source.has(s"${prefix}deepstack_merger_list.0.norm.weight")) Nil
        else
          deepstack
            .orElse(DeepstackByDepth.get(depth))
            .getOrElse(
              throw new FormatException(
                s"a Qwen3-VL tower of $depth blocks: its deepstack taps are unknown"
              )
            )
    )
  }

  /** transformers' `vision_config`. */
  def fromHuggingFace(root: ModelConfig): QwenVisionConfig = {
    val config = root.section("vision_config")
    QwenVisionConfig(
      hidden = config.int("hidden_size"),
      heads = config.int("num_heads"),
      intermediate = config.int("intermediate_size"),
      depth = config.int("depth"),
      patch = config.int("patch_size"),
      merge = config.int("spatial_merge_size"),
      positionSide =
        math.round(math.sqrt(config.int("num_position_embeddings"))).toInt,
      outputWidth = config.int("out_hidden_size"),
      epsilon = 1e-6f,
      deepstack =
        if (config.has("deepstack_visual_indexes"))
          config.ints("deepstack_visual_indexes")
        else Nil
    )
  }
}

/** What the tower makes of one image: its tokens for the language model (F32
  * `[merged tokens, outputWidth]`) and, for Qwen3-VL, one tensor like it per
  * deepstack tap. The caller releases them.
  */
final case class VisionOutput(tokens: Tensor, deepstack: Seq[Tensor])

final class QwenVision private (
    ops: Ops,
    source: WeightSource,
    configured: QwenVisionConfig,
    gguf: Boolean,
    prefix: String
) extends AutoCloseable {

  private val weights = new HybridWeights(ops, source, gguf)
  private val v = prefix

  private val positionTable: Array[Float] =
    weights.hostFloats(
      weights.name("v.position_embd.weight", v + "pos_embed.weight")
    )

  val config: QwenVisionConfig = configured.copy(positionSide =
    math.round(math.sqrt(positionTable.length / configured.hidden)).toInt
  )

  private val (hidden, heads, d) =
    (config.hidden, config.heads, config.headDimension)

  /** The attention kernels' head width the heads are padded to, with zeros. */
  private val paddedHead = Attention.paddedHeadWidth(d)

  private def f32(values: Array[Float], dimensions: Long*): Tensor =
    weights.keep(ops.fromFloats(Shape(dimensions.toVector), values))

  /** The patch embedding as one linear layer over a still's patch (`c × p² + y
    * × p + x`): transformers' Conv3d spans two frames and llama.cpp splits it
    * in two kernels; a still is its own second frame, so they add up.
    */
  private val patchWeight: Tensor = {
    val pixels = 3 * config.patch * config.patch
    val summed = new Array[Float](hidden * pixels)
    if (gguf) {
      Seq("v.patch_embd.weight", "v.patch_embd.weight.1").foreach { name =>
        val kernel = weights.hostFloats(name)
        kernel.indices.foreach(i => summed(i) += kernel(i))
      }
    } else {
      val kernel = weights.hostFloats(v + "patch_embed.proj.weight")
      val frame = config.patch * config.patch
      for {
        o <- 0 until hidden
        c <- 0 until 3
        t <- 0 until 2
        p <- 0 until frame
      }
        summed(o * pixels + c * frame + p) +=
          kernel(((o * 3 + c) * 2 + t) * frame + p)
    }
    f32(summed, hidden, pixels)
  }

  /** The patch embedding over a pair of frames (`(c × 2 + t) × p² + y × p + x`,
    * transformers' video patches): the Conv3d's kernel as it is, or llama.cpp's
    * two kernels interleaved per channel.
    */
  private lazy val pairWeight: Tensor = {
    val frame = config.patch * config.patch
    val pixels = 3 * 2 * frame
    val kernel =
      if (gguf) {
        val Seq(first, second) =
          Seq("v.patch_embd.weight", "v.patch_embd.weight.1")
            .map(weights.hostFloats)
        Array.tabulate(hidden * pixels) { i =>
          val (o, c, t, p) = (
            i / pixels,
            i % pixels / (2 * frame),
            i % (2 * frame) / frame,
            i % frame
          )
          (if (t == 0) first else second) ((o * 3 + c) * frame + p)
        }
      } else weights.hostFloats(v + "patch_embed.proj.weight")
    f32(kernel, hidden, pixels)
  }
  private val patchBias =
    weights.floats(
      "v.patch_embd.bias",
      v + "patch_embed.proj.bias",
      Shape.of(hidden)
    )

  /** Rows of `tensor` (`[rows, K]`) regrouped per head and padded with zero
    * rows to `paddedHead`, stored as they were.
    */
  private def paddedRows(name: String, firstRow: Int): Tensor = {
    val tensor = source(name)
    val columns = tensor.shape.dimensions(1)
    val rowBytes = tensor.dtype.byteSize(columns).toInt
    val bytes = MemorySegment.ofArray(weights.hostBytes(name))
    val out = new Array[Byte](heads * paddedHead * rowBytes)
    for {
      h <- 0 until heads
      j <- 0 until d
    }
      MemorySegment.copy(
        bytes,
        (firstRow + h * d + j).toLong * rowBytes,
        MemorySegment.ofArray(out),
        (h * paddedHead + j).toLong * rowBytes,
        rowBytes
      )
    weights.upload(tensor.dtype, Shape.of(heads * paddedHead, columns), out)
  }

  /** The output projection's columns padded per head (F32). */
  private def paddedColumns(name: String): Tensor = {
    val values = weights.hostFloats(name)
    val out = new Array[Float](hidden * heads * paddedHead)
    for {
      r <- 0 until hidden
      h <- 0 until heads
    }
      System.arraycopy(
        values,
        r * hidden + h * d,
        out,
        (r * heads + h) * paddedHead,
        d
      )
    f32(out, hidden, heads * paddedHead)
  }

  private def paddedBias(values: Array[Float], first: Int): Tensor = {
    val out = new Array[Float](heads * paddedHead)
    (0 until heads).foreach(h =>
      System.arraycopy(values, first + h * d, out, h * paddedHead, d)
    )
    f32(out, heads * paddedHead)
  }

  final private case class Affine(weight: Tensor, bias: Tensor)

  private def affine(g: String, h: String, outputs: Int): Affine =
    Affine(
      weights(g + ".weight", h + ".weight"),
      weights.floats(g + ".bias", h + ".bias", Shape.of(outputs))
    )

  final private class Norm(g: String, h: String, width: Int) {
    val weight: Tensor =
      weights.floats(g + ".weight", h + ".weight", Shape.of(width))
    val bias: Tensor = weights.floats(g + ".bias", h + ".bias", Shape.of(width))
  }

  final private class Block(i: Int) {
    private val g = s"v.blk.$i."
    private val h = s"${v}blocks.$i."
    val norm1 = new Norm(g + "ln1", h + "norm1", hidden)
    val norm2 = new Norm(g + "ln2", h + "norm2", hidden)
    private val qkv = weights.name(g + "attn_qkv.weight", h + "attn.qkv.weight")
    private val qkvBias =
      weights.hostFloats(weights.name(g + "attn_qkv.bias", h + "attn.qkv.bias"))
    val (q, k, value) = (
      paddedRows(qkv, 0),
      paddedRows(qkv, hidden),
      paddedRows(qkv, 2 * hidden)
    )
    val (qBias, kBias, valueBias) = (
      paddedBias(qkvBias, 0),
      paddedBias(qkvBias, hidden),
      paddedBias(qkvBias, 2 * hidden)
    )
    val out: Tensor =
      paddedColumns(weights.name(g + "attn_out.weight", h + "attn.proj.weight"))
    val outBias: Tensor =
      weights.floats(
        g + "attn_out.bias",
        h + "attn.proj.bias",
        Shape.of(hidden)
      )
    val up: Affine =
      affine(g + "ffn_up", h + "mlp.linear_fc1", config.intermediate)
    val down: Affine = affine(g + "ffn_down", h + "mlp.linear_fc2", hidden)
  }

  private val blocks = (0 until config.depth).map(new Block(_))

  /** A merger: 2 × 2 windows of patches joined into rows of `hidden × 4`,
    * normed before the join (the final one) or after it (deepstack's), then two
    * layers with an exact GELU between.
    */
  final private class Merger(
      val norm: Norm,
      val normAfterJoin: Boolean,
      val first: Affine,
      val second: Affine
  )

  private val joined = hidden * config.merge * config.merge

  private val merger = new Merger(
    new Norm("v.post_ln", v + "merger.norm", hidden),
    normAfterJoin = false,
    affine("mm.0", v + "merger.linear_fc1", joined),
    affine("mm.2", v + "merger.linear_fc2", config.outputWidth)
  )

  private val deepstackMergers = config.deepstack.zipWithIndex.map {
    (layer, index) =>
      val g = s"v.deepstack.$layer"
      val h = s"${v}deepstack_merger_list.$index"
      layer -> new Merger(
        new Norm(g + ".norm", h + ".norm", joined),
        normAfterJoin = true,
        affine(g + ".fc1", h + ".linear_fc1", joined),
        affine(g + ".fc2", h + ".linear_fc2", config.outputWidth)
      )
  }.toMap

  private val rope = Rope(
    10000f,
    d,
    RopeLayout.Neox,
    RopeSections.Axes(Seq(d / 4, d / 4))
  )
  private val attention =
    Attention((1 / math.sqrt(d)).toFloat, causal = false, None, None, None)

  /** The learned positions of a `gridHeight × gridWidth` grid of patches, in
    * merge-window order: the table (a square of `positionSide`) sampled
    * bilinearly at evenly spaced points, as transformers'
    * `fast_pos_embed_interpolate`.
    */
  private def positionRows(gridHeight: Int, gridWidth: Int): Array[Float] = {
    val side = config.positionSide
    def spaced(n: Int): Array[Float] =
      Array.tabulate(n)(i =>
        if (n == 1) 0f else (i * (side - 1).toFloat / (n - 1)).toFloat
      )
    val (ys, xs) = (spaced(gridHeight), spaced(gridWidth))
    val m = config.merge
    val out = new Array[Float](gridHeight * gridWidth * hidden)
    var row = 0
    for {
      by <- 0 until gridHeight / m
      bx <- 0 until gridWidth / m
      iy <- 0 until m
      ix <- 0 until m
    } {
      val (y, x) = (ys(by * m + iy), xs(bx * m + ix))
      val (y0, x0) = (y.toInt, x.toInt)
      val (y1, x1) = (math.min(y0 + 1, side - 1), math.min(x0 + 1, side - 1))
      val (dy, dx) = (y - y0, x - x0)
      val corners = Seq(
        (y0 * side + x0, (1 - dy) * (1 - dx)),
        (y0 * side + x1, (1 - dy) * dx),
        (y1 * side + x0, dy * (1 - dx)),
        (y1 * side + x1, dy * dx)
      )
      val base = row * hidden
      corners.foreach { (index, weight) =>
        val from = index * hidden
        var c = 0
        while (c < hidden) {
          out(base + c) += positionTable(from + c) * weight
          c += 1
        }
      }
      row += 1
    }
    out
  }

  /** The rotary positions of the patches in merge-window order: row then column
    * of each, in patches.
    */
  private def rotaryPositions(gridHeight: Int, gridWidth: Int): Array[Int] = {
    val m = config.merge
    val order = for {
      by <- 0 until gridHeight / m
      bx <- 0 until gridWidth / m
      iy <- 0 until m
      ix <- 0 until m
    } yield (by * m + iy, bx * m + ix)
    order.map(_._1).toArray ++ order.map(_._2)
  }

  /** Encodes one image: `image.patches` (`[patches, 3 × patch²]`, in
    * merge-window order) into the language model's tokens.
    */
  def encode(image: PreparedImage): VisionOutput = {
    val patches = image.gridHeight * image.gridWidth
    val p = patches.toLong
    val held = mutable.ArrayBuffer.empty[Tensor]
    def allocate(dimensions: Long*): Tensor = {
      val tensor = ops.allocate(DType.F32, Shape(dimensions.toVector))
      held += tensor
      tensor
    }
    val cache: KvCache =
      ops.allocateCache(1, (patches + 15) / 16 * 16, heads, paddedHead)
    val results = mutable.ArrayBuffer.empty[Tensor]
    try {
      require(
        image.frames == 1 || image.frames == 2,
        s"a patch of ${image.frames} frames for a tower of two-frame patches"
      )
      val pixels = ops.fromFloats(
        Shape.of(p, 3L * image.frames * config.patch * config.patch),
        image.patches
      )
      held += pixels
      val pageTable = ops.fromInts(Shape.of(1), Array(0))
      held += pageTable
      val positions = ops.fromInts(
        Shape.of(2, p),
        rotaryPositions(image.gridHeight, image.gridWidth)
      )
      held += positions
      val x = allocate(p, hidden)
      val normed = allocate(p, hidden)
      val projected = allocate(p, hidden)
      val width = heads.toLong * paddedHead
      val (q, k, value) =
        (allocate(p, width), allocate(p, width), allocate(p, width))
      val (rotatedQ, rotatedK) = (allocate(p, width), allocate(p, width))
      val attended = allocate(p, width)
      val inner = allocate(p, config.intermediate)
      ops.linear(
        pixels,
        if (image.frames == 2) pairWeight else patchWeight,
        x
      )
      ops.addRow(x, patchBias, x)
      val table = ops.fromFloats(
        Shape.of(p, hidden),
        positionRows(image.gridHeight, image.gridWidth)
      )
      held += table
      ops.add(x, table, x)
      def headed(t: Tensor) = t.view(p, heads, paddedHead)
      blocks.zipWithIndex.foreach { (b, i) =>
        ops.layerNorm(
          x,
          Some(b.norm1.weight),
          Some(b.norm1.bias),
          config.epsilon,
          normed
        )
        ops.linears(normed, Seq(b.q, b.k, b.value), Seq(q, k, value))
        ops.addRow(q, b.qBias, q)
        ops.addRow(k, b.kBias, k)
        ops.addRow(value, b.valueBias, value)
        ops.rope(headed(q), positions, rope, headed(rotatedQ))
        ops.rope(headed(k), positions, rope, headed(rotatedK))
        ops.cacheWrite(headed(rotatedK), headed(value), cache, pageTable, 0)
        ops.attention(
          headed(rotatedQ),
          cache,
          pageTable,
          0,
          patches,
          attention,
          headed(attended)
        )
        ops.linear(attended, b.out, projected)
        ops.addRow(projected, b.outBias, projected)
        ops.add(x, projected, x)
        ops.layerNorm(
          x,
          Some(b.norm2.weight),
          Some(b.norm2.bias),
          config.epsilon,
          normed
        )
        ops.linear(normed, b.up.weight, inner)
        ops.addRow(inner, b.up.bias, inner)
        ops.activation(Activation.GeluTanh, inner, inner)
        ops.linear(inner, b.down.weight, projected)
        ops.addRow(projected, b.down.bias, projected)
        ops.add(x, projected, x)
        deepstackMergers.get(i).foreach(m => results += merge(m, x, patches))
      }
      val tokens = merge(merger, x, patches)
      VisionOutput(tokens, results.toSeq)
    } catch {
      case error: Throwable =>
        results.foreach(ops.release)
        throw error
    } finally {
      held.foreach(ops.release)
      ops.release(cache.keys)
      ops.release(cache.values)
    }
  }

  /** `x` (`[patches, hidden]`) through a merger into a new tensor. */
  private def merge(m: Merger, x: Tensor, patches: Int): Tensor = {
    val tokens = (patches / (config.merge * config.merge)).toLong
    val normed = ops.allocate(DType.F32, x.shape)
    val inner = ops.allocate(DType.F32, Shape.of(tokens, joined))
    val out = ops.allocate(DType.F32, Shape.of(tokens, config.outputWidth))
    try {
      if (m.normAfterJoin)
        ops.layerNorm(
          x.view(tokens, joined),
          Some(m.norm.weight),
          Some(m.norm.bias),
          config.epsilon,
          normed.view(tokens, joined)
        )
      else
        ops.layerNorm(
          x,
          Some(m.norm.weight),
          Some(m.norm.bias),
          config.epsilon,
          normed
        )
      ops.linear(normed.view(tokens, joined), m.first.weight, inner)
      ops.addRow(inner, m.first.bias, inner)
      ops.activation(Activation.GeluErf, inner, inner)
      ops.linear(inner, m.second.weight, out)
      ops.addRow(out, m.second.bias, out)
      out
    } catch {
      case error: Throwable =>
        ops.release(out)
        throw error
    } finally {
      ops.release(normed)
      ops.release(inner)
    }
  }

  def close(): Unit = {
    weights.release()
    source.close()
  }
}

object QwenVision {

  /** transformers' names for the tower. */
  private val DefaultPrefix = "model.visual."

  /** The tower that shares `path` with its language model under transformers'
    * names at `prefix`, with no configuration (MiniMax H3's text encoder:
    * `visual.` in a GGUF without metadata); its deepstack taps those of
    * Qwen3-VL's depth unless given.
    */
  def fromWeights(
      ops: Ops,
      path: Path,
      prefix: String,
      deepstack: Option[Seq[Int]] = None
  ): QwenVision = {
    val source = WeightSource.open(ops, path)
    try
      new QwenVision(
        ops,
        source,
        QwenVisionConfig.fromWeights(source, prefix, deepstack),
        gguf = false,
        prefix
      )
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }

  /** An mmproj GGUF, or transformers' weights whose `config.json` has a
    * `vision_config` (the tower under `model.visual.`).
    */
  def open(ops: Ops, path: Path): QwenVision = {
    val source = WeightSource.open(ops, path)
    try
      source.gguf match {
        case Some(file) =>
          new QwenVision(
            ops,
            source,
            QwenVisionConfig.fromGguf(file),
            gguf = true,
            DefaultPrefix
          )
        case None =>
          val config = source.config.getOrElse(
            throw new FormatException(s"no config.json beside $path")
          )
          new QwenVision(
            ops,
            source,
            QwenVisionConfig.fromHuggingFace(config),
            gguf = false,
            DefaultPrefix
          )
      }
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
