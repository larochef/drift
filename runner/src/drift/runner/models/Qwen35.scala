package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.*
import drift.runner.state.{KvCache, Sequence}
import drift.runner.tensor.*

import java.nio.file.Path

/** Qwen 3.5, 3.6 and 3.8, dense (GGUF `qwen35`, transformers `qwen3_5`: the
  * 27B) or with a mixture of experts (`qwen35moe`, `qwen3_5_moe`: 35B-A3B):
  * three gated-DeltaNet layers out of four, a full-attention layer with a
  * sigmoid output gate the fourth, and in every layer a dense SwiGLU or a
  * routed MoE with a gated shared expert, and after them, when the weights
  * carry it, the multi-token-prediction layer that drafts the tokens after the
  * next. With its vision tower (`QwenVision`) it sees images: their tokens'
  * embeddings are given, and they turn by mRoPE on their grid.
  */
final case class Qwen35Config(
    layers: Int,
    fullAttention: Seq[Boolean],
    vocabulary: Int,
    blocks: HybridShape
)

object Qwen35Config {

  def fromGguf(file: GgufFile, vocabulary: Int): Qwen35Config = {
    val architecture = file.architecture
    if (!Set("qwen35", "qwen35moe").contains(architecture))
      throw new FormatException(
        s"${file.source} is $architecture, not qwen35 or qwen35moe"
      )
    def int(key: String) = file.long(s"$architecture.$key").toInt
    val layers = int("block_count") - file.metadata
      .get(s"$architecture.nextn_predict_layers")
      .fold(0)(_ => int("nextn_predict_layers"))
    val interval = int("full_attention_interval")
    Qwen35Config(
      layers = layers,
      fullAttention = (0 until layers).map(i => (i + 1) % interval == 0),
      vocabulary = vocabulary,
      blocks = HybridShape(
        hidden = int("embedding_length"),
        heads = int("attention.head_count"),
        kvHeads = int("attention.head_count_kv"),
        headDimension = int("attention.key_length"),
        rotaryDimensions = int("rope.dimension_count"),
        ropeTheta = file.double(s"$architecture.rope.freq_base").toFloat,
        ropeSections = HybridShape.ggufRopeSections(file, architecture),
        rmsEpsilon = file
          .double(s"$architecture.attention.layer_norm_rms_epsilon")
          .toFloat,
        // llama.cpp stores the value heads tiled (see DeltaRule)
        deltaRule = DeltaRule(
          int("ssm.group_count"),
          int("ssm.time_step_rank"),
          int("ssm.state_size"),
          tiledHeads = true
        ),
        convTaps = int("ssm.conv_kernel"),
        feedForward =
          if (architecture == "qwen35")
            DenseFeedForward(int("feed_forward_length"))
          else
            RoutedExperts(
              int("expert_count"),
              int("expert_used_count"),
              int("expert_feed_forward_length"),
              int("expert_shared_feed_forward_length")
            ),
        normOffset = 0f,
        deltaGate = Activation.Silu
      )
    )
  }

  def fromHuggingFace(root: ModelConfig): Qwen35Config = {
    val config = root.text
    Qwen35Config(
      layers = config.int("num_hidden_layers"),
      fullAttention = config.strings("layer_types").map(_ == "full_attention"),
      vocabulary = config.int("vocab_size"),
      blocks = HuggingFaceConfigs.hybridShape(config, Activation.Silu)
    )
  }
}

final class Qwen35 private (
    ops: Ops,
    source: WeightSource,
    val config: Qwen35Config,
    gguf: Boolean,
    /** A draft model's GGUF (`--model-draft`) whose MTP layer drafts instead of
      * the model's own: a whole model's file or an MTP-only one.
      */
    draftSource: Option[WeightSource]
) extends Drafting {

  def vocabulary: Int = config.vocabulary

  private val shape = config.blocks
  private val weights = new HybridWeights(ops, source, gguf)
  private val blocks = new HybridBlocks(ops, shape)

  private def weight(ggufName: String, huggingFaceName: String): Tensor =
    weights(ggufName, huggingFaceName)

  final private class Layer(
      from: HybridWeights,
      g: String,
      h: String,
      fullAttention: Boolean
  ) {
    val attentionNorm: Tensor =
      from(g + "attn_norm.weight", h + "input_layernorm.weight")
    val mlpNorm: Tensor =
      from(
        g + "post_attention_norm.weight",
        h + "post_attention_layernorm.weight"
      )
    val linear: Option[LinearAttentionWeights] =
      Option.when(!fullAttention)(
        new LinearAttentionWeights(ops, from, shape, g, h + "linear_attn.")
      )
    val full: Option[FullAttentionWeights] =
      Option.when(fullAttention)(
        new FullAttentionWeights(from, g, h + "self_attn.")
      )
    val feedForward: FeedForwardWeights =
      FeedForwardWeights(from, shape, g, h + "mlp.")
  }

  /** The MTP layer (GGUF `blk.<n>` with its `nextn.` tensors, transformers'
    * `mtp.`) in `from`: the next token's embedding and a hidden state, each
    * normed, joined and projected, then one full-attention layer, whose output
    * normed gives the token after (through the model's output head) and the
    * hidden state a chained draft starts from.
    */
  final private class Prediction(from: HybridWeights, g: String) {
    val embeddingNorm: Tensor =
      from(g + "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight")
    val hiddenNorm: Tensor =
      from(g + "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight")
    val join: Tensor = from(g + "nextn.eh_proj.weight", "mtp.fc.weight")
    if (join.shape.dimensions != Seq(shape.hidden.toLong, 2L * shape.hidden))
      throw new FormatException(
        s"the MTP layer joins into ${join.shape}, not a hidden size of ${shape.hidden}"
      )
    val layer = new Layer(from, g, "mtp.layers.0.", fullAttention = true)
    val headNorm: Tensor =
      from(g + "nextn.shared_head_norm.weight", "mtp.norm.weight")
  }

  private val embedding =
    weight("token_embd.weight", "model.embed_tokens.weight")
  private val finalNorm = weight("output_norm.weight", "model.norm.weight")
  private val output =
    if (source.has(if (gguf) "output.weight" else "lm_head.weight"))
      weight("output.weight", "lm_head.weight")
    else embedding
  private val stack = (0 until config.layers).map(i =>
    new Layer(
      weights,
      s"blk.$i.",
      s"model.layers.$i.",
      config.fullAttention(i)
    )
  )
  private val draftWeights =
    draftSource.map(new HybridWeights(ops, _, gguf = true))
  private val prediction = draftWeights match {
    case Some(from) =>
      val JoinName = """(blk\.\d+\.)nextn\.eh_proj\.weight""".r
      val g = from.source.names
        .collectFirst { case JoinName(prefix) => prefix }
        .getOrElse(
          throw new FormatException("the draft model has no MTP layer")
        )
      Some(new Prediction(from, g))
    case None =>
      Option.when(
        source.has(
          if (gguf) s"blk.${config.layers}.nextn.eh_proj.weight"
          else "mtp.fc.weight"
        )
      )(new Prediction(weights, s"blk.${config.layers}."))
  }

  def drafts: Boolean = prediction.isDefined

  /** Which cache (full-attention layers) or states (the others) each layer
    * uses.
    */
  private val attentionIndex =
    config.fullAttention.scanLeft(0)((n, full) => if (full) n + 1 else n)
  private val recurrentIndex =
    config.fullAttention.scanLeft(0)((n, full) => if (full) n else n + 1)

  def newSequence(context: Int, pageSize: Int): Sequence =
    new Sequence(
      ops,
      context,
      pageSize,
      // the MTP layer's cache last
      config.fullAttention.count(identity) + (if (drafts) 1 else 0),
      shape.kvHeads,
      shape.headDimension,
      config.fullAttention
        .filterNot(identity)
        .flatMap(_ => shape.recurrentShapes)
    )

  /** Activations for up to `capacity` tokens and logits for `logitRows`,
    * allocated once and reused.
    */
  final private class Workspace(val capacity: Int, val logitRows: Int)
      extends SizedWorkspace {
    private val buffers = scala.collection.mutable.ArrayBuffer.empty[Tensor]
    private def allocate(dtype: DType, dimensions: Long*): Tensor = {
      val tensor =
        ops.allocate(dtype, Shape(capacity.toLong +: dimensions.toVector))
      buffers += tensor
      tensor
    }
    private val h = shape.hidden.toLong
    val blocks = new BlockBuffers(ops, shape, capacity)
    val x: Tensor = allocate(DType.F32, h)
    val normed: Tensor = allocate(DType.F32, h)
    val projected: Tensor = allocate(DType.F32, h)
    val joined: Tensor = allocate(DType.F32, 2 * h)
    val tokens: Tensor = allocate(DType.I32)
    val logits: Tensor = {
      val tensor = ops.allocate(
        DType.F32,
        Shape.of(logitRows, config.vocabulary)
      )
      buffers += tensor
      tensor
    }
    def release(): Unit = {
      blocks.release()
      buffers.foreach(ops.release)
    }
  }

  private val mainWorkspace = new WorkspaceSlot(new Workspace(_, _))
  private val draftWorkspace = new WorkspaceSlot(new Workspace(_, _))

  private var lastHidden = Option.empty[Tensor]
  private var lastDraftHidden = Option.empty[Tensor]

  def hiddenSize: Int = shape.hidden

  /** Where slots `start …` turn; the MTP layer's slots are the model's. */
  private def rotaryPositions(sequence: Sequence, start: Int, tokens: Int) =
    sequence.positions(start, tokens, shape.ropeSections.positionsPerToken)

  def hidden: Tensor =
    lastHidden.getOrElse(throw new IllegalStateException("nothing ran yet"))

  def draftHidden: Tensor =
    lastDraftHidden.getOrElse(
      throw new IllegalStateException("nothing drafted yet")
    )

  def forward(
      ids: Array[Int],
      start: Int,
      sequence: Sequence,
      allLogits: Boolean,
      givenRows: Seq[GivenRows] = Nil
  ): Tensor =
    run(ids, start, sequence, allLogits, keepHistory = false, givenRows)

  def verify(ids: Array[Int], start: Int, sequence: Sequence): Tensor =
    run(ids, start, sequence, allLogits = true, keepHistory = true, Nil)

  private def run(
      ids: Array[Int],
      start: Int,
      sequence: Sequence,
      allLogits: Boolean,
      keepHistory: Boolean,
      givenRows: Seq[GivenRows]
  ): Tensor = {
    val tokens = ids.length
    require(tokens > 0, "no tokens")
    val w = mainWorkspace(tokens, allLogits)
    val t = tokens.toLong
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (epsilon, offset) = (shape.rmsEpsilon, shape.normOffset)
    val history = Option.when(keepHistory)(sequence.history(tokens))
    ops.writeInts(rows(w.tokens), ids)
    ops.writeInts(
      w.blocks.positions(t),
      rotaryPositions(sequence, start, tokens)
    )
    val (x, normed) = (rows(w.x), rows(w.normed))
    ops.embedding(embedding, rows(w.tokens), x)
    givenRows.foreach(g => ops.copy(g.rows, x.rows(g.at, g.count)))
    // each layer ends with the next one's input norm, the last with the final
    ops.rmsNorm(x, stack.head.attentionNorm, epsilon, offset, normed)
    stack.zipWithIndex.foreach { (layer, i) =>
      val index = recurrentIndex(i)
      decoderLayer(
        layer,
        RowNorm(
          stack.lift(i + 1).fold(finalNorm)(_.attentionNorm),
          epsilon,
          offset,
          normed
        ),
        sequence,
        Option.when(layer.full.isDefined)(sequence.caches(attentionIndex(i))),
        Option.when(layer.linear.isDefined)(
          (
            sequence.recurrent(2 * index),
            sequence.recurrent(2 * index + 1),
            history.map(_(2 * index)),
            history.map(_(2 * index + 1))
          )
        ),
        start,
        t,
        w
      )
    }
    lastHidden = Some(normed)
    if (allLogits) {
      val logits = rows(w.logits)
      ops.linear(normed, output, logits)
      logits
    } else {
      val logits = w.logits.rows(0, 1)
      ops.linear(normed.rows(t - 1, 1), output, logits)
      logits
    }
  }

  /** One decoder layer on `w.x` in place, from its normed input in `w.normed`:
    * attention (full, through `cache`, or linear, through `states`: the
    * convolution's, the delta rule's and their histories), then the
    * feed-forward, each with its residual, and `next`'s norm of the result.
    */
  private def decoderLayer(
      layer: Layer,
      next: RowNorm,
      sequence: Sequence,
      cache: Option[KvCache],
      states: Option[(Tensor, Tensor, Option[Tensor], Option[Tensor])],
      start: Int,
      t: Long,
      w: Workspace
  ): Unit = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (epsilon, offset) = (shape.rmsEpsilon, shape.normOffset)
    val (x, normed, projected) = (rows(w.x), rows(w.normed), rows(w.projected))
    for {
      linear <- layer.linear
      (convState, deltaState, convHistory, deltaHistory) <- states
    } blocks.linearAttention(
      linear,
      normed,
      convState,
      deltaState,
      convHistory,
      deltaHistory,
      t,
      w.blocks,
      projected
    )
    for {
      full <- layer.full
      cache <- cache
    } blocks.fullAttention(
      full,
      normed,
      cache,
      sequence,
      start,
      t,
      w.blocks,
      projected
    )
    ops.addRmsNorm(
      x,
      projected,
      RowNorm(layer.mlpNorm, epsilon, offset, normed)
    )
    blocks.feedForward(
      layer.feedForward,
      normed,
      x,
      t,
      w.blocks,
      projected,
      next
    )
  }

  def draft(
      ids: Array[Int],
      hidden: Tensor,
      start: Int,
      sequence: Sequence
  ): Tensor = {
    val mtp = prediction.getOrElse(
      throw new IllegalStateException("these weights carry no MTP layer")
    )
    val tokens = ids.length
    require(tokens > 0, "no tokens")
    val w = draftWorkspace(tokens, allLogits = false)
    val t = tokens.toLong
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (epsilon, offset) = (shape.rmsEpsilon, shape.normOffset)
    ops.writeInts(rows(w.tokens), ids)
    ops.writeInts(
      w.blocks.positions(t),
      rotaryPositions(sequence, start, tokens)
    )
    val (x, normed, projected) = (rows(w.x), rows(w.normed), rows(w.projected))
    ops.embedding(embedding, rows(w.tokens), x)
    ops.rmsNorm(x, mtp.embeddingNorm, epsilon, offset, normed)
    ops.rmsNorm(hidden, mtp.hiddenNorm, epsilon, offset, projected)
    ops.joinHalves(normed, projected, rows(w.joined))
    ops.linear(rows(w.joined), mtp.join, x)
    ops.rmsNorm(x, mtp.layer.attentionNorm, epsilon, offset, normed)
    decoderLayer(
      mtp.layer,
      RowNorm(mtp.headNorm, epsilon, offset, normed),
      sequence,
      Some(sequence.caches.last),
      None,
      start,
      t,
      w
    )
    val kept = normed.rows(t - 1, 1)
    lastDraftHidden = Some(kept)
    val logits = w.logits.rows(0, 1)
    ops.linear(kept, output, logits)
    logits
  }

  def close(): Unit = {
    mainWorkspace.release()
    draftWorkspace.release()
    weights.release()
    draftWeights.foreach(_.release())
    source.close()
    draftSource.foreach(_.close())
  }
}

object Qwen35 {

  /** `draftModel`: a GGUF whose MTP layer drafts (a whole model's, of which
    * only that layer is copied to the device, or an MTP-only file), for weights
    * in GGUF.
    */
  def open(ops: Ops, path: Path, draftModel: Option[Path]): Qwen35 = {
    val source = WeightSource.open(ops, path)
    var draftSource = Option.empty[WeightSource]
    try
      source.gguf match {
        case Some(file) =>
          val vocabulary = source(
            "token_embd.weight"
          ).shape.dimensions.head.toInt
          draftSource = draftModel.map(WeightSource.copied(ops, _))
          new Qwen35(
            ops,
            source,
            Qwen35Config.fromGguf(file, vocabulary),
            gguf = true,
            draftSource
          )
        case None =>
          draftModel.foreach(draft =>
            throw new FormatException(
              s"a draft model's MTP layer ($draft) goes with GGUF weights, not $path"
            )
          )
          val config = source.config.getOrElse(
            throw new FormatException(s"no config.json beside $path")
          )
          new Qwen35(
            ops,
            source,
            Qwen35Config.fromHuggingFace(config),
            gguf = false,
            None
          )
      }
    catch {
      case error: Throwable =>
        source.close()
        draftSource.foreach(_.close())
        throw error
    }
  }
}
