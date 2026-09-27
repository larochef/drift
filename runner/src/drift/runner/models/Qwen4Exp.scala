package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.*
import drift.runner.state.Sequence
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** Qwen 3.8 Flash Next (GGUF `qwen4exp`, transformers `qwen4_exp`): Qwen
  * 3.5/3.6's hybrid blocks (with sigmoid-gated DeltaNet norms) over a residual
  * of `streams` copies of the hidden state. Each block reads the streams
  * through a hyper-connection (a per-stream norm, a low-rank sigmoid mix, and
  * per-stream write weights) instead of a norm, and the head reads them through
  * one more, with no final norm. Before the `pleLayer`, the n-gram embedding
  * (PLE) adds rows hashed from each token and its predecessors, gated per
  * stream and passed through a dilated causal convolution. Its full attention
  * is Qwen Sparse Attention, dense up to its indexer's budget (2048 tokens),
  * which is all the runner does so far. Text only.
  */
final case class Qwen4ExpConfig(
    layers: Int,
    fullAttention: Seq[Boolean],
    vocabulary: Int,
    blocks: HybridShape,
    streams: Int,
    lowRank: Int,
    /** The layer the n-gram embedding comes before (0-based). */
    pleLayer: Option[Int],
    hash: NgramHash,
    /** The n-gram embedding's row width (its heads' rows join into `hidden`).
      */
    pleHeadDimension: Int,
    pleConvTaps: Int
)

object Qwen4ExpConfig {

  def fromGguf(file: GgufFile, vocabulary: Int): Qwen4ExpConfig = {
    if (file.architecture != "qwen4exp")
      throw new FormatException(
        s"${file.source} is ${file.architecture}, not qwen4exp"
      )
    def key(name: String) = s"qwen4exp.$name"
    def int(name: String) = file.long(key(name)).toInt
    val layers = int("block_count") - file.metadata
      .get(key("nextn_predict_layers"))
      .fold(0)(_ => int("nextn_predict_layers"))
    val interval = int("full_attention_interval")
    Qwen4ExpConfig(
      layers = layers,
      fullAttention = (0 until layers).map(i => (i + 1) % interval == 0),
      vocabulary = vocabulary,
      blocks = HybridShape(
        hidden = int("embedding_length"),
        heads = int("attention.head_count"),
        kvHeads = int("attention.head_count_kv"),
        headDimension = int("attention.key_length"),
        rotaryDimensions = int("rope.dimension_count"),
        ropeTheta = file.double(key("rope.freq_base")).toFloat,
        ropeSections = HybridShape.ggufRopeSections(file, "qwen4exp"),
        rmsEpsilon =
          file.double(key("attention.layer_norm_rms_epsilon")).toFloat,
        // llama.cpp stores the value heads tiled (see DeltaRule)
        deltaRule = DeltaRule(
          int("ssm.group_count"),
          int("ssm.time_step_rank"),
          int("ssm.state_size"),
          tiledHeads = true
        ),
        convTaps = int("ssm.conv_kernel"),
        feedForward = RoutedExperts(
          int("expert_count"),
          int("expert_used_count"),
          int("expert_feed_forward_length"),
          int("expert_shared_feed_forward_length")
        ),
        normOffset = 0f,
        deltaGate = Activation.Sigmoid
      ),
      streams = int("hyper_connection.count"),
      lowRank = int("hyper_connection.low_rank"),
      pleLayer = file.longs(key("ple.layers")).headOption.map(_.toInt),
      hash = NgramHash(
        ngramSize = int("ple.ngram_size"),
        headsPerNgram = int("ple.heads_per_ngram"),
        eos = int("ple.eos_token_id"),
        multipliers = file.longs(key("ple.layer_multipliers")).toVector,
        sizes = file.longs(key("ple.head_vocab_sizes")).toVector,
        offsets = file.longs(key("ple.head_offsets")).toVector
      ),
      pleHeadDimension = int("embedding_length_per_layer_input"),
      pleConvTaps = int("ple.conv_kernel")
    )
  }

  def fromHuggingFace(root: ModelConfig): Qwen4ExpConfig = {
    val config = root.text
    def intOr(name: String, default: Int) =
      if (config.has(name)) config.int(name) else default
    val layers = config.int("num_hidden_layers")
    val shape = HuggingFaceConfigs.hybridShape(config, Activation.Sigmoid)
    // 1-based in config.json; each PLE layer takes its own hash (only one here)
    val pleLayers =
      if (config.has("ple_layer_ids")) config.ints("ple_layer_ids").map(_ - 1)
      else Nil
    if (pleLayers.size > 1)
      throw new FormatException(
        s"${config.source}: n-gram embeddings before ${pleLayers.size} layers"
      )
    val ngramSize = intOr("ngram_size", 3)
    val headsPerNgram = intOr("heads_per_ngram", 8)
    val vocabulary = config.int("vocab_size")
    val eos = config.json("eos_token_id") match {
      case ujson.Arr(values) => values.head.num.toInt
      case value             => value.num.toInt
    }
    val pleDimension = intOr("ple_embed_dim", shape.hidden)
    Qwen4ExpConfig(
      layers = layers,
      fullAttention =
        config.strings("layer_types").map(_ != "linear_attention"),
      vocabulary = vocabulary,
      blocks = shape,
      streams = config.int("hc_count"),
      lowRank = config.int("hc_lowrank"),
      pleLayer = pleLayers.headOption,
      hash = NgramTables.hash(
        ngramSize,
        headsPerNgram,
        eos,
        vocabulary,
        intOr("seed", 1234),
        intOr("ngram_vocab_size_base", 20000000),
        pleIndex = 0
      ),
      pleHeadDimension = pleDimension / ((ngramSize - 1) * headsPerNgram),
      pleConvTaps = intOr("ple_conv_kernel_size", 4)
    )
  }
}

/** The n-gram hash's constants as transformers derives them (its
  * `Qwen4ExpTextNGramEmbedding`); a GGUF stores them.
  */
object NgramTables {

  private val Gamma = 0x9e3779b97f4a7c15L

  private def splitmix64(value: Long): Long = {
    var v = value + Gamma
    v = (v ^ (v >>> 30)) * 0xbf58476d1ce4e5b9L
    v = (v ^ (v >>> 27)) * 0x94d049bb133111ebL
    v ^ (v >>> 31)
  }

  private def isPrime(value: Long): Boolean =
    value >= 2 && (value == 2 || value % 2 != 0) &&
      (3L to math.sqrt(value.toDouble).toLong + 1 by 2)
        .forall(d => d >= value || value % d != 0)

  /** The `count`-th prime after `start`. */
  private def primeAfter(start: Long, count: Int): Long = {
    var prime = start
    (1 to count).foreach { _ =>
      prime += 1
      while (!isPrime(prime)) prime += 1
    }
    prime
  }

  def hash(
      ngramSize: Int,
      headsPerNgram: Int,
      eos: Int,
      vocabulary: Int,
      seed: Int,
      vocabularyBase: Int,
      pleIndex: Int
  ): NgramHash = {
    val half = math.max(1L, Long.MaxValue / math.max(vocabulary, 1) / 2)
    val base = seed.toLong + 10007L * pleIndex
    val multipliers = (0 until ngramSize).map { i =>
      2 * java.lang.Long.remainderUnsigned(
        splitmix64(base + Gamma * (i + 1)),
        half
      ) + 1
    }
    val heads = (ngramSize - 1) * headsPerNgram
    val sizes = (0 until heads).map(h =>
      primeAfter(vocabularyBase - 1L, pleIndex * heads + h + 1)
    )
    NgramHash(
      ngramSize,
      headsPerNgram,
      eos,
      multipliers.toVector,
      sizes.toVector,
      sizes.scanLeft(0L)(_ + _).init.toVector
    )
  }
}

final class Qwen4Exp private (
    ops: Ops,
    source: WeightSource,
    val config: Qwen4ExpConfig,
    gguf: Boolean,
    /** The MTP head's weights, a GGUF of their own (`--spec-draft-model`). */
    draftSource: Option[WeightSource]
) extends Drafting {

  def vocabulary: Int = config.vocabulary

  private val shape = config.blocks
  private val streams = config.streams
  private val wide = streams.toLong * shape.hidden
  private val weights = new HybridWeights(ops, source, gguf)
  private val blocks = new HybridBlocks(ops, shape)

  /** A hyper-connection (GGUF `g`, transformers `h`) in `from`: the per-stream
    * norm (its weights stored minus one when `normOffset` is 1), the low-rank
    * mix's down and up projections and, when it writes back, the per-stream
    * write weights.
    */
  final private class HyperConnection(
      from: HybridWeights,
      g: String,
      h: String,
      writes: Boolean,
      val normOffset: Float
  ) {
    val norm: Tensor =
      from(g + "_norm.weight", h + "hc_norm.weight").view(wide)
    val down: Tensor =
      from(g + "_down.weight", h + "input_mix_weight_down.weight")
    val up: Tensor = from(g + "_up.weight", h + "input_mix_weight_up.weight")
    val inject: Option[Tensor] = Option.when(writes)(
      from(g + "_inject.weight", h + "block_inject_weight.weight")
    )
  }

  private def trunkConnection(g: String, h: String, writes: Boolean) =
    new HyperConnection(weights, g, h, writes, shape.normOffset)

  final private class NgramEmbedding(g: String, h: String) {
    val table: Tensor =
      if (gguf) weights.source("per_layer_token_embd.weight")
      else {
        val shards = Iterator
          .from(0)
          .map(i => s"${h}ple_embedding.ngram_embedding.shard_$i.weight")
          .takeWhile(source.has)
          .toSeq
        weights.stacked(shards, join = true)
      }
    val key: Tensor = weights(g + "ple_key.weight", h + "key_proj.weight")
    val value: Tensor = weights(g + "ple_value.weight", h + "value_proj.weight")
    val keyNorm: Tensor =
      weights(g + "ple_norm_key.weight", h + "norm_key.weight").view(wide)
    val queryNorm: Tensor =
      weights(g + "ple_norm_query.weight", h + "norm_query.weight").view(wide)
    val convNorm: Tensor =
      weights(g + "ple_norm_conv.weight", h + "norm_conv.weight").view(wide)
    val conv: Tensor = weights.floats(
      g + "ple_conv1d.weight",
      h + "conv1d.weight",
      Shape.of(wide, config.pleConvTaps)
    )
  }

  final private class Layer(i: Int) {
    private val (g, h) = (s"blk.$i.", s"model.layers.$i.")
    val attentionMix =
      trunkConnection(g + "hc_attn", h + "attn_hyper_connection.", true)
    val expertsMix =
      trunkConnection(g + "hc_ffn", h + "mlp_hyper_connection.", true)
    val linear: Option[LinearAttentionWeights] =
      Option.when(!config.fullAttention(i))(
        new LinearAttentionWeights(ops, weights, shape, g, h + "linear_attn.")
      )
    val full: Option[FullAttentionWeights] =
      Option.when(config.fullAttention(i))(
        new FullAttentionWeights(weights, g, h + "self_attn.")
      )
    val experts = new ExpertWeights(weights, shape, g, h + "mlp.")
    val ngrams: Option[NgramEmbedding] =
      Option.when(config.pleLayer.contains(i))(
        new NgramEmbedding(g, h + "ple.")
      )
  }

  private val embedding =
    weights("token_embd.weight", "model.embed_tokens.weight")
  private val output = weights("output.weight", "lm_head.weight")
  private val outputMix =
    trunkConnection("output_hc", "model.hyper_connection_mixer.", false)
  private val stack = (0 until config.layers).map(new Layer(_))

  /** The MTP head, from its own file with GGUF names (llama.cpp's PR #28243):
    * `blk.<n>.nextn.` norms and join, one full-attention layer with its
    * hyper-connections and experts, and its own stream mixer
    * (`nextn.hc_head_*`, or `output_hc_*` in heads exported alone). The
    * embedding and output head are the model's own. Its norms are folded to
    * `1 + w`.
    */
  final private class Prediction(from: HybridWeights) {
    private val JoinName = """(blk\.\d+\.)nextn\.eh_proj\.weight""".r
    private val g = from.source.names
      .collectFirst { case JoinName(prefix) => prefix }
      .getOrElse(throw new FormatException("the draft model has no MTP layer"))
    private def connection(name: String, writes: Boolean) =
      new HyperConnection(from, name, name, writes, 0f)
    val embeddingNorm: Tensor = from(g + "nextn.enorm.weight", "")
    val hiddenNorm: Tensor = from(g + "nextn.hnorm.weight", "").view(wide)
    val join: Tensor = from(g + "nextn.eh_proj.weight", "")
    val attentionMix = connection(g + "hc_attn", writes = true)
    val expertsMix = connection(g + "hc_ffn", writes = true)
    val full = new FullAttentionWeights(from, g, "")
    val experts = new ExpertWeights(from, shape, g, "")
    val headMix =
      if (from.source.has(g + "nextn.hc_head_norm.weight"))
        connection(g + "nextn.hc_head", writes = false)
      else connection("output_hc", writes = false)
  }

  private val draftWeights =
    draftSource.map(new HybridWeights(ops, _, gguf = true))
  private val prediction = draftWeights.map(new Prediction(_))

  /** The MTP head's blocks: its norms are folded whatever the model's are. */
  private val draftBlocks =
    new HybridBlocks(ops, shape.copy(normOffset = 0f))

  def drafts: Boolean = prediction.isDefined

  def hiddenSize: Int = wide.toInt

  /** Which cache (full-attention layers) or states (the others) each layer
    * uses; the n-gram embedding's two states come after the layers'.
    */
  private val attentionIndex =
    config.fullAttention.scanLeft(0)((n, full) => if (full) n + 1 else n)
  private val recurrentIndex =
    config.fullAttention.scanLeft(0)((n, full) => if (full) n else n + 1)
  private val ngramStates = 2 * config.fullAttention.count(!_)

  def newSequence(context: Int, pageSize: Int): Sequence =
    new Sequence(
      ops,
      context,
      pageSize,
      // the MTP head's cache last
      config.fullAttention.count(identity) + (if (drafts) 1 else 0),
      shape.kvHeads,
      shape.headDimension,
      config.fullAttention
        .filterNot(identity)
        .flatMap(_ => shape.recurrentShapes) ++
        config.pleLayer.toSeq.flatMap(_ =>
          Seq(
            Shape.of((config.pleConvTaps - 1) * config.hash.ngramSize, wide),
            Shape.of(config.hash.kept)
          )
        )
    )

  /** Activations for up to `capacity` tokens and logits for `logitRows`,
    * allocated once and reused.
    */
  final private class Workspace(val capacity: Int, val logitRows: Int)
      extends SizedWorkspace {
    private val buffers = mutable.ArrayBuffer.empty[Tensor]
    private def allocate(dtype: DType, dimensions: Long*): Tensor = {
      val tensor =
        ops.allocate(dtype, Shape(capacity.toLong +: dimensions.toVector))
      buffers += tensor
      tensor
    }
    private def f32(dimensions: Long*) = allocate(DType.F32, dimensions*)
    private val hidden = shape.hidden.toLong
    val blocks = new BlockBuffers(ops, shape, capacity)
    val embedded: Tensor = f32(hidden)
    val residual: Tensor = f32(wide)
    val normed: Tensor = f32(wide)
    val gates: Tensor = f32(wide)
    val low: Tensor = f32(config.lowRank)
    val writes: Tensor = f32(streams)
    val mixed: Tensor = f32(hidden)
    val projected: Tensor = f32(hidden)
    val tokens: Tensor = allocate(DType.I32)
    // zero write logits: the embedding copied into every stream
    val copies: Tensor = {
      val tensor = f32(streams)
      ops.zero(tensor)
      tensor
    }
    // the n-gram embedding
    val ngramRows: Tensor = allocate(DType.I32, config.hash.heads)
    val ngrams: Tensor = f32(hidden)
    val keys: Tensor = f32(wide)
    val values: Tensor = f32(hidden)
    val queries: Tensor = f32(wide)
    val gated: Tensor = f32(wide)
    val convolved: Tensor = f32(wide)
    // the MTP head: each stream's embedding and hidden halves side by side
    val joined: Tensor = f32(2 * wide)
    val logits: Tensor = {
      val tensor =
        ops.allocate(DType.F32, Shape.of(logitRows, config.vocabulary))
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

  /** The last run's streams after its last layer, `[tokens, streams × hidden]`:
    * what the MTP head reads.
    */
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
    val count = ids.length
    require(count > 0, "no tokens")
    val w = mainWorkspace(count, allLogits)
    val t = count.toLong
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val history = Option.when(keepHistory)(sequence.history(count))
    ops.writeInts(rows(w.tokens), ids)
    ops.writeInts(
      w.blocks.positions(t),
      sequence.positions(start, count, shape.ropeSections.positionsPerToken)
    )
    val residual = rows(w.residual)
    // every stream starts as the token's embedding
    ops.embedding(embedding, rows(w.tokens), rows(w.embedded))
    givenRows.foreach(g => ops.copy(g.rows, w.embedded.rows(g.at, g.count)))
    ops.zero(residual)
    ops.streamsCombine(residual, rows(w.embedded), rows(w.copies))
    stack.zipWithIndex.foreach { (layer, i) =>
      layer.ngrams.foreach(ngrams =>
        ngramEmbedding(
          ngrams,
          sequence.recurrent(ngramStates),
          sequence.recurrent(ngramStates + 1),
          history.map(_(ngramStates)),
          history.map(_(ngramStates + 1)),
          t,
          w
        )
      )
      val index = recurrentIndex(i)
      val (mixed, projected) = (rows(w.mixed), rows(w.projected))
      mix(layer.attentionMix, t, w)
      for (linear <- layer.linear)
        blocks.linearAttention(
          linear,
          mixed,
          sequence.recurrent(2 * index),
          sequence.recurrent(2 * index + 1),
          history.map(_(2 * index)),
          history.map(_(2 * index + 1)),
          t,
          w.blocks,
          projected
        )
      for (full <- layer.full)
        blocks.fullAttention(
          full,
          mixed,
          sequence.caches(attentionIndex(i)),
          sequence,
          start,
          t,
          w.blocks,
          projected
        )
      ops.streamsCombine(residual, projected, rows(w.writes))
      mix(layer.expertsMix, t, w)
      blocks.experts(layer.experts, mixed, None, t, w.blocks, projected)
      ops.streamsCombine(residual, projected, rows(w.writes))
    }
    lastHidden = Some(residual)
    mix(outputMix, t, w)
    val mixed = rows(w.mixed)
    if (allLogits) {
      val logits = rows(w.logits)
      ops.linear(mixed, output, logits)
      logits
    } else {
      val logits = w.logits.rows(0, 1)
      ops.linear(mixed.rows(t - 1, 1), output, logits)
      logits
    }
  }

  /** A hyper-connection's read of `w.residual` into `w.mixed`: each stream
    * normed, a low-rank sigmoid gate per stream and channel, their mean; and
    * when it writes back, the logits of each stream's write weight into
    * `w.writes` (`inject · normed`; `streamsCombine` takes `2 sigmoid(· /
    * streams)`).
    */
  private def mix(connection: HyperConnection, t: Long, w: Workspace): Unit = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (normed, low, gates) = (rows(w.normed), rows(w.low), rows(w.gates))
    ops.groupRmsNorm(
      rows(w.residual),
      connection.norm,
      streams,
      shape.rmsEpsilon,
      connection.normOffset,
      normed
    )
    ops.linear(normed, connection.down, low)
    ops.scaledActivation(Activation.Silu, 1f / streams, low, low)
    ops.linear(low, connection.up, gates)
    ops.streamsMix(gates, normed, streams, rows(w.mixed))
    connection.inject.foreach(ops.linear(normed, _, rows(w.writes)))
  }

  /** The n-gram embedding added to every stream of `w.residual`: the hashed
    * rows' key gates the rows' value per stream (against the stream normed),
    * and the gated value plus its dilated causal convolution join the stream.
    */
  private def ngramEmbedding(
      ngrams: NgramEmbedding,
      convState: Tensor,
      tokenState: Tensor,
      convHistory: Option[Tensor],
      tokenHistory: Option[Tensor],
      t: Long,
      w: Workspace
  ): Unit = {
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val (epsilon, offset) = (shape.rmsEpsilon, shape.normOffset)
    val heads = config.hash.heads.toLong
    val ngramRows = rows(w.ngramRows)
    ops.ngramRows(
      rows(w.tokens),
      tokenState,
      config.hash,
      tokenHistory,
      ngramRows
    )
    val embedded = rows(w.ngrams)
    ops.embedding(
      ngrams.table,
      ngramRows.view(t * heads),
      embedded.view(t * heads, config.pleHeadDimension)
    )
    val (keys, values, queries) =
      (rows(w.keys), rows(w.values), rows(w.queries))
    ops.linears(embedded, Seq(ngrams.key, ngrams.value), Seq(keys, values))
    ops.groupRmsNorm(keys, ngrams.keyNorm, streams, epsilon, offset, keys)
    val residual = rows(w.residual)
    ops.groupRmsNorm(
      residual,
      ngrams.queryNorm,
      streams,
      epsilon,
      offset,
      queries
    )
    val gated = rows(w.gated)
    ops.pleGate(keys, queries, values, gated)
    val normed = rows(w.normed)
    ops.groupRmsNorm(gated, ngrams.convNorm, streams, epsilon, offset, normed)
    val convolved = rows(w.convolved)
    ops.causalConv(
      normed,
      ngrams.conv,
      config.hash.ngramSize,
      convState,
      convHistory,
      convolved
    )
    ops.add(residual, gated, residual)
    ops.add(residual, convolved, residual)
  }

  /** The MTP head on `hidden` (`[t, streams × hidden]`, the model's streams at
    * positions `start …`) and the tokens after them (`ids`): each stream's
    * normed hidden state joins the token's normed embedding through `eh_proj`,
    * then one full-attention layer (its cache the sequence's last), and its own
    * mixer and the output head give the logits after the last token.
    */
  def draft(
      ids: Array[Int],
      hidden: Tensor,
      start: Int,
      sequence: Sequence
  ): Tensor = {
    val mtp = prediction.getOrElse(
      throw new IllegalStateException(
        "no MTP head: pass its GGUF as the draft model"
      )
    )
    val count = ids.length
    require(count > 0, "no tokens")
    val w = draftWorkspace(count, allLogits = false)
    val t = count.toLong
    def rows(tensor: Tensor) = tensor.rows(0, t)
    val epsilon = shape.rmsEpsilon
    ops.writeInts(rows(w.tokens), ids)
    ops.writeInts(
      w.blocks.positions(t),
      sequence.positions(start, count, shape.ropeSections.positionsPerToken)
    )
    val (embedded, residual, normed) =
      (rows(w.embedded), rows(w.residual), rows(w.normed))
    ops.embedding(embedding, rows(w.tokens), embedded)
    ops.rmsNorm(embedded, mtp.embeddingNorm, epsilon, 0f, embedded)
    ops.zero(residual)
    ops.streamsCombine(residual, embedded, rows(w.copies))
    // per stream, as llama.cpp's draft graph; one norm over the whole row
    // (gufo's reading) accepted fewer drafts: 53% against 62% on prose
    ops.groupRmsNorm(hidden, mtp.hiddenNorm, streams, epsilon, 0f, normed)
    val perStream = t * streams
    val joined = w.joined.rows(0, t).view(perStream, 2L * shape.hidden)
    ops.joinHalves(
      residual.view(perStream, shape.hidden),
      normed.view(perStream, shape.hidden),
      joined
    )
    ops.linear(joined, mtp.join, residual.view(perStream, shape.hidden))
    val (mixed, projected) = (rows(w.mixed), rows(w.projected))
    mix(mtp.attentionMix, t, w)
    draftBlocks.fullAttention(
      mtp.full,
      mixed,
      sequence.caches.last,
      sequence,
      start,
      t,
      w.blocks,
      projected
    )
    ops.streamsCombine(residual, projected, rows(w.writes))
    mix(mtp.expertsMix, t, w)
    draftBlocks.experts(mtp.experts, mixed, None, t, w.blocks, projected)
    ops.streamsCombine(residual, projected, rows(w.writes))
    lastDraftHidden = Some(residual.rows(t - 1, 1))
    mix(mtp.headMix, t, w)
    val logits = w.logits.rows(0, 1)
    ops.linear(mixed.rows(t - 1, 1), output, logits)
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

object Qwen4Exp {

  def open(ops: Ops, path: Path, draftModel: Option[Path]): Qwen4Exp = {
    val source = WeightSource.open(ops, path)
    val draftSource = draftModel.map(WeightSource.open(ops, _))
    try
      source.gguf match {
        case Some(file) =>
          val vocabulary =
            source("token_embd.weight").shape.dimensions.head.toInt
          new Qwen4Exp(
            ops,
            source,
            Qwen4ExpConfig.fromGguf(file, vocabulary),
            gguf = true,
            draftSource
          )
        case None =>
          val config = source.config.getOrElse(
            throw new FormatException(s"no config.json beside $path")
          )
          new Qwen4Exp(
            ops,
            source,
            Qwen4ExpConfig.fromHuggingFace(config),
            gguf = false,
            draftSource
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
