package drift.runner.models

import drift.runner.diffusion.LoraUpdates
import drift.runner.formats.*
import drift.runner.ops.*
import drift.runner.state.Sequence
import drift.runner.tensor.*

/** What sets a family of dense decoders apart from the others:
  *   - `normOffset`: RMS norms take `w + normOffset` (Gemma stores `w − 1`);
  *   - `headNorms`: an RMS norm on each query and key head before RoPE;
  *   - `sandwichNorms`: each sublayer's output normed too before its residual;
  *   - `activation`: the MLP's gate;
  *   - `scaledEmbedding`: embeddings × √hidden;
  *   - `attentionSoftcap`: attention logits `c · tanh(s / c)`;
  *   - `window`: a sliding window on the even layers;
  *   - `outputSoftcap`: the same on the logits.
  */
final case class DenseStyle(
    normOffset: Float,
    headNorms: Boolean,
    sandwichNorms: Boolean,
    activation: Activation,
    scaledEmbedding: Boolean,
    attentionSoftcap: Option[Float],
    window: Option[Int],
    outputSoftcap: Option[Float]
)

object DenseStyle {
  val Qwen3: DenseStyle =
    DenseStyle(0f, true, false, Activation.Silu, false, None, None, None)

  /** Mistral (Llama's shape): no head norms, a SwiGLU MLP. */
  val Mistral: DenseStyle =
    DenseStyle(0f, false, false, Activation.Silu, false, None, None, None)

  def gemma2(
      attentionSoftcap: Float,
      window: Int,
      outputSoftcap: Float
  ): DenseStyle = DenseStyle(
    1f,
    false,
    true,
    Activation.GeluTanh,
    true,
    Some(attentionSoftcap),
    Some(window),
    Some(outputSoftcap)
  )
}

/** A dense decoder's sizes. `queryScale` multiplies the attention logits;
  * `ropeLayout` is how the stored q and k weights pair their rotated values
  * (llama.cpp permutes Llama's and Mistral's into interleaved pairs).
  */
final case class DenseConfig(
    layers: Int,
    hidden: Int,
    intermediate: Int,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    ropeTheta: Float,
    /** Plain RoPE, or mRoPE's sections (Qwen3-VL). */
    ropeSections: RopeSections,
    ropeLayout: RopeLayout,
    rmsEpsilon: Float,
    vocabulary: Int,
    queryScale: Float,
    style: DenseStyle
)

/** Qwen 3, dense (0.6B to 32B; the text encoder of Z-Image and Flux.2 Klein),
  * and Qwen3-VL's language model (the text encoder of Krea 2 and Qwen Image
  * 2.1; GGUF `qwen3vl`, whose mRoPE axes all carry the same position for text,
  * which is plain RoPE): pre-norm layers of GQA attention, RMS norm on each
  * query and key head before a NeoX RoPE, and a SwiGLU MLP.
  */
object Qwen3Config {

  /** The GGUF architectures read as Qwen 3. */
  val GgufArchitectures: Set[String] = Set("qwen3", "qwen3vl")

  def fromGguf(file: GgufFile, vocabulary: Int): DenseConfig = {
    val architecture = file.architecture
    if (!GgufArchitectures.contains(architecture))
      throw new FormatException(
        s"${file.source} is $architecture, not ${GgufArchitectures.mkString(" or ")}"
      )
    def int(key: String) = file.long(s"$architecture.$key").toInt
    DenseConfig(
      layers = int("block_count"),
      hidden = int("embedding_length"),
      intermediate = int("feed_forward_length"),
      heads = int("attention.head_count"),
      kvHeads = int("attention.head_count_kv"),
      headDimension = int("attention.key_length"),
      ropeTheta = file.double(s"$architecture.rope.freq_base").toFloat,
      ropeSections = HybridShape.ggufRopeSections(file, architecture),
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon =
        file.double(s"$architecture.attention.layer_norm_rms_epsilon").toFloat,
      vocabulary = vocabulary,
      queryScale = (1 / math.sqrt(int("attention.key_length"))).toFloat,
      style = DenseStyle.Qwen3
    )
  }

  /** Qwen3-VL's language model from its weights alone, transformers' names
    * without a `config.json` (MiniMax H3's text encoder, a GGUF with no
    * metadata): RoPE's θ is Qwen3-VL's 5 000 000, its interleaved mRoPE of 24,
    * 20 and 20 pairs (text turns alike on every axis; images by their grid),
    * RMS norms' ε 10⁻⁶.
    */
  def fromWeights(source: WeightSource, prefix: String): DenseConfig = {
    def dimensions(name: String) =
      source.shape(s"$prefix$name").dimensions.map(_.toInt)
    val headDimension = dimensions("layers.0.self_attn.q_norm.weight").head
    DenseConfig(
      layers = Iterator
        .from(0)
        .takeWhile(i =>
          source.has(s"${prefix}layers.$i.input_layernorm.weight")
        )
        .size,
      hidden = dimensions("embed_tokens.weight").last,
      intermediate = dimensions("layers.0.mlp.gate_proj.weight").head,
      heads =
        dimensions("layers.0.self_attn.q_proj.weight").head / headDimension,
      kvHeads =
        dimensions("layers.0.self_attn.k_proj.weight").head / headDimension,
      headDimension = headDimension,
      ropeTheta = 5000000f,
      ropeSections = RopeSections.Interleaved(24, 20, 20),
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = 1e-6f,
      vocabulary = dimensions("embed_tokens.weight").head,
      queryScale = (1 / math.sqrt(headDimension)).toFloat,
      style = DenseStyle.Qwen3
    )
  }

  def fromHuggingFace(config: ModelConfig): DenseConfig =
    DenseConfig(
      layers = config.int("num_hidden_layers"),
      hidden = config.int("hidden_size"),
      intermediate = config.int("intermediate_size"),
      heads = config.int("num_attention_heads"),
      kvHeads = config.int("num_key_value_heads"),
      headDimension = config.int("head_dim"),
      ropeTheta = HuggingFaceConfigs.ropeTheta(config).toFloat,
      ropeSections = HuggingFaceConfigs.ropeSections(config),
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = config.double("rms_norm_eps").toFloat,
      vocabulary = config.int("vocab_size"),
      queryScale = (1 / math.sqrt(config.int("head_dim"))).toFloat,
      style = DenseStyle.Qwen3
    )
}

/** Gemma 2 (the text encoder of PiD: Gemma 2 2B): Qwen 3's shape with norms of
  * `1 + w` before and after each sublayer, no head norms, a GeGLU MLP,
  * embeddings × √hidden, softcapped attention (queries scaled by
  * `query_pre_attn_scalar^−½`) and a sliding window on the even layers.
  */
object Gemma2Config {

  def fromHuggingFace(config: ModelConfig): DenseConfig =
    DenseConfig(
      layers = config.int("num_hidden_layers"),
      hidden = config.int("hidden_size"),
      intermediate = config.int("intermediate_size"),
      heads = config.int("num_attention_heads"),
      kvHeads = config.int("num_key_value_heads"),
      headDimension = config.int("head_dim"),
      ropeTheta = HuggingFaceConfigs.ropeTheta(config).toFloat,
      ropeSections = RopeSections.Single,
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = config.double("rms_norm_eps").toFloat,
      vocabulary = config.int("vocab_size"),
      queryScale =
        (1 / math.sqrt(config.double("query_pre_attn_scalar"))).toFloat,
      style = DenseStyle.gemma2(
        config.double("attn_logit_softcapping").toFloat,
        config.int("sliding_window"),
        config.double("final_logit_softcapping").toFloat
      )
    )

  /** Gemma 2 2B's configuration, for a single file without its `config.json`
    * (ComfyUI's): 26 layers of 2304, 8 query heads over 4 of 256.
    */
  def gemma2b(source: WeightSource, prefix: String): DenseConfig = {
    val Seq(vocabulary, hidden) =
      source(s"${prefix}embed_tokens.weight").shape.dimensions.map(_.toInt)
    if (hidden != 2304)
      throw new FormatException(
        s"a Gemma 2 of width $hidden without its config.json: only 2B (2304) is known"
      )
    DenseConfig(
      layers = Iterator
        .from(0)
        .takeWhile(i =>
          source.has(s"${prefix}layers.$i.input_layernorm.weight")
        )
        .size,
      hidden = hidden,
      intermediate = 9216,
      heads = 8,
      kvHeads = 4,
      headDimension = 256,
      ropeTheta = 10000f,
      ropeSections = RopeSections.Single,
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = 1e-6f,
      vocabulary = vocabulary,
      queryScale = (1 / math.sqrt(256)).toFloat,
      style = DenseStyle.gemma2(50f, 4096, 30f)
    )
  }
}

/** Mistral Small 3.x's language model (the text encoder of FLUX.2 [dev]: 40
  * layers of 5120, 32 query heads over 8 of 128, RoPE θ 10⁹): Llama's shape,
  * pre-norm GQA attention without head norms and a SwiGLU MLP. Its GGUFs are
  * llama.cpp's `llama`, whose q and k weights are permuted to rotate
  * interleaved pairs; transformers' weights rotate halves.
  */
object MistralConfig {

  /** The GGUF architectures read as Mistral. */
  val GgufArchitectures: Set[String] = Set("llama", "mistral3")

  def fromGguf(file: GgufFile, vocabulary: Int): DenseConfig = {
    val architecture = file.architecture
    if (!GgufArchitectures.contains(architecture))
      throw new FormatException(
        s"${file.source} is $architecture, not ${GgufArchitectures.mkString(" or ")}"
      )
    def int(key: String) = file.long(s"$architecture.$key").toInt
    val heads = int("attention.head_count")
    val headDimension =
      if (file.metadata.contains(s"$architecture.attention.key_length"))
        int("attention.key_length")
      else int("embedding_length") / heads
    DenseConfig(
      layers = int("block_count"),
      hidden = int("embedding_length"),
      intermediate = int("feed_forward_length"),
      heads = heads,
      kvHeads = int("attention.head_count_kv"),
      headDimension = headDimension,
      ropeTheta = file.double(s"$architecture.rope.freq_base").toFloat,
      ropeSections = RopeSections.Single,
      ropeLayout = RopeLayout.Interleaved,
      rmsEpsilon =
        file.double(s"$architecture.attention.layer_norm_rms_epsilon").toFloat,
      vocabulary = vocabulary,
      queryScale = (1 / math.sqrt(headDimension)).toFloat,
      style = DenseStyle.Mistral
    )
  }

  def fromHuggingFace(config: ModelConfig): DenseConfig = {
    val heads = config.int("num_attention_heads")
    val headDimension =
      if (config.has("head_dim")) config.int("head_dim")
      else config.int("hidden_size") / heads
    DenseConfig(
      layers = config.int("num_hidden_layers"),
      hidden = config.int("hidden_size"),
      intermediate = config.int("intermediate_size"),
      heads = heads,
      kvHeads = config.int("num_key_value_heads"),
      headDimension = headDimension,
      ropeTheta = HuggingFaceConfigs.ropeTheta(config).toFloat,
      ropeSections = RopeSections.Single,
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = config.double("rms_norm_eps").toFloat,
      vocabulary = config.int("vocab_size"),
      queryScale = (1 / math.sqrt(headDimension)).toFloat,
      style = DenseStyle.Mistral
    )
  }
}

/** The names of a dense decoder's weights in a GGUF (llama.cpp's) or in
  * transformers' safetensors. `mlp_norm` is the norm before the MLP;
  * `post_attention_norm` and `post_mlp_norm` are Gemma's sandwich norms.
  */
final private case class DenseNames(
    embedding: String,
    finalNorm: String,
    output: String,
    layer: (Int, String) => String
)

private object DenseNames {
  val Gguf: DenseNames = DenseNames(
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
    (i, part) =>
      s"blk.$i." + (part match {
        case "attention_norm" => "attn_norm"
        case "q"              => "attn_q"
        case "k"              => "attn_k"
        case "v"              => "attn_v"
        case "q_norm"         => "attn_q_norm"
        case "k_norm"         => "attn_k_norm"
        case "o"              => "attn_output"
        case "mlp_norm"       => "ffn_norm"
        case "gate"           => "ffn_gate"
        case "up"             => "ffn_up"
        case "down"           => "ffn_down"
      }) + ".weight"
  )
  val HuggingFace: DenseNames = huggingFace("model.")

  /** transformers' names under `prefix` (`model.language_model.` in a
    * Qwen3-VL's whole checkpoint).
    */
  def huggingFace(prefix: String): DenseNames = DenseNames(
    s"${prefix}embed_tokens.weight",
    s"${prefix}norm.weight",
    "lm_head.weight",
    (i, part) =>
      s"${prefix}layers.$i." + (part match {
        case "attention_norm" => "input_layernorm"
        case "q"              => "self_attn.q_proj"
        case "k"              => "self_attn.k_proj"
        case "v"              => "self_attn.v_proj"
        case "q_norm"         => "self_attn.q_norm"
        case "k_norm"         => "self_attn.k_norm"
        case "o"              => "self_attn.o_proj"
        case "mlp_norm"       => "post_attention_layernorm"
        case "gate"           => "mlp.gate_proj"
        case "up"             => "mlp.up_proj"
        case "down"           => "mlp.down_proj"
      }) + ".weight"
  )

  /** Gemma 2's, under `prefix` (`model.` in a causal LM's file, none in a bare
    * `Gemma2Model`'s).
    */
  def gemma2(prefix: String): DenseNames = DenseNames(
    s"${prefix}embed_tokens.weight",
    s"${prefix}norm.weight",
    "lm_head.weight",
    (i, part) =>
      s"${prefix}layers.$i." + (part match {
        case "attention_norm"      => "input_layernorm"
        case "q"                   => "self_attn.q_proj"
        case "k"                   => "self_attn.k_proj"
        case "v"                   => "self_attn.v_proj"
        case "o"                   => "self_attn.o_proj"
        case "post_attention_norm" => "post_attention_layernorm"
        case "mlp_norm"            => "pre_feedforward_layernorm"
        case "post_mlp_norm"       => "post_feedforward_layernorm"
        case "gate"                => "mlp.gate_proj"
        case "up"                  => "mlp.up_proj"
        case "down"                => "mlp.down_proj"
        case other                 => other
      }) + ".weight"
  )
}

/** A dense decoder (`DenseStyle`: Qwen 3, Gemma 2, Mistral). With `loras`, each
  * layer's linears take the active updates of their weight's name (less
  * `.weight`, as `loraSites` lists them) at run time.
  */
final class DenseDecoder private[models] (
    ops: Ops,
    source: WeightSource,
    val config: DenseConfig,
    names: DenseNames,
    loras: Option[LoraUpdates] = None
) extends CausalModel {

  private val style = config.style

  def vocabulary: Int = config.vocabulary

  private val LinearParts = Seq("q", "k", "v", "o", "gate", "up", "down")

  private def site(i: Int, part: String) =
    names.layer(i, part).stripSuffix(".weight")

  /** The names LoRA updates reach the layers' linears by. */
  def loraSites: Set[String] =
    (0 until config.layers).flatMap(i => LinearParts.map(site(i, _))).toSet

  /** Layer `i`'s linear `part`, its LoRA updates added. */
  private def linear(
      x: Tensor,
      i: Int,
      part: String,
      weight: Tensor,
      out: Tensor
  ): Unit = {
    ops.linear(x, weight, out)
    loras.foreach(_(x, site(i, part), out))
  }

  def newSequence(context: Int, pageSize: Int): Sequence =
    new Sequence(
      ops,
      context,
      pageSize,
      config.layers,
      config.kvHeads,
      config.headDimension,
      Nil
    )

  /** Norm weights stored otherwise (Gemma's BF16), converted to F32 once. */
  private val converted = scala.collection.mutable.ArrayBuffer.empty[Tensor]
  private def floats(tensor: Tensor): Tensor =
    if (tensor.dtype == DType.F32) tensor
    else {
      val copy = ops.allocate(DType.F32, tensor.shape)
      ops.convert(tensor, copy)
      converted += copy
      copy
    }

  final private class Layer(i: Int) {
    private def weight(part: String) = source(names.layer(i, part))
    private def norm(part: String) = floats(weight(part))
    val attentionNorm: Tensor = norm("attention_norm")
    val q: Tensor = weight("q")
    val k: Tensor = weight("k")
    val v: Tensor = weight("v")
    val qNorm: Option[Tensor] =
      Option.when(style.headNorms)(norm("q_norm"))
    val kNorm: Option[Tensor] =
      Option.when(style.headNorms)(norm("k_norm"))
    val o: Tensor = weight("o")
    val postAttentionNorm: Option[Tensor] =
      Option.when(style.sandwichNorms)(norm("post_attention_norm"))
    val mlpNorm: Tensor = norm("mlp_norm")
    val postMlpNorm: Option[Tensor] =
      Option.when(style.sandwichNorms)(norm("post_mlp_norm"))
    val gate: Tensor = weight("gate")
    val up: Tensor = weight("up")
    val down: Tensor = weight("down")
  }

  private val embedding = source(names.embedding)
  // a text encoder's file may be cut before it (MiniMax H3's Qwen3-VL)
  private lazy val finalNorm = floats(source(names.finalNorm))
  // small Qwen 3 models tie the output head to the embeddings
  private val output =
    if (source.has(names.output)) source(names.output) else embedding
  private val stack = (0 until config.layers).map(new Layer(_))

  private val rope = Rope(
    config.ropeTheta,
    config.headDimension,
    config.ropeLayout,
    config.ropeSections
  )

  /** Layer `i`'s attention: the window on even layers when the style has one.
    */
  private def attention(i: Int) =
    Attention(
      config.queryScale,
      causal = true,
      style.window.filter(_ => i % 2 == 0),
      style.attentionSoftcap,
      None
    )

  /** Activations for up to `capacity` tokens, allocated once and reused. */
  final private class Workspace(val capacity: Int, withAllLogits: Boolean) {
    private val (h, q, kv, i) =
      (
        config.hidden,
        config.heads * config.headDimension,
        config.kvHeads * config.headDimension,
        config.intermediate
      )
    private def buffer(columns: Long) =
      ops.allocate(DType.F32, Shape.of(capacity, columns))
    val x: Tensor = buffer(h)
    val normed: Tensor = buffer(h)
    val queries: Tensor = buffer(q)
    val keys: Tensor = buffer(kv)
    val values: Tensor = buffer(kv)
    val rotatedQueries: Tensor = buffer(q)
    val rotatedKeys: Tensor = buffer(kv)
    val attended: Tensor = buffer(q)
    val projected: Tensor = buffer(h)
    val gate: Tensor = buffer(i)
    val up: Tensor = buffer(i)
    val logits: Tensor = ops.allocate(
      DType.F32,
      Shape.of(if (withAllLogits) capacity else 1, config.vocabulary)
    )
    val ids: Tensor = ops.allocate(DType.I32, Shape.of(capacity))
    val positions: Tensor = ops.allocate(DType.I32, Shape.of(3L * capacity))
    val all: Boolean = withAllLogits
    def release(): Unit =
      Seq(
        x,
        normed,
        queries,
        keys,
        values,
        rotatedQueries,
        rotatedKeys,
        attended,
        projected,
        gate,
        up,
        logits,
        ids,
        positions
      )
        .foreach(ops.release)
  }

  private var workspace = Option.empty[Workspace]

  private def workspaceFor(tokens: Int, allLogits: Boolean): Workspace =
    workspace
      .filter(w => w.capacity >= tokens && (w.all || !allLogits))
      .getOrElse {
        workspace.foreach(_.release())
        val created = new Workspace(
          math.max(tokens, workspace.fold(1)(_.capacity)),
          allLogits
        )
        workspace = Some(created)
        created
      }

  /** No recurrent state to keep: a forward with every logit. */
  def verify(ids: Array[Int], start: Int, sequence: Sequence): Tensor =
    forward(ids, start, sequence, allLogits = true)

  def forward(
      ids: Array[Int],
      start: Int,
      sequence: Sequence,
      allLogits: Boolean,
      givenRows: Seq[GivenRows] = Nil
  ): Tensor = {
    require(givenRows.isEmpty, "this model reads text only")
    if (style.outputSoftcap.isDefined)
      throw new UnsupportedOperationException(
        "Gemma 2 runs as a text encoder only (its logits' softcap is not implemented)"
      )
    val tokens = ids.length
    val w = workspaceFor(tokens, allLogits)
    val x = run(ids, start, sequence, w, config.layers, tokens, _ => ())
    val normed = w.normed.rows(0, tokens)
    ops.rmsNorm(x, finalNorm, config.rmsEpsilon, style.normOffset, normed)
    if (allLogits) {
      val logits = w.logits.rows(0, tokens)
      ops.linear(normed, output, logits)
      logits
    } else {
      val logits = w.logits.rows(0, 1)
      ops.linear(normed.rows(tokens - 1, 1), output, logits)
      logits
    }
  }

  /** As a text encoder: the residual stream (no final norm) of a whole prompt
    * after each layer count in `taps` (0 is the embedding, `k` the output of
    * layer `k`; transformers' `hidden_states[k]` but for the last, which it
    * norms), into `out` (`[taps × tokens, hidden]`, tap-major). The layers
    * after the last tap are not run.
    *
    * `padding` rows of `padId` are appended (`[taps × (tokens + padding),
    * hidden]`), masked as keys the way a padded batch's attention mask masks
    * them: the prompt never sees them, and each sees the prompt alone.
    */
  def encode(
      ids: Array[Int],
      taps: Seq[Int],
      out: Tensor,
      padding: Int = 0,
      padId: Int = 0,
      images: Option[SeenImages] = None
  ): Unit = {
    val prompt = ids.length
    val tokens = prompt + padding
    require(
      out.shape == Shape.of(taps.size.toLong * tokens, config.hidden),
      s"encode: out ${out.shape} for ${taps.size} taps of $tokens tokens"
    )
    val sequence = newSequence(tokens, 64)
    try {
      sequence.reserve(tokens)
      images.foreach(seen => sequence.place(0, seen.positions))
      val w = workspaceFor(tokens, allLogits = false)
      def keep(done: Int, x: Tensor): Unit =
        taps.zipWithIndex
          .filter(_._1 == done)
          .foreach((_, i) => ops.copy(x, out.rows(i.toLong * tokens, tokens)))
      run(
        ids ++ Array.fill(padding)(padId),
        0,
        sequence,
        w,
        taps.max,
        prompt,
        done => keep(done, w.x.rows(0, tokens)),
        images
      )
    } finally sequence.close()
  }

  /** As a text encoder, transformers' `last_hidden_state`: every layer, then
    * the final norm, into `out` (`[tokens + padding, hidden]`), the padding as
    * `encode` masks it.
    */
  def lastHiddenState(
      ids: Array[Int],
      out: Tensor,
      padding: Int = 0,
      padId: Int = 0
  ): Unit = {
    val prompt = ids.length
    val tokens = prompt + padding
    require(
      out.shape == Shape.of(tokens, config.hidden),
      s"lastHiddenState: out ${out.shape} for $tokens tokens"
    )
    val sequence = newSequence(tokens, 64)
    try {
      sequence.reserve(tokens)
      val w = workspaceFor(tokens, allLogits = false)
      val x = run(
        ids ++ Array.fill(padding)(padId),
        0,
        sequence,
        w,
        config.layers,
        prompt,
        _ => ()
      )
      ops.rmsNorm(x, finalNorm, config.rmsEpsilon, style.normOffset, out)
    } finally sequence.close()
  }

  /** A prompt's tokens into `sequence` from slot 0, causally, no logits: the
    * prefix later rows attend to (`attendAll`).
    */
  def prefill(ids: Array[Int], sequence: Sequence): Unit = {
    sequence.reserve(ids.length)
    run(
      ids,
      0,
      sequence,
      workspaceFor(ids.length, allLogits = false),
      config.layers,
      ids.length,
      _ => ()
    )
  }

  /** Rows given as embeddings (`x`, `[tokens, hidden]`) at slots `start` on,
    * each attending to every slot before `start + tokens` — the cached prefix
    * and one another, both ways (HiDream O1's generated tokens) — then the
    * final norm, into `out` (like `x`). The rows' keys stay in the sequence
    * until the next call writes over them.
    */
  def attendAll(
      x: Tensor,
      start: Int,
      sequence: Sequence,
      out: Tensor
  ): Unit = {
    val tokens = x.shape.dimensions.head.toInt
    sequence.reserve(start + tokens)
    val w = workspaceFor(tokens, allLogits = false)
    val hidden = run(
      Array.emptyIntArray,
      start,
      sequence,
      w,
      config.layers,
      tokens,
      _ => (),
      embedded = Some(x),
      bidirectional = true
    )
    ops.rmsNorm(hidden, finalNorm, config.rmsEpsilon, style.normOffset, out)
  }

  /** The embedding then the first `layers` layers, in place in `w.x`;
    * `after(k)` once layer `k` (1-based) is done, and `after(0)` before any.
    * Only the first `keys` tokens are cached and attend causally; the rest
    * attend to those alone. The tokens turn by the sequence's positions; the
    * `images`' rows replace their tokens' embeddings, and their deepstack rows
    * join the first layers' outputs. `embedded` rows stand for the tokens'
    * embeddings (`ids` then empty); `bidirectional` caches every token and has
    * each attend to all the cached slots, none causally.
    */
  private def run(
      ids: Array[Int],
      start: Int,
      sequence: Sequence,
      w: Workspace,
      layers: Int,
      keys: Int,
      after: Int => Unit,
      images: Option[SeenImages] = None,
      embedded: Option[Tensor] = None,
      bidirectional: Boolean = false
  ): Tensor = {
    val tokens = embedded.fold(ids.length)(_.shape.dimensions.head.toInt)
    require(tokens > 0, "no tokens")
    require(keys > 0 && keys <= tokens, s"$keys keys of $tokens tokens")
    val (caches, pageTable) = (sequence.caches, sequence.pageTable)
    val (heads, kvHeads, d) =
      (config.heads, config.kvHeads, config.headDimension)
    val idsTensor = w.ids.rows(0, tokens)
    val axes = config.ropeSections.positionsPerToken
    val positions =
      if (axes == 1) w.positions.prefix(tokens)
      else w.positions.prefix(axes, tokens)
    if (embedded.isEmpty) ops.writeInts(idsTensor, ids)
    ops.writeInts(
      positions.view(axes.toLong * tokens),
      sequence.positions(start, tokens, axes)
    )
    locally {
      def rows(t: Tensor) = t.rows(0, tokens)
      val x = rows(w.x)
      val normed = rows(w.normed)
      embedded match {
        case Some(rows) => ops.copy(rows, x)
        case None       => ops.embedding(embedding, idsTensor, x)
      }
      images.foreach(
        _.rows.foreach(g => ops.copy(g.rows, x.rows(g.at, g.count)))
      )
      if (style.scaledEmbedding)
        ops.scale(x, math.sqrt(config.hidden).toFloat, x)
      after(0)
      stack.zip(caches).take(layers).zipWithIndex.foreach {
        case ((layer, cache), index) =>
          ops.rmsNorm(
            x,
            layer.attentionNorm,
            config.rmsEpsilon,
            style.normOffset,
            normed
          )
          val (q, k, v) = (rows(w.queries), rows(w.keys), rows(w.values))
          linear(normed, index, "q", layer.q, q)
          linear(normed, index, "k", layer.k, k)
          linear(normed, index, "v", layer.v, v)
          // RMS norm per head, in place (each row is read whole before it is written)
          layer.qNorm.foreach(weight =>
            ops.rmsNorm(
              q.view(tokens.toLong * heads, d),
              weight,
              config.rmsEpsilon,
              style.normOffset,
              q.view(tokens.toLong * heads, d)
            )
          )
          layer.kNorm.foreach(weight =>
            ops.rmsNorm(
              k.view(tokens.toLong * kvHeads, d),
              weight,
              config.rmsEpsilon,
              style.normOffset,
              k.view(tokens.toLong * kvHeads, d)
            )
          )
          val rotatedQueries = rows(w.rotatedQueries).view(tokens, heads, d)
          val rotatedKeys = rows(w.rotatedKeys).view(tokens, kvHeads, d)
          ops.rope(q.view(tokens, heads, d), positions, rope, rotatedQueries)
          ops.rope(k.view(tokens, kvHeads, d), positions, rope, rotatedKeys)
          val cached = if (bidirectional) tokens else keys
          ops.cacheWrite(
            rotatedKeys.rows(0, cached),
            v.view(tokens, kvHeads, d).rows(0, cached),
            cache,
            pageTable,
            start
          )
          val attended = rows(w.attended).view(tokens, heads, d)
          if (bidirectional)
            ops.attention(
              rotatedQueries,
              cache,
              pageTable,
              start,
              start + tokens,
              attention(index).copy(causal = false, window = None),
              attended
            )
          else
            ops.attention(
              rotatedQueries.rows(0, keys),
              cache,
              pageTable,
              start,
              start + keys,
              attention(index),
              attended.rows(0, keys)
            )
          if (!bidirectional && keys < tokens)
            ops.attention(
              rotatedQueries.rows(keys, tokens - keys),
              cache,
              pageTable,
              start + keys,
              start + keys,
              attention(index).copy(causal = false, window = None),
              attended.rows(keys, tokens - keys)
            )
          val projected = rows(w.projected)
          linear(
            attended.view(tokens, heads.toLong * d),
            index,
            "o",
            layer.o,
            projected
          )
          // Gemma's sandwich norms: the sublayer's output normed, in place
          def residual(postNorm: Option[Tensor]): Unit = {
            postNorm.foreach(weight =>
              ops.rmsNorm(
                projected,
                weight,
                config.rmsEpsilon,
                style.normOffset,
                projected
              )
            )
            ops.add(x, projected, x)
          }
          residual(layer.postAttentionNorm)
          ops.rmsNorm(
            x,
            layer.mlpNorm,
            config.rmsEpsilon,
            style.normOffset,
            normed
          )
          val (gate, up) = (rows(w.gate), rows(w.up))
          linear(normed, index, "gate", layer.gate, gate)
          linear(normed, index, "up", layer.up, up)
          ops.gated(style.activation, gate, up, gate)
          linear(gate, index, "down", layer.down, projected)
          residual(layer.postMlpNorm)
          for {
            seen <- images
            rows <- seen.deepstack.lift(index)
            g <- rows
          } ops.add(x.rows(g.at, g.count), g.rows, x.rows(g.at, g.count))
          after(index + 1)
      }
      x
    }
  }

  def close(): Unit = {
    workspace.foreach(_.release())
    converted.foreach(ops.release)
    source.close()
  }
}

object Qwen3 {

  /** Opens a Qwen 3 from a GGUF, or from safetensors with their `config.json`,
    * or Qwen3-VL's language model from transformers' names alone (MiniMax H3's
    * text encoder, `Qwen3Config.fromWeights`).
    */
  def open(ops: Ops, path: java.nio.file.Path): DenseDecoder = {
    val source = WeightSource.open(ops, path)
    try {
      source.gguf match {
        case _
            if source.has("model.embed_tokens.weight") &&
              (source.gguf.isDefined || source.config.isEmpty &&
                source.has("visual.patch_embed.proj.weight")) =>
          // transformers' names with no metadata: MiniMax H3's text encoder
          // (a GGUF, or safetensors holding Qwen3-VL's tower beside it)
          new DenseDecoder(
            ops,
            source,
            Qwen3Config.fromWeights(source, "model."),
            DenseNames.HuggingFace
          )
        case Some(file) =>
          val vocabulary = source(
            DenseNames.Gguf.embedding
          ).shape.dimensions.head.toInt
          new DenseDecoder(
            ops,
            source,
            Qwen3Config.fromGguf(file, vocabulary),
            DenseNames.Gguf
          )
        case None =>
          val config = source.config.getOrElse(
            throw new FormatException(s"no config.json beside $path")
          )
          new DenseDecoder(
            ops,
            source,
            Qwen3Config.fromHuggingFace(config.text),
            DenseNames.HuggingFace
          )
      }
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}

object Gemma2 {

  /** Opens a Gemma 2 from safetensors, with their `config.json` when beside
    * them, else as Gemma 2 2B (ComfyUI's single file).
    */
  def open(ops: Ops, path: java.nio.file.Path): DenseDecoder = {
    val source = WeightSource.open(ops, path)
    try {
      val prefix =
        if (source.has("model.embed_tokens.weight")) "model." else ""
      new DenseDecoder(
        ops,
        source,
        source.config
          .map(Gemma2Config.fromHuggingFace)
          .getOrElse(Gemma2Config.gemma2b(source, prefix)),
        DenseNames.gemma2(prefix)
      )
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}

object Mistral {

  /** Opens Mistral Small's language model from a GGUF, or from safetensors with
    * their `config.json` (a Mistral 3's `text_config`).
    */
  def open(ops: Ops, path: java.nio.file.Path): DenseDecoder = {
    val source = WeightSource.open(ops, path)
    try {
      source.gguf match {
        case Some(file) =>
          val vocabulary = source(
            DenseNames.Gguf.embedding
          ).shape.dimensions.head.toInt
          new DenseDecoder(
            ops,
            source,
            MistralConfig.fromGguf(file, vocabulary),
            DenseNames.Gguf
          )
        case None =>
          val config = source.config.getOrElse(
            throw new FormatException(s"no config.json beside $path")
          )
          new DenseDecoder(
            ops,
            source,
            MistralConfig.fromHuggingFace(config.text),
            DenseNames.HuggingFace
          )
      }
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
