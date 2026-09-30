package drift.runner.models

import drift.runner.diffusion.{Lora, LoraUpdates}
import drift.runner.ops.*
import drift.runner.state.Sequence
import drift.runner.tensor.*

import scala.collection.mutable.ArrayBuffer

/** HiDream O1 Image (HiDream-ai's `Qwen3VLForConditionalGeneration` with a
  * pixel head, `models/qwen3_vl_transformers.py`): Qwen3-VL-8B's language model
  * is the diffusion transformer itself, over the prompt and the image's 32 × 32
  * pixel patches — no VAE, no separate text encoder. The prompt's tokens attend
  * causally among themselves; the timestep token (its embedding the timestep's)
  * and the patches, the generated tokens, attend to everything both ways. So
  * the prompt runs once into the sequence's cache (`prefill`), and each step
  * runs the generated tokens alone over it (`predict`), which the official
  * two-pass attention computes the same.
  *
  * A patch is its pixels channel-major (`(C p1 p2)`, 3072 values in [−1, 1]):
  * in through a bottleneck (`x_embedder`: 3072 → 1024 → hidden), out through
  * `final_layer2` after the final norm, as the model's clean image x̂.
  *
  * LoRAs reach every linear at run time (`useLoras`): the decoder's, the
  * timestep MLP's, the bottleneck's and the pixel head's, by the checkpoint's
  * names.
  */
final class HiDreamO1 private (
    ops: Ops,
    source: WeightSource,
    val decoder: DenseDecoder,
    updates: LoraUpdates
) extends AutoCloseable {

  /** Values per patch: 3 channels of 32 × 32. */
  val patchValues: Int = 3 * HiDreamO1.Patch * HiDreamO1.Patch

  private def hidden = decoder.config.hidden

  /** Biases stored in BF16, converted to F32 once. */
  private val converted = ArrayBuffer.empty[Tensor]
  private def bias(name: String): Tensor = {
    val stored = source(name)
    val copy = ops.allocate(DType.F32, stored.shape)
    ops.convert(stored, copy)
    converted += copy
    copy
  }

  private val timeIn = source("model.t_embedder1.mlp.0.weight")
  private val timeInBias = bias("model.t_embedder1.mlp.0.bias")
  private val timeOut = source("model.t_embedder1.mlp.2.weight")
  private val timeOutBias = bias("model.t_embedder1.mlp.2.bias")
  private val patchIn = source("model.x_embedder.proj1.weight")
  private val patchOut = source("model.x_embedder.proj2.weight")
  private val patchOutBias = bias("model.x_embedder.proj2.bias")
  private val pixels = source("model.final_layer2.linear.weight")
  private val pixelsBias = bias("model.final_layer2.linear.bias")

  private val frequencies = timeIn.shape.dimensions.last.toInt

  private val HeadSites = Set(
    "model.t_embedder1.mlp.0",
    "model.t_embedder1.mlp.2",
    "model.x_embedder.proj1",
    "model.x_embedder.proj2",
    "model.final_layer2.linear"
  )

  /** A head's linear (`site` its weight's name less `.weight`), its LoRA
    * updates added.
    */
  private def linear(x: Tensor, site: String, weight: Tensor, out: Tensor) = {
    ops.linear(x, weight, out)
    updates(x, site, out)
  }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last.
    * A file's targets are the checkpoint's names less `model.` (ComfyUI's
    * `diffusion_model.` gone by then). Returns the targets that matched
    * nothing, left unapplied.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val sites = decoder.loraSites ++ HeadSites
    val all = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map((target, pair) =>
        (if (target.startsWith("model.")) target else s"model.$target") ->
          pair.copy(scale = pair.scale * multiplier)
      )
    )
    updates.use(
      all.filter((site, _) => sites.contains(site)).groupMap(_._1)(_._2)
    )
    all.map(_._1).distinct.filterNot(sites.contains)
  }

  /** A sequence for a prompt of `promptTokens` (the timestep token included)
    * and `patches` patches in a `gridHeight × gridWidth` grid, the patches
    * placed as the official `get_rope_index_fix_point` places them: the prompt
    * at its slots on all three axes, the patches at (4096, 4096 + row, 4096 +
    * column).
    */
  def newSequence(
      promptTokens: Int,
      gridHeight: Int,
      gridWidth: Int
  ): Sequence = {
    val patches = gridHeight * gridWidth
    val sequence = decoder.newSequence(promptTokens + patches, 64)
    val at = HiDreamO1.PatchPositions
    val positions = new Array[Int](3 * patches)
    (0 until patches).foreach { p =>
      positions(p) = at
      positions(patches + p) = at + p / gridWidth
      positions(2 * patches + p) = at + p % gridWidth
    }
    sequence.place(promptTokens, positions)
    sequence
  }

  /** The prompt before its timestep token into `sequence`. */
  def prefill(ids: Array[Int], sequence: Sequence): Unit =
    decoder.prefill(ids, sequence)

  /** The timestep token's embedding into `out` (`[1, hidden]`): `1000 t`'s
    * sinusoid (cosines then sines of 128 frequencies `10000^(−i/128)`) through
    * an MLP with a SiLU.
    */
  private def timeEmbedding(t: Float, out: Tensor): Unit = {
    val half = frequencies / 2
    val sinusoid = new Array[Float](frequencies)
    (0 until half).foreach { i =>
      val angle = 1000.0 * t * math.exp(-math.log(10000.0) * i / half)
      sinusoid(i) = math.cos(angle).toFloat
      sinusoid(half + i) = math.sin(angle).toFloat
    }
    val input = ops.fromFloats(Shape.of(1, frequencies), sinusoid)
    val inner = ops.allocate(DType.F32, Shape.of(1, hidden))
    try {
      linear(input, "model.t_embedder1.mlp.0", timeIn, inner)
      ops.addRow(inner, timeInBias, inner)
      ops.activation(Activation.Silu, inner, inner)
      linear(inner, "model.t_embedder1.mlp.2", timeOut, out)
      ops.addRow(out, timeOutBias, out)
    } finally {
      ops.release(input)
      ops.release(inner)
    }
  }

  /** x̂, the model's clean image, of `patches` (`[L, 3072]`) at timestep `t` (1
    * − σ), the prompt cached in `sequence` before `start` (the timestep token's
    * slot), into `out` (like `patches`).
    */
  def predict(
      patches: Tensor,
      t: Float,
      start: Int,
      sequence: Sequence,
      out: Tensor
  ): Unit = {
    val count = patches.shape.dimensions.head
    val rows = ops.allocate(DType.F32, Shape.of(1 + count, hidden))
    val bottleneck =
      ops.allocate(DType.F32, Shape.of(count, patchIn.shape.dimensions.head))
    val states = ops.allocate(DType.F32, Shape.of(1 + count, hidden))
    try {
      timeEmbedding(t, rows.rows(0, 1))
      val image = rows.rows(1, count)
      linear(patches, "model.x_embedder.proj1", patchIn, bottleneck)
      linear(bottleneck, "model.x_embedder.proj2", patchOut, image)
      ops.addRow(image, patchOutBias, image)
      decoder.attendAll(rows, start, sequence, states)
      linear(states.rows(1, count), "model.final_layer2.linear", pixels, out)
      ops.addRow(out, pixelsBias, out)
    } finally Seq(rows, bottleneck, states).foreach(ops.release)
  }

  def close(): Unit = {
    converted.foreach(ops.release)
    updates.close()
    decoder.close()
  }
}

object HiDreamO1 {

  /** Pixels per patch side. */
  val Patch = 32

  /** Where the target image's patches start on every mRoPE axis (`fix_point`).
    */
  val PatchPositions = 4096

  /** The timestep token, `<|tms_token|>`. */
  val TimestepToken = 151673

  private val Prefix = "model.language_model."

  def holds(source: WeightSource): Boolean =
    source.has("model.x_embedder.proj1.weight")

  def open(ops: Ops, path: java.nio.file.Path): HiDreamO1 = {
    val source = WeightSource.open(ops, path)
    val updates = new LoraUpdates(ops)
    try
      new HiDreamO1(
        ops,
        source,
        new DenseDecoder(
          ops,
          source,
          config(source),
          DenseNames.huggingFace(Prefix),
          Some(updates)
        ),
        updates
      )
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }

  /** Qwen3-VL-8B's text model, read off the shapes; RoPE as HiDream's
    * `config.json` has it (θ 5·10⁶, interleaved mRoPE sections 24/20/20).
    */
  private def config(source: WeightSource): DenseConfig = {
    def dimensions(name: String) = source.shape(s"$Prefix$name").dimensions
    val Seq(vocabulary, hidden) = dimensions("embed_tokens.weight")
    val headDimension = dimensions("layers.0.self_attn.q_norm.weight").head
    val layers = Iterator
      .from(0)
      .takeWhile(i => source.has(s"${Prefix}layers.$i.input_layernorm.weight"))
      .size
    DenseConfig(
      layers = layers,
      hidden = hidden.toInt,
      intermediate = dimensions("layers.0.mlp.gate_proj.weight").head.toInt,
      heads = (dimensions(
        "layers.0.self_attn.q_proj.weight"
      ).head / headDimension).toInt,
      kvHeads = (dimensions(
        "layers.0.self_attn.k_proj.weight"
      ).head / headDimension).toInt,
      headDimension = headDimension.toInt,
      ropeTheta = 5000000f,
      ropeSections = RopeSections.Interleaved(24, 20, 20),
      ropeLayout = RopeLayout.Neox,
      rmsEpsilon = 1e-6f,
      vocabulary = vocabulary.toInt,
      queryScale = (1 / math.sqrt(headDimension.toDouble)).toFloat,
      style = DenseStyle.Qwen3
    )
  }
}
