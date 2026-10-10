package drift.runner.models

import drift.runner.diffusion.{Lora, LoraUpdates}
import drift.runner.ops.*
import drift.runner.state.Sequence
import drift.runner.tensor.*

import scala.collection.mutable

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
  * Reference images (`models/pipeline.py`) come in twice. The prompt carries
  * each as Qwen3-VL reads a picture — its vision tower's tokens, with deepstack
  * — in the cached prefix (`prefill`'s `images`). And their pixel patches, in
  * through the same bottleneck, follow the target's among the generated tokens
  * (`embed`, `predict`'s `references`), each reference on its own grid past the
  * target's on every axis; only the target's rows go through the pixel head.
  *
  * LoRAs reach every linear at run time (`useLoras`): the decoder's, the
  * timestep MLP's, the bottleneck's and the pixel head's, by the checkpoint's
  * names; a file's whole-tensor differences (ComfyUI's `.diff` and `.diff_b`,
  * which a LoRA extracted from a checkpoint carries) are added to the norms'
  * weights and to the heads' biases.
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

  /** Biases stored in BF16, converted to F32 once, by their linear's site; and
    * those a LoRA file's differences are added to, which stand for them while
    * active.
    */
  private val storedBiases = mutable.Map.empty[String, Tensor]
  private val changedBiases = mutable.Map.empty[String, Tensor]
  private def bias(site: String): () => Tensor = {
    val stored = source(s"$site.bias")
    val copy = ops.allocate(DType.F32, stored.shape)
    if (stored.dtype == DType.F32) ops.copy(stored, copy)
    else ops.convert(stored, copy)
    storedBiases(site) = copy
    () => changedBiases.getOrElse(site, copy)
  }

  private val timeIn = source("model.t_embedder1.mlp.0.weight")
  private val timeInBias = bias("model.t_embedder1.mlp.0")
  private val timeOut = source("model.t_embedder1.mlp.2.weight")
  private val timeOutBias = bias("model.t_embedder1.mlp.2")
  private val patchIn = source("model.x_embedder.proj1.weight")
  private val patchOut = source("model.x_embedder.proj2.weight")
  private val patchOutBias = bias("model.x_embedder.proj2")
  private val pixels = source("model.final_layer2.linear.weight")
  private val pixelsBias = bias("model.final_layer2.linear")

  private val frequencies = timeIn.shape.dimensions.last.toInt

  /** A head's linear (`site` its weight's name less `.weight`), its LoRA
    * updates added.
    */
  private def linear(x: Tensor, site: String, weight: Tensor, out: Tensor) = {
    ops.linear(x, weight, out)
    updates(x, site, out)
  }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last:
    * the pairs whose target is one of the model's linears (`HiDreamO1.site`
    * reads the namings), and the whole-tensor differences of a norm's weight
    * (`.diff`) or of a head's bias (`.diff_b`). What a file holds for the
    * vision tower or the token embeddings is left out and counted.
    */
  def useLoras(loras: Seq[(Lora, Float)]): HiDreamO1Loras = {
    val linears = decoder.loraSites ++ HiDreamO1.HeadSites
    val sites = linears ++ decoder.normSites ++ storedBiases.keySet
    def site(target: String) = HiDreamO1.site(target, sites)
    val pairs = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map((target, pair) =>
        site(target) -> pair.copy(scale = pair.scale * multiplier)
      )
    )
    updates.use(
      pairs.filter((site, _) => linears.contains(site)).groupMap(_._1)(_._2)
    )
    val differences = for {
      (lora, multiplier) <- loras
      name <- lora.unread
      (suffix, ofBias) <- Seq(".diff_b" -> true, ".diff" -> false)
      if name.endsWith(suffix)
    } yield HiDreamO1.Difference(
      site(name.stripSuffix(suffix)),
      ofBias,
      lora,
      name,
      multiplier
    )
    val norms =
      differences.filter(d => !d.ofBias && decoder.normSites.contains(d.site))
    val biases =
      differences.filter(d => d.ofBias && storedBiases.contains(d.site))
    decoder.useNormDifferences(norms.groupMap(_.site)(_.values))
    changedBiases.values.foreach(ops.release)
    changedBiases.clear()
    biases.groupBy(_.site).foreach { (site, added) =>
      val stored = storedBiases(site)
      val values = ops.toFloats(stored)
      added.foreach { difference =>
        val more = difference.values
        require(
          more.length == values.length,
          s"${difference.name}: ${more.length} values for a bias of ${values.length}"
        )
        values.indices.foreach(i => values(i) += more(i))
      }
      changedBiases(site) = ops.fromFloats(stored.shape, values)
    }
    val taken = (norms ++ biases).map(_.name).toSet
    val left =
      pairs.map(_._1).distinct.filterNot(linears.contains) ++
        differences.filterNot(d => taken(d.name)).map(_.site)
    val (outside, unmatched) = left.partition(HiDreamO1.outside)
    HiDreamO1Loras(
      updates = pairs.size + differences.size,
      applied = pairs.count((site, _) => linears.contains(site)) + taken.size,
      outside = outside.size,
      unmatched = unmatched,
      unread = loras
        .flatMap(_._1.unread)
        .filterNot(name => name.endsWith(".diff") || name.endsWith(".diff_b"))
    )
  }

  /** A sequence for a prompt whose tokens (the timestep token included) turn by
    * `promptPositions` (`[3, tokens]` axis-major, `Prompt.positions`), a target
    * of `gridHeight × gridWidth` patches and `references`' patches (each its
    * grid's height and width), placed as `HiDreamO1.patchPositions` places
    * them.
    */
  def newSequence(
      promptPositions: Array[Int],
      gridHeight: Int,
      gridWidth: Int,
      references: Seq[(Int, Int)] = Nil
  ): Sequence = {
    val promptTokens = promptPositions.length / 3
    val grids = (gridHeight, gridWidth) +: references
    val sequence =
      decoder.newSequence(promptTokens + grids.map(_ * _).sum, 64)
    sequence.place(0, promptPositions)
    sequence.place(promptTokens, HiDreamO1.patchPositions(grids))
    sequence
  }

  /** The prompt before its timestep token into `sequence`, the reference images
    * it carries as their vision tower's tokens (`images`).
    */
  def prefill(
      ids: Array[Int],
      sequence: Sequence,
      images: Option[SeenImages] = None
  ): Unit =
    decoder.prefill(ids, sequence, images)

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
      ops.addRow(inner, timeInBias(), inner)
      ops.activation(Activation.Silu, inner, inner)
      linear(inner, "model.t_embedder1.mlp.2", timeOut, out)
      ops.addRow(out, timeOutBias(), out)
    } finally {
      ops.release(input)
      ops.release(inner)
    }
  }

  /** Pixel patches (`[L, 3072]`) through the bottleneck into `out` (`[L,
    * hidden]`): what the model reads of a patch, the target's each step and a
    * reference's once.
    */
  def embed(patches: Tensor, out: Tensor): Unit = {
    val count = patches.shape.dimensions.head
    val bottleneck =
      ops.allocate(DType.F32, Shape.of(count, patchIn.shape.dimensions.head))
    try {
      linear(patches, "model.x_embedder.proj1", patchIn, bottleneck)
      linear(bottleneck, "model.x_embedder.proj2", patchOut, out)
      ops.addRow(out, patchOutBias(), out)
    } finally ops.release(bottleneck)
  }

  /** x̂, the model's clean image, of `patches` (`[L, 3072]`) at timestep `t` (1
    * − σ), the prompt cached in `sequence` before `start` (the timestep token's
    * slot), into `out` (like `patches`). `references` are the reference images'
    * embedded patches (`embed`), read after the target's and given no x̂.
    */
  def predict(
      patches: Tensor,
      t: Float,
      start: Int,
      sequence: Sequence,
      out: Tensor,
      references: Option[Tensor] = None
  ): Unit = {
    val count = patches.shape.dimensions.head
    val read = references.fold(0L)(_.shape.dimensions.head)
    val rows = ops.allocate(DType.F32, Shape.of(1 + count + read, hidden))
    val states = ops.allocate(DType.F32, Shape.of(1 + count + read, hidden))
    try {
      timeEmbedding(t, rows.rows(0, 1))
      embed(patches, rows.rows(1, count))
      references.foreach(ops.copy(_, rows.rows(1 + count, read)))
      decoder.attendAll(rows, start, sequence, states)
      linear(states.rows(1, count), "model.final_layer2.linear", pixels, out)
      ops.addRow(out, pixelsBias(), out)
    } finally Seq(rows, states).foreach(ops.release)
  }

  def close(): Unit = {
    (storedBiases.values ++ changedBiases.values).foreach(ops.release)
    updates.close()
    decoder.close()
  }
}

/** What `HiDreamO1.useLoras` made of the files: how many `updates` they hold
  * (pairs and whole-tensor differences), how many are `applied`, how many are
  * for parts the runner leaves as they are (`outside`: the vision tower, the
  * token embeddings), the updates of no weight of the model (`unmatched`) and
  * the tensors that are no update it reads (`unread`).
  */
final case class HiDreamO1Loras(
    updates: Int,
    applied: Int,
    outside: Int,
    unmatched: Seq[String],
    unread: Seq[String]
)

object HiDreamO1 {

  /** Pixels per patch side. */
  val Patch = 32

  /** Where the target image's patches start on every mRoPE axis (`fix_point`).
    */
  val PatchPositions = 4096

  /** The timestep token, `<|tms_token|>`. */
  val TimestepToken = 151673

  private val Prefix = "model.language_model."

  /** The linears outside the decoder's layers, by their weight's name less
    * `.weight`.
    */
  val HeadSites: Set[String] = Set(
    "model.t_embedder1.mlp.0",
    "model.t_embedder1.mlp.2",
    "model.x_embedder.proj1",
    "model.x_embedder.proj2",
    "model.final_layer2.linear"
  )

  /** A file's whole-tensor difference: of `site`'s bias or weight, stored under
    * `name` in `lora`, applied at `multiplier`.
    */
  final private case class Difference(
      site: String,
      ofBias: Boolean,
      lora: Lora,
      name: String,
      multiplier: Float
  ) {
    def values: Array[Float] = {
      val stored = lora.values(name)
      if (multiplier != 1f) stored.mapInPlace(_ * multiplier)
      stored
    }
  }

  /** What trainers put before the checkpoint's names: PEFT's wrapper,
    * DiffSynth-Studio's pipeline, ComfyUI's and diffusers' (the last two gone
    * by the time most files are read).
    */
  private val Prefixes =
    Seq("base_model.model.", "pipe.dit.", "diffusion_model.", "transformer.")

  /** kohya's prefix, before the name with `_` for every `.`. */
  private val Kohya = "lora_unet_"

  /** The model's name (a weight's less `.weight`) for a file's `target`: the
    * checkpoint's own name with or without `model.` (ai-toolkit and ComfyUI
    * leave it out, DiffSynth-Studio keeps it), behind any of `Prefixes`, or
    * kohya's `lora_unet_` form of one of `sites`. A target that is none of them
    * comes back as it is read.
    */
  def site(target: String, sites: Set[String]): String = {
    val name = Prefixes.find(target.startsWith).fold(target)(target.stripPrefix)
    if (name.startsWith(Kohya)) {
      val underscored = name.stripPrefix(Kohya)
      sites
        .find(_.stripPrefix("model.").replace('.', '_') == underscored)
        .getOrElse(name)
    } else if (name.startsWith("model.")) name
    else s"model.$name"
  }

  /** Whether `site` belongs to a part the runner leaves as the checkpoint has
    * it: the vision tower (Qwen3-VL-8B's own, which the official checkpoints do
    * not change) or the token embeddings.
    */
  def outside(site: String): Boolean = {
    val name = site.stripPrefix("model.").stripPrefix(Kohya)
    name.startsWith("visual.") || name.startsWith("visual_") ||
    name.startsWith("language_model.embed_tokens") ||
    name.startsWith("language_model_embed_tokens")
  }

  /** The mRoPE positions (`[3, patches]`, axis-major) of the patches of `grids`
    * (each its height and width, the target's first), one grid after the other,
    * as the official `get_rope_index_fix_point` places them: the target at
    * (4096, 4096 + row, 4096 + column), each reference at `(base, base + row,
    * base + column)`, `base` past the longer side of every grid before it.
    */
  def patchPositions(grids: Seq[(Int, Int)]): Array[Int] = {
    val patches = grids.map(_ * _).sum
    val positions = new Array[Int](3 * patches)
    var slot = 0
    var base = PatchPositions
    grids.foreach { (height, width) =>
      (0 until height * width).foreach { p =>
        positions(slot) = base
        positions(patches + slot) = base + p / width
        positions(2 * patches + slot) = base + p % width
        slot += 1
      }
      base += math.max(height, width)
    }
    positions
  }

  def holds(source: WeightSource): Boolean =
    source.has("model.x_embedder.proj1.weight")

  /** The vision tower's names in a checkpoint that carries it. */
  val VisionPrefix = "model.visual."

  /** Whether the checkpoint at `path` carries its vision tower whole: ComfyUI's
    * single files keep the tower and leave out the deepstack mergers the
    * references are read through, so theirs does not count.
    */
  def holdsVision(ops: Ops, path: java.nio.file.Path): Boolean = {
    val source = WeightSource.open(ops, path)
    try
      source.has(s"${VisionPrefix}patch_embed.proj.weight") &&
        source.has(s"${VisionPrefix}deepstack_merger_list.0.norm.weight")
    finally source.close()
  }

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
