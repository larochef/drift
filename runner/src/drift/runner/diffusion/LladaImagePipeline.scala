package drift.runner.diffusion

import drift.runner.formats.ModelConfig
import drift.runner.models.*
import drift.runner.models.Flux2.Grid
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.SplittableRandom
import scala.collection.mutable

/** LLaDA-Image end to end (inclusionAI's `LLaDAImagePipeline`), base or Turbo.
  *   - The prompt: `Generate an image: …` in LLaDA2's roles, its token
  *     embeddings read by the QueryFormer, whose 256 queries follow them
  *     through the LLaDA2 text model (the text blind to the queries); every
  *     hidden state, the text's and the queries', through the text projection
  *     to the transformer's caption features.
  *   - Noise of the FLUX.2 VAE's 128 packed features, denoised on the released
  *     Kumaraswamy schedule: Euler steps for the base model; for Turbo, whose
  *     scheduler shifts the schedule by 3 and samples stochastically, each
  *     step's clean image under fresh noise at the next level. The latents
  *     decoded by the FLUX.2 VAE. `--cfg-scale` above 1 runs the negative
  *     prompt too (5 at 50 steps for the base model; Turbo runs 4 steps at 1);
  *     an empty negative prompt is the bare `Generate an image.`
  *   - Editing (a reference image; sides multiples of 32): the reference at the
  *     output's size as latents beside the target's, and at half that size
  *     through SigVQ to semantic tokens; with guidance, the negative side keeps
  *     the latents and drops the semantic tokens.
  *   - img2img: the init image's latents mixed with the noise at the schedule's
  *     step `steps − ⌊steps × strength⌋`, with a mask of the part to repaint
  *     (the released pipeline has none).
  * The files: the released folders (`transformer/`, `text_encoder/` with its
  * `config.json`, `queryformer/`, `text_projection/`, `sigvq/`, `tokenizer/`)
  * found from the transformer's; or each small model's file given (`parts`); or
  * `connectors` (one file holding the three under `queryformer.`,
  * `text_projection.` and `sigvq.`, as sd-cpp takes them); `tokenizer` given or
  * found.
  */
final class LladaImagePipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Option[Path],
    connectors: Option[Path],
    /** The small models' own files, by name (`queryformer`, `text_projection`,
      * `sigvq`).
      */
    parts: Map[String, Path]
) extends ImagePipeline {

  def family: String = "LLaDA-Image"
  def takesInitImage: Boolean = true
  override def takesMask: Boolean = true
  def takesReferences: Boolean = sigvq.isDefined
  def takesLoras: Boolean = false
  override def takesSigmas: Boolean = true
  override def ownShift: Boolean = true
  def takesGuidance: Boolean = false

  /** The released repository's folder, when the transformer sits in one. */
  private val repository = Option(diffusionModel.toAbsolutePath.getParent)
    .flatMap(folder => Option(folder.getParent))
  private def beside(folder: String, file: String): Option[Path] =
    repository.map(_.resolve(folder).resolve(file)).filter(Files.exists(_))

  private val tokenizer: Tokenizer = TokenizerJson.load(
    tokenizerFile
      .orElse(beside("tokenizer", "tokenizer.json"))
      .getOrElse(
        throw new IllegalArgumentException(
          "LLaDA-Image needs LLaDA2's tokenizer.json: pass --tokenizer <file>"
        )
      )
  )
  private val encoder = Llada2.open(ops, textEncoder)
  private val transformer = LladaImage.open(ops, diffusionModel)
  private val autoencoder = FluxVae.open(ops, vae)

  private val opened = mutable.ArrayBuffer.empty[WeightSource]
  private def part(name: String): Option[(WeightSource, String)] =
    if (parts.contains(name)) {
      val source = WeightSource.open(ops, parts(name))
      opened += source
      Some((source, ""))
    } else
      connectors match {
        case Some(file) =>
          val source = opened.headOption.getOrElse {
            val one = WeightSource.open(ops, file)
            opened += one
            one
          }
          Option.when(source.names.exists(_.startsWith(s"$name.")))(
            (source, s"$name.")
          )
        case None =>
          beside(name, "diffusion_pytorch_model.safetensors").map { file =>
            val source = WeightSource.open(ops, file)
            opened += source
            (source, "")
          }
      }
  private def needed(name: String) = part(name).getOrElse(
    throw new IllegalArgumentException(
      s"LLaDA-Image needs its $name: pass --llada-${name.replace('_', '-')} <file> or --embeddings-connectors <file>, or keep the released folders together"
    )
  )
  private val queryFormer = {
    val (source, prefix) = needed("queryformer")
    new LladaQueryFormer(ops, source, prefix)
  }
  private val projection = {
    val (source, prefix) = needed("text_projection")
    new LladaTextProjection(ops, source, prefix)
  }
  private val sigvq =
    part("sigvq").map((source, prefix) => new LladaSigVq(ops, source, prefix))

  /** The released scheduler's settings, when its `scheduler_config.json` sits
    * in the repository: the base model's is unshifted and deterministic,
    * Turbo's shifted by 3 and stochastic.
    */
  private val scheduler =
    beside("scheduler", "scheduler_config.json").map(ModelConfig.read)
  private def scheduled[A](key: String, read: ModelConfig => A): Option[A] =
    scheduler.filter(_.has(key)).map(read)

  require(
    projection.outputs == transformer.config.captionWidth,
    s"the transformer takes ${transformer.config.captionWidth} caption features, " +
      s"not the projection's ${projection.outputs}"
  )

  /** The prompt's caption features, `[tokens + queries, captionWidth]`. */
  private def caption(prompt: String): Tensor = {
    val asked = prompt.trim
    val ids = tokenizer.encode(
      (if (asked.isEmpty) "<role>HUMAN</role> Generate an image."
       else s"<role>HUMAN</role> Generate an image: $asked") +
        "\n<role>ASSISTANT</role>\n<IMAGE1>",
      addSpecial = true
    )
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    try {
      val embeds = hold(encoder.embed(ids))
      val queries = hold(queryFormer(embeds))
      val width = embeds.shape.last
      val (text, asking) =
        (ids.length.toLong, queries.shape.dimensions.head)
      val all = hold(ops.allocate(DType.F32, Shape.of(text + asking, width)))
      ops.copy(embeds, all.rows(0, text))
      ops.copy(queries, all.rows(text, asking))
      projection(hold(encoder.encode(all, ids.length)))
    } finally held.foreach(ops.release)
  }

  /** `picture` at `width × height` as pixels `[height, width, 3]` in [−1, 1].
    */
  private def pixels(picture: BufferedImage, width: Int, height: Int) =
    Images.pixels(Images.resized(picture, width, height))

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val editing = request.references.nonEmpty
    val multiple = if (editing) 32 else 16
    require(
      request.width % multiple == 0 && request.height % multiple == 0,
      s"${request.width} × ${request.height}: LLaDA-Image takes multiples of $multiple" +
        (if (editing) " when it edits" else "")
    )
    if (request.references.size > 1)
      println(
        "[WARN] LLaDA-Image edits one image: the other references are left out"
      )
    val grid = Grid(request.height / 16, request.width / 16)
    val channels = transformer.config.latentChannels
    val guided = request.cfgScale > 1f
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = {
      held += tensor
      tensor
    }
    val conditions = mutable.ArrayBuffer.empty[LladaCondition]
    def keep(condition: LladaCondition) = {
      conditions += condition
      condition
    }
    try {
      // an edit's source: its latents, and its semantic tokens at half size
      val source = request.references.headOption.map { picture =>
        val values = pixels(picture, request.width, request.height)
        val image =
          ops.fromFloats(Shape.of(request.height, request.width, 3), values)
        val half = ops.fromFloats(
          Shape.of(request.height / 2, request.width / 2, 3),
          LladaImagePipeline.halved(values, request.width, request.height)
        )
        try {
          val reader = sigvq.getOrElse(
            throw new IllegalArgumentException(
              "LLaDA-Image edits with its SigVQ encoder: use connectors that hold it"
            )
          )
          (
            hold(autoencoder.encode(image)),
            hold(reader.semantic(reader.tokens(half)))
          )
        } finally {
          ops.release(image)
          ops.release(half)
        }
      }
      def condition(prompt: String, semantic: Boolean) = {
        val features = caption(prompt)
        try
          keep(
            transformer.condition(
              features,
              source.filter(_ => semantic).map(_._2),
              grid.tokens,
              editing
            )
          )
        finally ops.release(features)
      }
      val conditional = condition(request.prompt, semantic = true)
      val unconditional =
        Option.when(guided)(condition(request.negativePrompt, semantic = false))
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(grid.tokens * channels)(Images.gaussian(random))
      val x = hold(ops.fromFloats(Shape.of(grid.tokens, channels), noise))
      // the flow shift: the request's or the launch's, else the released
      // scheduler's, else none; above 1 it is Turbo's scheduler, whose steps
      // are stochastic
      val shift = request.shift
        .orElse(scheduled("shift", _.double("shift")))
        .getOrElse(1.0)
      val stochastic = request.sigmas.isEmpty && scheduled(
        "stochastic_sampling",
        _.boolean("stochastic_sampling")
      ).getOrElse(shift > 1)
      val sigmas = request.sigmas.getOrElse(
        LladaImagePipeline.sigmas(request.steps, shift)
      )
      val inpainting =
        request.mask.filter(_ => request.initImage.isDefined).map { mask =>
          new Inpainting(
            ops,
            x.shape,
            Inpainting.weights(mask, grid.height, grid.width, channels),
            noise
          )
        }
      // img2img: x = σ × noise + (1 − σ) × init at the first step run
      val first = request.initImage.fold(0) { init =>
        val start = FlowSchedule.firstStep(request.steps, request.strength)
        val image = ops.fromFloats(
          Shape.of(request.height, request.width, 3),
          pixels(init, request.width, request.height)
        )
        val encoded =
          try autoencoder.encode(image)
          finally ops.release(image)
        try {
          inpainting.foreach(_.remember(encoded))
          ops.scale(x, sigmas(start), x)
          ops.scale(encoded, 1 - sigmas(start), encoded)
          ops.add(x, encoded, x)
        } finally ops.release(encoded)
        start
      }
      val velocity = hold(ops.allocate(DType.F32, x.shape))
      val other =
        unconditional.map(_ => hold(ops.allocate(DType.F32, x.shape)))
      val steps = request.steps - first
      progress(0, steps)
      (first until request.steps).foreach { i =>
        transformer.velocity(
          x,
          grid,
          conditional,
          source.map(_._1),
          sigmas(i),
          velocity
        )
        for {
          uncond <- unconditional
          v <- other
        } {
          transformer.velocity(x, grid, uncond, source.map(_._1), sigmas(i), v)
          // v = uncond + scale × (cond − uncond)
          ops.scale(v, 1 - request.cfgScale, v)
          ops.scale(velocity, request.cfgScale, velocity)
          ops.add(velocity, v, velocity)
        }
        if (stochastic) {
          // the clean image the step points at, under fresh noise at the
          // next level
          ops.scale(velocity, -sigmas(i), velocity)
          ops.add(x, velocity, x)
          ops.scale(x, 1 - sigmas(i + 1), x)
          val fresh = ops.fromFloats(
            x.shape,
            Array.fill(grid.tokens * channels)(
              sigmas(i + 1) * Images.gaussian(random)
            )
          )
          try ops.add(x, fresh, x)
          finally ops.release(fresh)
        } else {
          ops.scale(velocity, sigmas(i + 1) - sigmas(i), velocity)
          ops.add(x, velocity, x)
        }
        inpainting.foreach(_.restore(x, sigmas(i + 1)))
        progress(i + 1 - first, steps)
      }
      inpainting.foreach(_.release())
      val rgb = autoencoder.decode(x, grid.height)
      try Images.toImage(ops.toFloats(rgb), request.width, request.height)
      finally ops.release(rgb)
    } finally {
      conditions.foreach(_.close())
      held.foreach(ops.release)
    }
  }

  def close(): Unit = {
    sigvq.foreach(_.close())
    projection.close()
    queryFormer.close()
    opened.foreach(_.close())
    autoencoder.close()
    transformer.close()
    encoder.close()
  }
}

object LladaImagePipeline {

  /** The released schedule: `steps` levels on a Kumaraswamy curve, `1 − (1 − (1
    * − u^1.17)^0.8)^1.1` for `u` from 0.001 towards 1, each shifted (`shift × σ
    * / (1 + (shift − 1) × σ)`), then 0.
    */
  def sigmas(steps: Int, shift: Double): IndexedSeq[Float] =
    (0 until steps).map { i =>
      val u = 0.001 + i * (1.0 - 0.001) / steps
      val level = 1 - math.pow(1 - math.pow(1 - math.pow(u, 1.17), 0.8), 1.1)
      (shift * level / (1 + (shift - 1) * level)).toFloat
    } :+ 0f

  /** A channels-last image at half its size, each pixel the mean of four
    * (bilinear at exactly half, as `F.interpolate` samples it).
    */
  def halved(values: Array[Float], width: Int, height: Int): Array[Float] = {
    val channels = values.length / (width * height)
    Array.tabulate(height / 2 * (width / 2) * channels) { i =>
      val (pixel, c) = (i / channels, i % channels)
      val (y, x) = (pixel / (width / 2), pixel % (width / 2))
      def at(dy: Int, dx: Int) =
        values(((2 * y + dy) * width + 2 * x + dx) * channels + c)
      (at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)) / 4
    }
  }
}
