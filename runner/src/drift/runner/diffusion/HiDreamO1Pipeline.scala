package drift.runner.diffusion

import drift.runner.decode.Prompt
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.state.Sequence
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}
import drift.runner.vision.*

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** HiDream O1 Image end to end (HiDream-ai's `models/pipeline.py`): the prompt
  * in Qwen's chat template, `<|boi_token|><|tms_token|>` after the assistant's
  * turn, cached once; the image as pixel patches from noise, each step's x̂
  * from the model. Guided (CFG above 1: the full checkpoint, at 5) runs the
  * unconditional prompt too, a space unless a negative prompt is given.
  *
  * Three samplers, a request's own when it names one:
  *   - `flash`, the Dev checkpoint's (unguided, and not editing one picture):
  *     the distilled timesteps (`DistilledTimesteps`, resampled for other step
  *     counts) and `z = (1 − σ_next) x̂ + σ_next × 7.5 × noise`, fresh noise
  *     clipped to 2.5 of its deviation each step, from noise of deviation 7.5;
  *   - `unipc`, the full checkpoint's (guided): `UniPc` over its own levels at
  *     shift 3, the model given the level's whole timestep as the official loop
  *     gives it, from noise of deviation 8;
  *   - `euler` on the flow `(x̂ − z) / σ`, what the Dev checkpoint edits one
  *     reference with (over the distilled timesteps); guided, over the
  *     flow-shifted schedule.
  *
  * img2img starts from the init image mixed with the noise at the step `steps −
  * ⌊steps × strength⌋`. LoRAs apply to every linear, the cached prompt's
  * included.
  *
  * References (`vision`, Qwen3-VL-8B's mmproj — the tower the checkpoints were
  * trained with, unchanged, and which a single-file checkpoint leaves out) are
  * read twice, as the official pipeline does: small (384² of area, less beyond
  * four) by the vision tower in the prompt, and as pixel patches after the
  * target's, each resized to an area that depends on how many there are (the
  * target's longer side squared for one).
  */
final class HiDreamO1Pipeline(
    ops: Ops,
    modelFile: Path,
    tokenizerFile: Path,
    vision: Option[Path]
) extends ImagePipeline {

  import HiDreamO1Pipeline.*

  def family: String = "HiDream O1"
  def takesInitImage: Boolean = true
  def takesReferences: Boolean = tower.isDefined
  def takesLoras: Boolean = true
  override def takesSigmas: Boolean = true
  override def ownShift: Boolean = true
  def takesGuidance: Boolean = false
  override def samplers: Seq[String] = Samplers
  override def defaultSampler: String = "default"

  private val tokenizer: Tokenizer = TokenizerJson.load(tokenizerFile)
  private val model = HiDreamO1.open(ops, modelFile)
  private val loraFiles = new LoraFiles(ops)
  // the mmproj given, else the checkpoint's own tower when it carries one
  private val tower = vision
    .map(QwenVision.open(ops, _))
    .orElse(
      Option.when(HiDreamO1.holdsVision(ops, modelFile))(
        QwenVision.fromWeights(ops, modelFile, HiDreamO1.VisionPrefix)
      )
    )
  private val sizing = tower.map(t =>
    ImageSizing(
      t.config.patch,
      t.config.merge,
      ConditionMinimumPixels,
      ConditionMaximumPixels
    )
  )

  /** Makes `loras` the model's active set, and says in the log what became of
    * them.
    */
  private def useLoras(loras: Seq[(Path, Float)]): Unit = {
    val (opened, _) = loraFiles.open(loras)
    val use = model.useLoras(opened)
    LoraReport.told(
      LoraReport
        .lines(opened.size, use.updates, use.applied, use.unread, use.unmatched)
    )
    if (use.outside > 0)
      println(
        s"LoRA: ${use.outside} updates are the vision tower's or the token embeddings', which stay the checkpoint's"
      )
  }

  /** The prompt's tokens, the timestep token last, each of `images` in its
    * place before the text.
    */
  private def prompt(text: String, images: Seq[PreparedImage]): Prompt = {
    val pictures =
      images.map(_ => "<|vision_start|><|image_pad|><|vision_end|>").mkString
    val ids = tokenizer.encode(
      s"<|im_start|>user\n$pictures$text<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>",
      addSpecial = false
    )
    require(
      ids.last == HiDreamO1.TimestepToken,
      s"the tokenizer ends HiDream O1's prompt on ${ids.last}, not <|tms_token|> (${HiDreamO1.TimestepToken}): is $tokenizerFile HiDream O1's tokenizer.json?"
    )
    if (images.isEmpty) Prompt.text(ids)
    else
      Prompt.withImages(
        ids,
        tokenizer
          .id("<|image_pad|>")
          .getOrElse(throw new IllegalStateException("no <|image_pad|> token")),
        images
      )
  }

  /** A reference as both halves of the model read it: the vision tower's tokens
    * of the small picture, and the pixel patches of the large one.
    */
  final private case class Reference(
      image: PreparedImage,
      seen: VisionOutput,
      patches: Array[Float],
      gridHeight: Int,
      gridWidth: Int
  ) {
    def release(): Unit = {
      ops.release(seen.tokens)
      seen.deepstack.foreach(ops.release)
    }
  }

  /** `picture`, one of `count` references of a `width × height` target. */
  private def reference(
      picture: BufferedImage,
      index: Int,
      count: Int,
      width: Int,
      height: Int
  ): Reference = {
    val (tower, sizing) = (this.tower.get, this.sizing.get)
    val large = referencePixels(
      Resampling.rgbOverWhite(picture),
      picture.getWidth,
      picture.getHeight,
      if (count == 1)
        editedSide(picture.getWidth, picture.getHeight, width, height)
      else referenceSide(count, width, height)
    )
    val (smallWidth, smallHeight) =
      conditionSize(conditionSide(count), large.width, large.height)
    val small = Resampling.resampled(
      large.pixels,
      large.width,
      large.height,
      smallWidth,
      smallHeight,
      Resampling.Filter.Lanczos
    )
    // the processor's own resize, which leaves these sizes alone but the
    // smallest (below its 256² of area)
    val (seenHeight, seenWidth) = sizing.size(smallHeight, smallWidth)
    val seenPixels =
      if (seenHeight == smallHeight && seenWidth == smallWidth) small
      else
        Resampling.bicubic(
          small,
          smallWidth,
          smallHeight,
          seenWidth,
          seenHeight
        )
    val image = PreparedImage(
      sizing.patches(seenPixels.map(_ / 127.5f - 1f), seenHeight, seenWidth),
      seenHeight / sizing.patch,
      seenWidth / sizing.patch,
      sizing.merge,
      s"reference $index"
    )
    Reference(
      image,
      tower.encode(image),
      patches(large.pixels.map(_ / 127.5f - 1f), large.width, large.height),
      large.height / HiDreamO1.Patch,
      large.width / HiDreamO1.Patch
    )
  }

  /** A prompt cached for every step: its sequence and its timestep token's
    * slot.
    */
  final private class Cached(
      text: String,
      gridHeight: Int,
      gridWidth: Int,
      references: Seq[Reference]
  ) extends AutoCloseable {
    private val tokens = prompt(text, references.map(_.image))
    val start: Int = tokens.ids.length - 1
    val sequence: Sequence = model.newSequence(
      tokens.positions,
      gridHeight,
      gridWidth,
      references.map(r => (r.gridHeight, r.gridWidth))
    )
    try {
      val placed = tokens.images.zip(references)
      model.prefill(
        tokens.ids.init,
        sequence,
        Option.when(references.nonEmpty)(
          SeenImages(
            tokens.positions,
            placed.map((p, r) => GivenRows(p.at, r.seen.tokens)),
            references.head.seen.deepstack.indices.map(k =>
              placed.map((p, r) => GivenRows(p.at, r.seen.deepstack(k)))
            )
          )
        )
      )
    } catch {
      case error: Throwable =>
        sequence.close()
        throw error
    }
    def close(): Unit = sequence.close()
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val patch = HiDreamO1.Patch
    require(
      request.width % patch == 0 && request.height % patch == 0,
      s"${request.width} × ${request.height}: HiDream O1 takes multiples of $patch"
    )
    if (request.references.nonEmpty && tower.isEmpty)
      throw new IllegalArgumentException(
        "HiDream O1 reads references with Qwen3-VL-8B's vision tower: pass --llm_vision <mmproj>"
      )
    val (gridHeight, gridWidth) =
      (request.height / patch, request.width / patch)
    val count = gridHeight * gridWidth
    val values = count * model.patchValues
    val guided = request.cfgScale > 1f
    val sampler = request.sampler
      .filter(Samplers.contains)
      .getOrElse(ownSampler(guided, request.references.size))
    useLoras(request.loras)
    // a request's own levels (a turbo LoRA's) in place of any schedule
    val sigmas = request.sigmas.getOrElse(
      if (sampler == Flash || !guided) distilledSigmas(request.steps)
      else {
        val shift = request.shift.getOrElse(GuidedShift)
        if (sampler == UniPcSampler) UniPc.sigmas(request.steps, shift)
        else FlowSchedule.sigmas(request.steps, math.log(shift))
      }
    )
    println(
      s"sampler $sampler, ${request.steps} steps from σ ${sigmas.head} to ${sigmas(request.steps - 1)}"
    )
    val noiseScale = if (sampler == Flash) DistilledNoise else GuidedNoise
    val random = new SplittableRandom(request.seed)
    val shape = Shape.of(count, model.patchValues)
    val first = request.initImage.fold(0)(_ =>
      FlowSchedule.firstStep(request.steps, request.strength)
    )
    val start = {
      val noise = Array.fill(values)(noiseScale * Images.gaussian(random))
      request.initImage.foreach { init =>
        val sigma = sigmas(first)
        val image = patches(Images.pixels(init), request.width, request.height)
        (0 until values).foreach(i =>
          noise(i) = sigma * noise(i) + (1 - sigma) * image(i)
        )
      }
      noise
    }
    val held = scala.collection.mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = { held += tensor; tensor }
    val references = scala.collection.mutable.ArrayBuffer.empty[Reference]
    val prompts = scala.collection.mutable.ArrayBuffer.empty[Cached]
    try {
      request.references.zipWithIndex.foreach((picture, i) =>
        references += reference(
          picture,
          i,
          request.references.size,
          request.width,
          request.height
        )
      )
      // the references' patches as the model reads them, the same every step
      val read = Option.when(references.nonEmpty) {
        val rows = references.map(r => r.gridHeight * r.gridWidth).sum
        val pixels = hold(
          ops.fromFloats(
            Shape.of(rows, model.patchValues),
            references.toArray.flatMap(_.patches)
          )
        )
        val embedded = hold(
          ops.allocate(DType.F32, Shape.of(rows, model.decoder.config.hidden))
        )
        model.embed(pixels, embedded)
        embedded
      }
      def cached(text: String) = {
        val made = new Cached(text, gridHeight, gridWidth, references.toSeq)
        prompts += made
        made
      }
      val conditional = cached(request.prompt)
      val unconditional = Option.when(guided)(
        cached(
          if (request.negativePrompt.isEmpty) " " else request.negativePrompt
        )
      )
      // the tower's tokens are in the prompts' caches by now
      references.foreach(_.release())
      references.clear()
      val z = hold(ops.fromFloats(shape, start))
      val clean = hold(ops.allocate(DType.F32, shape))
      val other = unconditional.map(_ => hold(ops.allocate(DType.F32, shape)))
      val solver =
        Option.when(sampler == UniPcSampler)(new UniPc(sigmas, first))
      // UniPC's memory: the clean images of the two steps before, the sample
      // the last step was run on, and room for a weighted sum
      var (latest, earlier) = (z, z)
      val scratch = solver.toSeq.flatMap(_ =>
        Seq.fill(5)(hold(ops.allocate(DType.F32, shape)))
      )
      solver.foreach { _ =>
        latest = scratch(0)
        earlier = scratch(1)
      }
      def blend(out: Tensor, terms: (Tensor, Double)*): Unit = {
        val (sum, term) = (scratch(3), scratch(4))
        val weighted = terms.filter(_._2 != 0)
        ops.scale(weighted.head._1, weighted.head._2.toFloat, sum)
        weighted.tail.foreach { (tensor, weight) =>
          ops.scale(tensor, weight.toFloat, term)
          ops.add(sum, term, sum)
        }
        ops.copy(sum, out)
      }
      val steps = request.steps - first
      progress(0, steps)
      (first until request.steps).foreach { i =>
        val (sigma, next) = (sigmas(i), sigmas(i + 1))
        // UniPC's level as the official loop reads it off the scheduler: a
        // whole timestep
        val level = if (solver.isDefined) wholeLevel(sigma) else sigma
        model.predict(
          z,
          1 - level,
          conditional.start,
          conditional.sequence,
          clean,
          read
        )
        for {
          prompt <- unconditional
          x <- other
        } {
          model.predict(z, 1 - level, prompt.start, prompt.sequence, x, read)
          // x̂ = uncond + scale × (cond − uncond), the flow being linear in x̂
          ops.scale(x, 1 - request.cfgScale, x)
          ops.scale(clean, request.cfgScale, clean)
          ops.add(clean, x, clean)
        }
        // the estimate is a picture already (`specs/54-step-preview.md`)
        preview(
          i + 1 - first,
          steps,
          () =>
            HiDreamO1Pipeline.reduced(
              ops.toFloats(clean),
              request.width,
              request.height
            )
        )
        solver match {
          case Some(unipc) =>
            // the scheduler's clean image, from the flow at the whole
            // timestep's level: z + σ / level × (x̂ − z)
            val reach = sigma / math.max(level, MinimumSigma)
            blend(clean, z -> (1.0 - reach), clean -> reach.toDouble)
            unipc
              .corrector(i)
              .foreach(c =>
                blend(
                  z,
                  scratch(2) -> c.sample,
                  latest -> c.latest,
                  earlier -> c.earlier,
                  clean -> c.current
                )
              )
            val oldest = earlier
            earlier = latest
            latest = oldest
            ops.copy(clean, latest)
            ops.copy(z, scratch(2))
            val p = unipc.predictor(i)
            blend(z, z -> p.sample, latest -> p.latest, earlier -> p.earlier)
          case None if sampler == Flash =>
            ops.scale(clean, 1 - next, z)
            if (next > 0) {
              val noise =
                ops.fromFloats(shape, clippedNoise(random, values, next))
              try ops.add(z, noise, z)
              finally ops.release(noise)
            }
          case None =>
            // Euler: z += (σ_next − σ) (x̂ − z) / σ
            val kept = next / math.max(sigma, MinimumSigma)
            ops.scale(z, kept, z)
            ops.scale(clean, 1 - kept, clean)
            ops.add(z, clean, z)
        }
        progress(i + 1 - first, steps)
      }
      Images.toImage(
        pixels(ops.toFloats(z), request.width, request.height),
        request.width,
        request.height
      )
    } finally {
      references.foreach(_.release())
      held.foreach(ops.release)
      prompts.foreach(_.close())
    }
  }

  def close(): Unit = {
    loraFiles.close()
    tower.foreach(_.close())
    model.close()
  }
}

object HiDreamO1Pipeline {

  /** The Dev checkpoint's 28 timesteps (`DEFAULT_TIMESTEPS`), σ × 1000. */
  val DistilledTimesteps: IndexedSeq[Int] =
    IndexedSeq(999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790,
      764, 737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8)

  /** The starting noise's deviation and the flash scheduler's (the official
    * `noise_scale_start` and `_end`).
    */
  val DistilledNoise = 7.5f

  /** The same for the other samplers (`NOISE_SCALE`). */
  val GuidedNoise = 8f

  /** The flash scheduler's noise, clipped to this many of its deviations. */
  val NoiseClip = 2.5f

  /** σ floor of the flow's division (`T_EPS`). */
  val MinimumSigma = 0.001f

  /** The full checkpoint's flow shift (`--shift`), when a request gives none.
    */
  val GuidedShift = 3.0

  val Flash = "flash"
  val UniPcSampler = "unipc"
  val Euler = "euler"

  /** The samplers a request may name. */
  val Samplers: Seq[String] = Seq(UniPcSampler, Euler, Flash)

  /** The sampler of a request that names none, as the official `inference.py`
    * picks it: UniPC guided (the full checkpoint); unguided (Dev) the flash
    * scheduler, but Euler to edit a single reference.
    */
  def ownSampler(guided: Boolean, references: Int): String =
    if (guided) UniPcSampler else if (references == 1) Euler else Flash

  /** A noise level as the official loop reads it off a scheduler's timesteps: a
    * whole number of thousandths, the rest dropped.
    */
  def wholeLevel(sigma: Float): Float = (sigma.toDouble * 1000).toInt / 1000f

  /** The vision tower's bounds on a picture's pixels (the checkpoint's
    * `preprocessor_config.json`).
    */
  val ConditionMinimumPixels: Int = 65536
  val ConditionMaximumPixels: Int = 16777216

  /** The side whose square is the area a reference's pixel patches cover, for
    * `count` references of a `width × height` target (`max_size`).
    */
  def referenceSide(count: Int, width: Int, height: Int): Int = {
    val longer = math.max(width, height)
    if (count == 1) longer
    else if (count == 2) longer * 48 / 64
    else if (count <= 4) longer / 2
    else if (count <= 8) longer * 24 / 64
    else longer / 4
  }

  /** `referenceSide` for the one reference of an edit. A reference of the
    * target's own shape is read at the target's size: the official pipeline's
    * edit takes its target from the reference (`keep_original_aspect`), so the
    * two are the same size there — half the tokens of `max_size`, and the
    * picture the model was shown how to edit. Any other shape keeps `max_size`.
    */
  def editedSide(
      referenceWidth: Int,
      referenceHeight: Int,
      width: Int,
      height: Int
  ): Int = {
    val own = math.ceil(math.sqrt(width.toDouble * height)).toInt
    if (
      referenceSize(referenceWidth, referenceHeight, own)._2 == (width, height)
    )
      own
    else referenceSide(1, width, height)
  }

  /** The side whose square is the area the vision tower sees of each of `count`
    * references (`CONDITION_IMAGE_SIZE`).
    */
  def conditionSide(count: Int): Int =
    if (count <= 4) 384 else if (count <= 8) 384 * 48 / 64 else 384 / 2

  /** The vision tower's picture of a `width × height` reference
    * (`calculate_dimensions`): `side²` of area in its shape, each side cut to a
    * multiple of 32.
    */
  def conditionSize(side: Int, width: Int, height: Int): (Int, Int) = {
    val ratio = width.toDouble / height
    val wide = math.sqrt(side.toDouble * side * ratio)
    ((wide / 32).toInt * 32, (wide / ratio / 32).toInt * 32)
  }

  /** How a `width × height` reference comes to at most `side²` of area
    * (`resize_pilimage`): the size it is scaled to, which covers the size it is
    * then cut to around its centre — sides of multiples of 32, the largest of
    * the rounded candidates that fit the area.
    */
  def referenceSize(
      width: Int,
      height: Int,
      side: Int
  ): ((Int, Int), (Int, Int)) = {
    val patch = HiDreamO1.Patch
    val area = side.toLong * side
    val scale = math.sqrt(area.toDouble / (width.toLong * height))
    def rounded(v: Double) = math.rint(v).toInt / patch * patch
    def floored(v: Double) = math.floor(v).toInt / patch * patch
    val (w, h) = (width * scale, height * scale)
    val candidates = Seq(
      (rounded(w), rounded(h)),
      (rounded(w), floored(h)),
      (floored(w), rounded(h)),
      (floored(w), floored(h))
    ).sortBy((cw, ch) => -cw.toLong * ch)
    val (cutWidth, cutHeight) = {
      val (cw, ch) = candidates
        .find((cw, ch) => cw.toLong * ch <= area)
        .getOrElse(candidates.last)
      (math.max(cw, patch), math.max(ch, patch))
    }
    val (across, down) =
      (width.toDouble / cutWidth, height.toDouble / cutHeight)
    val scaled =
      if (across < down) (cutWidth, math.rint(height / across).toInt)
      else (math.rint(width / down).toInt, cutHeight)
    (scaled, (cutWidth, cutHeight))
  }

  /** Pixels (`[height, width, 3]`, 0–255) of a `width × height` picture. */
  final case class Pixels(pixels: Array[Float], width: Int, height: Int)

  /** A reference's pixels as its patches are cut from them (`resize_pilimage`):
    * halved while its shorter side is twice `side` or more, scaled (bicubic)
    * and cut around its centre to `referenceSize`.
    */
  def referencePixels(
      pixels: Array[Float],
      width: Int,
      height: Int,
      side: Int
  ): Pixels = {
    var from = Pixels(pixels, width, height)
    while (math.min(from.width, from.height) >= 2 * side) from = halved(from)
    val ((scaledWidth, scaledHeight), (cutWidth, cutHeight)) =
      referenceSize(from.width, from.height, side)
    val scaled = Resampling.bicubic(
      from.pixels,
      from.width,
      from.height,
      scaledWidth,
      scaledHeight
    )
    val (left, top) =
      ((scaledWidth - cutWidth) / 2, (scaledHeight - cutHeight) / 2)
    val cut = new Array[Float](cutWidth * cutHeight * 3)
    (0 until cutHeight).foreach(y =>
      System.arraycopy(
        scaled,
        ((top + y) * scaledWidth + left) * 3,
        cut,
        y * cutWidth * 3,
        cutWidth * 3
      )
    )
    Pixels(cut, cutWidth, cutHeight)
  }

  /** A picture at half its sides, each pixel the mean of four. */
  private def halved(from: Pixels): Pixels = {
    val (width, height) = (from.width / 2, from.height / 2)
    val out = new Array[Float](width * height * 3)
    for {
      y <- 0 until height
      x <- 0 until width
      c <- 0 until 3
    } {
      def at(dy: Int, dx: Int) =
        from.pixels(((2 * y + dy) * from.width + 2 * x + dx) * 3 + c)
      out((y * width + x) * 3 + c) =
        math.rint((at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)) / 4.0).toFloat
    }
    Pixels(out, width, height)
  }

  /** `steps` noise levels along the distilled timesteps, then 0: the 28
    * themselves at 28 steps, else the curve through them sampled evenly.
    */
  def distilledSigmas(steps: Int): IndexedSeq[Float] = {
    require(steps >= 1, s"$steps steps")
    val last = DistilledTimesteps.size - 1
    val levels = (0 until steps).map { i =>
      val at = if (steps == 1) 0.0 else i.toDouble * last / (steps - 1)
      val below = math.min(at.toInt, last - 1)
      val fraction = at - below
      val t = DistilledTimesteps(below) * (1 - fraction) +
        DistilledTimesteps(below + 1) * fraction
      (t / 1000).toFloat
    }
    levels :+ 0f
  }

  /** The flash scheduler's noise for the level `sigma`: standard normal,
    * clipped to `NoiseClip` of its measured deviation, times `sigma ×
    * DistilledNoise`.
    */
  private def clippedNoise(
      random: SplittableRandom,
      count: Int,
      sigma: Float
  ): Array[Float] = {
    val noise = Array.fill(count)(Images.gaussian(random))
    val mean = noise.iterator.map(_.toDouble).sum / count
    val deviation = math.sqrt(
      noise.iterator.map(v => (v - mean) * (v - mean)).sum / (count - 1)
    )
    val clip = (NoiseClip * deviation).toFloat
    val scale = sigma * DistilledNoise
    var i = 0
    while (i < count) {
      noise(i) = math.max(-clip, math.min(clip, noise(i))) * scale
      i += 1
    }
    noise
  }

  /** Channels-last pixels (`[H, W, 3]`) as patches, each channel-major: row `l`
    * holds patch `l`'s `(C p1 p2)` values.
    */
  def patches(pixels: Array[Float], width: Int, height: Int): Array[Float] =
    reorder(pixels, width, height, toPatches = true)

  /** Pixels a side to one of a step's preview. */
  val PreviewReduction = 8

  /** A picture in patches (`patches`' layout) as a small image, each pixel the
    * mean of `PreviewReduction`² of its own.
    */
  def reduced(patches: Array[Float], width: Int, height: Int): BufferedImage = {
    val patch = HiDreamO1.Patch
    val cells = patch / PreviewReduction
    val area = PreviewReduction * PreviewReduction
    val (gridHeight, gridWidth) = (height / patch, width / patch)
    val image = new BufferedImage(
      gridWidth * cells,
      gridHeight * cells,
      BufferedImage.TYPE_INT_RGB
    )
    val colours = new Array[Int](3)
    var index = 0
    while (index < gridHeight * gridWidth) {
      val at = index * 3 * patch * patch
      var cell = 0
      while (cell < cells * cells) {
        val (cellY, cellX) = (cell / cells, cell % cells)
        var colour = 0
        while (colour < 3) {
          var sum = 0f
          var y = 0
          while (y < PreviewReduction) {
            val row = at + colour * patch * patch +
              (cellY * PreviewReduction + y) * patch + cellX * PreviewReduction
            var x = 0
            while (x < PreviewReduction) {
              sum += patches(row + x)
              x += 1
            }
            y += 1
          }
          colours(colour) = math.round(
            (math.max(-1f, math.min(1f, sum / area)) * 0.5f + 0.5f) * 255f
          )
          colour += 1
        }
        image.setRGB(
          index % gridWidth * cells + cellX,
          index / gridWidth * cells + cellY,
          (colours(0) << 16) | (colours(1) << 8) | colours(2)
        )
        cell += 1
      }
      index += 1
    }
    image
  }

  /** `patches`' inverse. */
  def pixels(patches: Array[Float], width: Int, height: Int): Array[Float] =
    reorder(patches, width, height, toPatches = false)

  private def reorder(
      values: Array[Float],
      width: Int,
      height: Int,
      toPatches: Boolean
  ): Array[Float] = {
    val patch = HiDreamO1.Patch
    val gridWidth = width / patch
    val out = new Array[Float](values.length)
    for {
      y <- 0 until height
      x <- 0 until width
      c <- 0 until 3
    } {
      val pixel = (y * width + x) * 3 + c
      val patched =
        ((y / patch) * gridWidth + x / patch) * 3 * patch * patch +
          c * patch * patch + (y % patch) * patch + x % patch
      if (toPatches) out(patched) = values(pixel)
      else out(pixel) = values(patched)
    }
    out
  }
}
