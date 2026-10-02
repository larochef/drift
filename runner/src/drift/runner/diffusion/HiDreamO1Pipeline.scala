package drift.runner.diffusion

import drift.runner.models.HiDreamO1
import drift.runner.ops.Ops
import drift.runner.state.Sequence
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** HiDream O1 Image end to end (HiDream-ai's `models/pipeline.py`): the prompt
  * in Qwen's chat template, `<|boi_token|><|tms_token|>` after the assistant's
  * turn, cached once; the image as pixel patches from noise of standard
  * deviation 7.5 (8 guided), each step's x̂ from the model.
  *
  * Unguided (CFG 1: the Dev checkpoint) steps as the official "flash"
  * scheduler: the distilled timesteps (`DistilledTimesteps`, resampled for
  * other step counts) and `z = (1 − σ_next) x̂ + σ_next × 7.5 × noise`, fresh
  * noise clipped to 2.5 of its deviation each step. Guided (the full
  * checkpoint, CFG 5) steps by Euler on the flow `(x̂ − z) / σ` over sd-cpp's
  * flow-shifted schedule, the unconditional prompt a space unless a negative
  * prompt is given; the official full pipeline runs UniPC there, not this.
  * img2img starts from the init image mixed with the noise at the step
  * `steps − ⌊steps × strength⌋`. LoRAs apply to every linear, the cached
  * prompt's included.
  */
final class HiDreamO1Pipeline(ops: Ops, modelFile: Path, tokenizerFile: Path)
    extends ImagePipeline {

  import HiDreamO1Pipeline.*

  def family: String = "HiDream O1"
  def takesInitImage: Boolean = true
  def takesReferences: Boolean = false
  def takesLoras: Boolean = true
  override def takesSigmas: Boolean = true
  def takesGuidance: Boolean = false

  private val tokenizer: Tokenizer = TokenizerJson.load(tokenizerFile)
  private val model = HiDreamO1.open(ops, modelFile)
  private val loraFiles = new LoraFiles(ops)

  /** Makes `loras` the model's active set; returns what is left unapplied. */
  private def useLoras(loras: Seq[(Path, Float)]): Seq[String] = {
    val (opened, problems) = loraFiles.open(loras)
    problems ++ model
      .useLoras(opened)
      .map(target => s"a target of no weight: $target")
  }

  /** The prompt's tokens, the timestep token last. */
  private def tokens(prompt: String): Array[Int] = {
    val ids = tokenizer.encode(
      s"<|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>",
      addSpecial = false
    )
    require(
      ids.last == HiDreamO1.TimestepToken,
      s"the tokenizer ends HiDream O1's prompt on ${ids.last}, not <|tms_token|> (${HiDreamO1.TimestepToken}): is $tokenizerFile HiDream O1's tokenizer.json?"
    )
    ids
  }

  /** A prompt cached for every step: its sequence and its timestep token's
    * slot.
    */
  final private class Prompt(text: String, gridHeight: Int, gridWidth: Int)
      extends AutoCloseable {
    private val ids = tokens(text)
    val start: Int = ids.length - 1
    val sequence: Sequence =
      model.newSequence(ids.length, gridHeight, gridWidth)
    try model.prefill(ids.init, sequence)
    catch {
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
    val (gridHeight, gridWidth) =
      (request.height / patch, request.width / patch)
    val count = gridHeight * gridWidth
    val values = count * model.patchValues
    val guided = request.cfgScale > 1f
    useLoras(request.loras).foreach(problem =>
      println(s"[WARN] LoRA left unapplied: $problem")
    )
    // a request's own levels (a turbo LoRA's) in place of either schedule
    val sigmas = request.sigmas.getOrElse(
      if (guided)
        FlowSchedule.sigmas(
          request.steps,
          math.log(request.shift.getOrElse(FlowSchedule.DefaultShift))
        )
      else distilledSigmas(request.steps)
    )
    val noiseScale = if (guided) GuidedNoise else DistilledNoise
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
    val conditional = new Prompt(request.prompt, gridHeight, gridWidth)
    val unconditional = Option.when(guided)(
      new Prompt(
        if (request.negativePrompt.isEmpty) " " else request.negativePrompt,
        gridHeight,
        gridWidth
      )
    )
    val z = ops.fromFloats(shape, start)
    val clean = ops.allocate(DType.F32, shape)
    val other = unconditional.map(_ => ops.allocate(DType.F32, shape))
    try {
      val steps = request.steps - first
      progress(0, steps)
      (first until request.steps).foreach { i =>
        val (sigma, next) = (sigmas(i), sigmas(i + 1))
        model.predict(
          z,
          1 - sigma,
          conditional.start,
          conditional.sequence,
          clean
        )
        for {
          prompt <- unconditional
          x <- other
        } {
          model.predict(z, 1 - sigma, prompt.start, prompt.sequence, x)
          // x̂ = uncond + scale × (cond − uncond), the flow being linear in x̂
          ops.scale(x, 1 - request.cfgScale, x)
          ops.scale(clean, request.cfgScale, clean)
          ops.add(clean, x, clean)
        }
        if (guided) {
          // Euler: z += (σ_next − σ) (x̂ − z) / σ
          val kept = next / math.max(sigma, MinimumSigma)
          ops.scale(z, kept, z)
          ops.scale(clean, 1 - kept, clean)
          ops.add(z, clean, z)
        } else {
          ops.scale(clean, 1 - next, z)
          if (next > 0) {
            val noise =
              ops.fromFloats(shape, clippedNoise(random, values, next))
            try ops.add(z, noise, z)
            finally ops.release(noise)
          }
        }
        progress(i + 1 - first, steps)
      }
      Images.toImage(
        pixels(ops.toFloats(z), request.width, request.height),
        request.width,
        request.height
      )
    } finally {
      (Seq(z, clean) ++ other).foreach(ops.release)
      (Seq(conditional) ++ unconditional).foreach(_.close())
    }
  }

  def close(): Unit = {
    loraFiles.close()
    model.close()
  }
}

object HiDreamO1Pipeline {

  /** The Dev checkpoint's 28 timesteps (`DEFAULT_TIMESTEPS`), σ × 1000. */
  val DistilledTimesteps: IndexedSeq[Int] =
    IndexedSeq(999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790,
      764, 737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8)

  /** The starting noise's deviation and the flash scheduler's (the official
    * `noise_scale_start` and `_end`), unguided.
    */
  val DistilledNoise = 7.5f

  /** The same, guided (`NOISE_SCALE`). */
  val GuidedNoise = 8f

  /** The flash scheduler's noise, clipped to this many of its deviations. */
  val NoiseClip = 2.5f

  /** σ floor of the flow's division (`T_EPS`). */
  val MinimumSigma = 0.001f

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
