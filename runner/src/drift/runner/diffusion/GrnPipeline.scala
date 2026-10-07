package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.Unigram

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.SplittableRandom

/** GRN text to image end to end (bytedance's `GRNPipeline`,
  * `autoregressive_infer`): no diffusion. A picture is 256 bits a latent (the
  * HBQ tokenizer's 64 channels, four bits each, coarse to fine); it starts as
  * random bits, and at every step the transformer predicts all of them from the
  * picture as it stands, bits are drawn from the prediction, and a growing
  * random share of them is kept, the rest staying the starting noise's. The
  * last step's draw is the picture, decoded by the tokenizer.
  *   - The prompt, after `<T2I>`, through umT5 (as Wan's pipelines encode it);
  *     `--cfg-scale` above 1 runs the negative prompt too and steers the logits
  *     (3 is the released setting). The logits are divided by a temperature of
  *     1.1, the released example's.
  *   - The share kept after step `s` of `n` is `0.95 × (1 − cos(π / 2 × (s + 1)
  *     / (n − 1)))`; the progress the transformer is told is the share kept in
  *     fact. 50 steps are the released setting.
  * The released code ignores its seed; here the seed is the picture.
  */
final class GrnPipeline(
    ops: Ops,
    transformerFile: Path,
    tokenizerModel: Path,
    textEncoder: Path,
    textTokenizer: Path
) extends ImagePipeline {

  def family: String = "GRN"
  def takesInitImage: Boolean = false
  def takesReferences: Boolean = false
  def takesLoras: Boolean = false
  override def ownShift: Boolean = true
  def takesGuidance: Boolean = false

  private val tokenizer = Unigram.load(textTokenizer)
  private val encoder = Umt5.open(ops, textEncoder)
  private val transformer = Grn.open(ops, transformerFile)
  private val pictures = QwenImage21Vae.open(ops, tokenizerModel)

  private val Temperature = 1.1f
  private val TextTokens = 512

  /** The rounds a channel's value is written in: bits over the channels. */
  private val rounds = transformer.config.bits / pictures.channels.toInt
  require(
    rounds * pictures.channels == transformer.config.bits && pictures.pixelPatch == 2,
    s"${transformer.config.bits} bits a token over the tokenizer's ${pictures.channels} channels"
  )
  require(
    encoder.width == transformer.config.textWidth,
    s"the transformer takes ${transformer.config.textWidth} text features, not the encoder's ${encoder.width}"
  )

  private def text(prompt: String): Tensor = {
    val cleaned = ("<T2I>" + prompt).trim.replaceAll("\\s+", " ")
    val ids = tokenizer.encode(cleaned, addSpecial = true).take(TextTokens)
    val features = encoder.encode(ids)
    try transformer.text(features)
    finally ops.release(features)
  }

  def generate(
      request: ImageRequest,
      progress: (Int, Int) => Unit
  ): BufferedImage = {
    val scale = pictures.scale
    require(
      request.width % scale == 0 && request.height % scale == 0,
      s"${request.width} × ${request.height}: GRN takes multiples of $scale"
    )
    val (gridHeight, gridWidth) =
      (request.height / scale, request.width / scale)
    val latents = gridHeight * gridWidth
    val bits = transformer.config.bits
    val guided = request.cfgScale > 1f
    val conditional = text(request.prompt)
    val unconditional = Option.when(guided)(text(request.negativePrompt))
    val shape = Shape.of(latents, 2L * bits)
    val (own, other) =
      (ops.allocate(DType.F32, shape), ops.allocate(DType.F32, shape))
    try {
      val random = new SplittableRandom(request.seed)
      val noise = Array.fill(latents * bits)(random.nextInt(2).toByte)
      var standing = noise
      var drawn = noise
      var kept = 0f
      val steps = request.steps
      progress(0, steps)
      (0 until steps).foreach { step =>
        val oneHot = ops.fromFloats(
          shape, {
            val values = new Array[Float](latents * 2 * bits)
            standing.indices.foreach(i => values(2 * i + standing(i)) = 1f)
            values
          }
        )
        val logits =
          try {
            transformer.logits(
              oneHot,
              conditional,
              kept,
              gridHeight,
              gridWidth,
              own
            )
            val cond = ops.toFloats(own)
            unconditional.fold(cond) { negative =>
              transformer.logits(
                oneHot,
                negative,
                kept,
                gridHeight,
                gridWidth,
                other
              )
              val uncond = ops.toFloats(other)
              Array.tabulate(cond.length)(i =>
                uncond(i) + request.cfgScale * (cond(i) - uncond(i))
              )
            }
          } finally ops.release(oneHot)
        // each bit drawn from its two logits over the temperature
        drawn = Array.tabulate(latents * bits) { i =>
          val one = 1 / (1 + math.exp(
            (logits(2 * i) - logits(2 * i + 1)).toDouble / Temperature
          ))
          (if (random.nextDouble() < one) 1 else 0).toByte
        }
        // a random share of the draw is kept, the rest stays the noise's
        val share = 0.95 * (1 - math.cos(
          math.Pi / 2 * math.min(1.0, (step + 1.0) / math.max(1, steps - 1))
        ))
        var count = 0
        standing = Array.tabulate(latents * bits) { i =>
          if (random.nextDouble() < share) {
            count += 1
            drawn(i)
          } else noise(i)
        }
        kept = count.toFloat / (latents * bits)
        progress(step + 1, steps)
      }
      // a channel's value: ±1/2, ±1/4, … summed over its rounds' bits
      val channels = pictures.channels.toInt
      val features = Array.tabulate(latents * channels) { i =>
        val (token, channel) = (i / channels, i % channels)
        (0 until rounds).map { round =>
          val interval = math.pow(0.5, round + 1).toFloat
          if (drawn(token * bits + round * channels + channel) == 1) interval
          else -interval
        }.sum
      }
      val latent =
        ops.fromFloats(Shape.of(gridHeight, gridWidth, channels), features)
      val rgb =
        try pictures.decode(latent)
        finally ops.release(latent)
      try Images.toImage(ops.toFloats(rgb), request.width, request.height)
      finally ops.release(rgb)
    } finally
      (Seq(conditional, own, other) ++ unconditional).foreach(ops.release)
  }

  def close(): Unit = {
    pictures.close()
    transformer.close()
    encoder.close()
  }
}
