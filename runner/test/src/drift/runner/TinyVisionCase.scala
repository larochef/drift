package drift.runner

import drift.runner.decode.{Prompt, PromptImage}
import drift.runner.models.{GivenRows, Models, Qwen3, QwenVision, SeenImages}
import drift.runner.tensor.{DType, Shape}
import drift.runner.ops.Ops
import drift.runner.vision.{ImageSizing, PreparedImage, Resampling}

/** The golden tiny Qwen 3.5 MoE that sees (`fixtures/tiny_models.py
  * qwen35moe_vision`): transformers' image processor on a 64 × 96 image and on
  * a 70 × 100 photo it resizes, the vision tower's six tokens, the mRoPE
  * positions of 22 tokens with the image at 6–11, and their logits.
  */
object TinyVisionCase {

  private val folder = "tiny/qwen35moe_vision"
  private val sizing = ImageSizing.qwen(16, 2)

  private lazy val golden: Map[String, Array[Float]] =
    Fixtures.withSafetensors(s"$folder/expected.safetensors") { file =>
      Seq(
        "ids",
        "image",
        "pixel_values",
        "features",
        "positions",
        "logits",
        "photo",
        "photo_pixel_values",
        "photo_grid"
      ).map(name => name -> file(name).decode()).toMap
    }

  lazy val ids: Array[Int] = golden("ids").map(_.toInt)

  /** transformers' patches (two equal frames each) as the runner's: the first
    * frame.
    */
  private def firstFrames(values: Array[Float]): Array[Float] = {
    val rows = values.length / 1536
    Array.tabulate(rows * 768) { i =>
      val (row, rest) = (i / 768, i % 768)
      values(row * 1536 + (rest / 256) * 512 + rest % 256)
    }
  }

  private def worst(actual: Array[Float], expected: Array[Float]): Double = {
    require(
      actual.length == expected.length,
      s"${actual.length} values, expected ${expected.length}"
    )
    actual.indices.map(i => math.abs(actual(i) - expected(i)).toDouble).max
  }

  private def relative(actual: Array[Float], expected: Array[Float]): Double =
    worst(actual, expected) / expected.map(math.abs).max

  lazy val image: PreparedImage =
    PreparedImage(
      sizing.patches(golden("image").map(_ / 127.5f - 1f), 64, 96),
      4,
      6,
      2,
      "image"
    )

  /** The runner's patches against transformers', in pixel levels. */
  def patchError: Double =
    worst(image.patches, firstFrames(golden("pixel_values"))) * 127.5

  /** The photo resized: the size, and the worst pixel difference in levels. */
  def photoResize: ((Int, Int), Double) = {
    val (height, width) = sizing.size(70, 100)
    val pixels = Resampling.bicubic(golden("photo"), 100, 70, width, height)
    val patches = sizing.patches(pixels.map(_ / 127.5f - 1f), height, width)
    val grid = golden("photo_grid").map(_.toInt)
    require(
      grid(1) * 16 == height && grid(2) * 16 == width,
      s"transformers resized to ${grid.mkString("×")} patches, the runner to ${height}×$width pixels"
    )
    (
      (height, width),
      worst(patches, firstFrames(golden("photo_pixel_values"))) * 127.5
    )
  }

  lazy val prompt: Prompt = Prompt(ids, Seq(PromptImage(6, image)))

  def positionsMatch: Boolean =
    prompt.positions.sameElements(golden("positions").map(_.toInt))

  private def modelPath = Fixtures.path(s"$folder/model.safetensors")

  /** The vision tower's tokens against transformers', relative to the largest.
    */
  def featuresError(ops: Ops): Double = {
    val tower = QwenVision.open(ops, modelPath)
    try {
      val output = tower.encode(image)
      try relative(ops.toFloats(output.tokens), golden("features"))
      finally ops.release(output.tokens)
    } finally tower.close()
  }

  /** The logits of the 22 tokens, `prefill` of them at once and the rest one by
    * one, the image's tokens given by the tower, relative to the largest.
    */
  def logitsError(ops: Ops, prefill: Int): Double = {
    val tower = QwenVision.open(ops, modelPath)
    val model = Models.open(ops, modelPath, None)
    val sequence = model.newSequence(64, 16)
    val features = tower.encode(image).tokens
    try {
      sequence.reserve(ids.length)
      sequence.place(0, prompt.positions)
      def run(from: Int, until: Int): Array[Float] = {
        val (first, last) = (math.max(6, from), math.min(12, until))
        val givenRows =
          Option.when(first < last)(
            GivenRows(first - from, features.rows(first - 6, last - first))
          )
        ops.toFloats(
          model.forward(
            ids.slice(from, until),
            from,
            sequence,
            allLogits = true,
            givenRows.toSeq
          )
        )
      }
      val logits = run(0, prefill) ++
        (prefill until ids.length).flatMap(i => run(i, i + 1))
      relative(logits, golden("logits"))
    } finally {
      ops.release(features)
      sequence.close()
      model.close()
      tower.close()
    }
  }

  /** Qwen3-VL as a text encoder reading an image (`qwen3vl_vision`): its
    * tower's tokens and deepstack taps, and the last layer's residual stream of
    * 16 tokens with the image at 4–9, relative to the largest of each.
    */
  def qwen3vlErrors(ops: Ops): (Double, Double, Double) = {
    val folder = "tiny/qwen3vl_vision"
    val path = Fixtures.path(s"$folder/model.safetensors")
    Fixtures.withSafetensors(s"$folder/expected.safetensors") { file =>
      val ids = file("ids").decode().map(_.toInt)
      val prepared = PreparedImage(
        sizing.patches(file("image").decode().map(_ / 127.5f - 1f), 64, 96),
        4,
        6,
        2,
        "image"
      )
      val tower = QwenVision.open(ops, path)
      val encoder = Qwen3.open(ops, path)
      val seen = tower.encode(prepared)
      val out = ops.allocate(DType.F32, Shape.of(ids.length, 64))
      try {
        val deepstack = seen.deepstack.flatMap(ops.toFloats).toArray
        val prompt = Prompt(ids, Seq(PromptImage(4, prepared)))
        encoder.encode(
          ids,
          Seq(encoder.config.layers),
          out,
          images = Some(
            SeenImages(
              prompt.positions,
              Seq(GivenRows(4, seen.tokens)),
              seen.deepstack.map(rows => Seq(GivenRows(4, rows)))
            )
          )
        )
        (
          relative(ops.toFloats(seen.tokens), file("features").decode()),
          relative(deepstack, file("deepstack").decode()),
          relative(ops.toFloats(out), file("hidden").decode())
        )
      } finally {
        ops.release(out)
        ops.release(seen.tokens)
        seen.deepstack.foreach(ops.release)
        encoder.close()
        tower.close()
      }
    }
  }
}
