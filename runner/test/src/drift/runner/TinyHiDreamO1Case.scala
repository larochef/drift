package drift.runner

import drift.runner.lora.Lora
import drift.runner.lora.LoraReport
import drift.runner.decode.{Prompt, PromptImage}
import drift.runner.diffusion.HiDreamO1Pipeline
import drift.runner.models.{
  GivenRows,
  HiDreamO1,
  HiDreamO1Loras,
  QwenVision,
  SeenImages
}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}
import drift.runner.vision.{ImageSizing, PreparedImage}

/** The golden tiny HiDream O1 (`fixtures/tiny_hidream_o1.py`: the official
  * model class, its masked attention on the CPU): one forward of a prompt and a
  * 2 × 3 grid of patches; the same with two reference pictures, seen by the
  * vision tower in the prompt and read as pixel patches after the target's; and
  * the first with a LoRA of each published naming merged into the weights.
  */
object TinyHiDreamO1Case {

  private val folder = "tiny/hidream_o1"
  private def path = Fixtures.path(s"$folder/model.safetensors")
  private val sizing = ImageSizing.qwen(16, 2)
  private val (gridHeight, gridWidth) = (2, 3)
  private val patches = gridHeight * gridWidth

  private def relative(actual: Array[Float], expected: Array[Float]): Double = {
    require(
      actual.length == expected.length,
      s"${actual.length} values, expected ${expected.length}"
    )
    actual.indices
      .map(i => math.abs(actual(i) - expected(i)).toDouble)
      .max / expected.map(math.abs).max
  }

  /** x̂ of the prompt alone at t = 0.37, the model as `prepare` leaves it. */
  private def text(
      ops: Ops,
      model: HiDreamO1,
      golden: String => Array[Float]
  ): (Array[Int], Array[Float]) = {
    val ids = golden("text_ids").map(_.toInt)
    val prompt = Prompt.text(ids)
    val sequence = model.newSequence(prompt.positions, gridHeight, gridWidth)
    val z = ops.fromFloats(Shape.of(patches, model.patchValues), golden("z"))
    val out = ops.allocate(DType.F32, Shape.of(patches, model.patchValues))
    try {
      model.prefill(ids.init, sequence)
      model.predict(z, 0.37f, ids.length - 1, sequence, out)
      (sequence.positions(0, ids.length + patches, 3), ops.toFloats(out))
    } finally {
      ops.release(z)
      ops.release(out)
      sequence.close()
    }
  }

  private def withGolden[A](body: (String => Array[Float]) => A): A =
    Fixtures.withSafetensors(s"$folder/expected.safetensors")(file =>
      body(name => file(name).decode())
    )

  /** The prompt alone: whether every token sits where the official code puts
    * it, and x̂ against the official model's, relative to the largest.
    */
  def textError(ops: Ops): (Boolean, Double) = withGolden { golden =>
    val model = HiDreamO1.open(ops, path)
    try {
      val (positions, x) = text(ops, model, golden)
      (
        positions.sameElements(golden("text_positions").map(_.toInt)),
        relative(x, golden("text_x"))
      )
    } finally model.close()
  }

  /** Two references — 64 × 96 and 64 × 64, seen as 2 × 3 and 2 × 2 tokens at
    * slots 4 and 12, read as 2 × 3 and 2 × 2 patches — the same two ways.
    */
  def referencesError(ops: Ops): (Boolean, Double) = withGolden { golden =>
    val ids = golden("references_ids").map(_.toInt)
    val sizes = Seq((96, 64), (64, 64))
    val pictures =
      sizes.indices.map(i => golden(s"picture_$i").map(_ / 127.5f - 1f))
    val images = pictures.zip(sizes).zipWithIndex.map {
      case ((pixels, (width, height)), i) =>
        PreparedImage(
          sizing.patches(pixels, height, width),
          height / 16,
          width / 16,
          2,
          s"picture $i"
        )
    }
    val prompt =
      Prompt(ids, Seq(PromptImage(4, images(0)), PromptImage(12, images(1))))
    val grids = sizes.map((width, height) => (height / 32, width / 32))
    val model = HiDreamO1.open(ops, path)
    val tower = QwenVision.open(ops, path)
    val seen = images.map(tower.encode)
    val sequence =
      model.newSequence(prompt.positions, gridHeight, gridWidth, grids)
    val read = grids.map(_ * _).sum
    val z = ops.fromFloats(Shape.of(patches, model.patchValues), golden("z"))
    val pixels = ops.fromFloats(
      Shape.of(read, model.patchValues),
      pictures
        .zip(sizes)
        .flatMap { case (picture, (width, height)) =>
          HiDreamO1Pipeline.patches(picture, width, height)
        }
        .toArray
    )
    val rows =
      ops.allocate(DType.F32, Shape.of(read, model.decoder.config.hidden))
    val out = ops.allocate(DType.F32, Shape.of(patches, model.patchValues))
    try {
      val placed = prompt.images.zip(seen)
      model.prefill(
        ids.init,
        sequence,
        Some(
          SeenImages(
            prompt.positions,
            placed.map((p, s) => GivenRows(p.at, s.tokens)),
            seen.head.deepstack.indices.map(k =>
              placed.map((p, s) => GivenRows(p.at, s.deepstack(k)))
            )
          )
        )
      )
      model.embed(pixels, rows)
      model.predict(z, 0.62f, ids.length - 1, sequence, out, Some(rows))
      (
        sequence
          .positions(0, ids.length + patches + read, 3)
          .sameElements(golden("references_positions").map(_.toInt)),
        relative(ops.toFloats(out), golden("references_x"))
      )
    } finally {
      Seq(z, pixels, rows, out).foreach(ops.release)
      seen.foreach { s =>
        ops.release(s.tokens)
        s.deepstack.foreach(ops.release)
      }
      sequence.close()
      tower.close()
      model.close()
    }
  }

  /** What a LoRA file came to: what the model made of it, the log's lines, x̂
    * with it at 0.8 against the official model with it merged, and how far that
    * x̂ is from the model's own (the update's size, on the same scale).
    */
  final case class LoraOutcome(
      use: HiDreamO1Loras,
      report: Seq[String],
      error: Double,
      moved: Double
  )

  /** `lora_<name>.safetensors` on the prompt alone. */
  def lora(ops: Ops, name: String): LoraOutcome = withGolden { golden =>
    val model = HiDreamO1.open(ops, path)
    val file = Lora.open(ops, Fixtures.path(s"$folder/lora_$name.safetensors"))
    try {
      val (_, plain) = text(ops, model, golden)
      val loras = Seq(file -> 0.8f)
      val use = model.useLoras(loras)
      val (_, changed) = text(ops, model, golden)
      // and gone again once no LoRA is asked for
      model.useLoras(Nil)
      val (_, back) = text(ops, model, golden)
      require(back.sameElements(plain), s"$name: left a trace once removed")
      val expected = golden(s"lora_${name}_x")
      LoraOutcome(
        use,
        LoraReport
          .lines(1, use.updates, use.applied, use.unread, use.unmatched),
        relative(changed, expected),
        relative(plain, expected)
      )
    } finally {
      file.close()
      model.close()
    }
  }
}
