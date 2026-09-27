package drift.runner

import drift.runner.models.{Drafting, Models}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** A golden tiny model (`fixtures/tiny_models.py`) on a backend: its 20 tokens
  * at once, or `prefill` of them then the rest one at a time, giving every
  * position's logits.
  */
final class TinyModelCase(folder: String) {

  lazy val (ids: Array[Int], expected: Array[Float]) =
    Fixtures.withSafetensors(s"tiny/$folder/expected.safetensors") { golden =>
      (golden("ids").decode().map(_.toInt), golden("logits").decode())
    }

  def run(ops: Ops, prefill: Int): Array[Float] = {
    val model =
      Models.open(
        ops,
        Fixtures.path(s"tiny/$folder/model.safetensors"),
        TinyModelCase.draftModel(folder)
      )
    val sequence = model.newSequence(64, 16)
    try {
      sequence.reserve(ids.length)
      val first = ops.toFloats(
        model.forward(ids.take(prefill), 0, sequence, allLogits = true)
      )
      val rest = (prefill until ids.length).flatMap { position =>
        ops.toFloats(
          model.forward(
            Array(ids(position)),
            position,
            sequence,
            allLogits = true
          )
        )
      }
      first ++ rest
    } finally {
      sequence.close()
      model.close()
    }
  }

  /** The largest gap between the logits of `count` tokens verified at once and
    * the same tokens decoded one at a time, after a prefill of `prefill`: a
    * backend must give none, or drafting would change greedy output.
    */
  def verifyGap(ops: Ops, prefill: Int, count: Int): Double = {
    val model =
      Models.open(
        ops,
        Fixtures.path(s"tiny/$folder/model.safetensors"),
        TinyModelCase.draftModel(folder)
      )
    val tokens = ids.slice(prefill, prefill + count)
    def prefilled() = {
      val sequence = model.newSequence(64, 16)
      sequence.reserve(ids.length)
      ops.toFloats(
        model.forward(ids.take(prefill), 0, sequence, allLogits = false)
      )
      sequence
    }
    val (alone, together) = (prefilled(), prefilled())
    try {
      val decoded = tokens.indices.flatMap { i =>
        ops.toFloats(
          model.forward(Array(tokens(i)), prefill + i, alone, allLogits = false)
        )
      }
      val verified = ops.toFloats(model.verify(tokens, prefill, together))
      decoded.indices.map(i => math.abs(decoded(i) - verified(i)).toDouble).max
    } finally {
      alone.close()
      together.close()
      model.close()
    }
  }

  /** The MTP layer's logits for its entries 8 and 18 (the fixture's rows),
    * drafted in two calls: entries 0–8, then 9–18 through the cache.
    */
  def drafts(ops: Ops): (Array[Float], Array[Float]) = {
    val model = Models
      .open(
        ops,
        Fixtures.path(s"tiny/$folder/model.safetensors"),
        TinyModelCase.draftModel(folder)
      )
      .asInstanceOf[Drafting]
    val sequence = model.newSequence(64, 16)
    val hidden = ops.allocate(DType.F32, Shape.of(ids.length, model.hiddenSize))
    try {
      assert(model.drafts, "the fixture carries an MTP layer")
      sequence.reserve(ids.length)
      model.forward(ids, 0, sequence, allLogits = false)
      ops.copy(model.hidden, hidden)
      val first = ops.toFloats(
        model.draft(ids.slice(1, 10), hidden.rows(0, 9), 0, sequence)
      )
      val last = ops.toFloats(
        model.draft(ids.slice(10, 20), hidden.rows(9, 10), 9, sequence)
      )
      (first, last)
    } finally {
      ops.release(hidden)
      sequence.close()
      model.close()
    }
  }

  lazy val expectedDrafts: Array[Float] =
    Fixtures.withSafetensors(s"tiny/$folder/expected.safetensors")(
      _("mtp_logits").decode()
    )

  /** The worst error of both drafts relative to the largest expected logit. */
  def draftError(ops: Ops): Double = {
    val (first, last) = drafts(ops)
    val vocabulary = first.length
    def row(r: Int) = expectedDrafts.slice(r * vocabulary, (r + 1) * vocabulary)
    val scale = expectedDrafts.map(math.abs).max.toDouble
    Seq(first -> row(8), last -> row(18)).map { (actual, wanted) =>
      actual.indices.map(i => math.abs(actual(i) - wanted(i))).max / scale
    }.max
  }

  /** The worst error relative to the logits' largest magnitude, over the first
    * `positions` (all of them when not given).
    */
  def relativeError(
      actual: Array[Float],
      positions: Int = ids.length
  ): Double = {
    val scale = expected.map(math.abs).max.toDouble
    val compared = expected.length / ids.length * positions
    (0 until compared).map(i => math.abs(actual(i) - expected(i))).max / scale
  }
}

object TinyModelCase {

  /** The fixture's MTP head in a file of its own, when it has one. */
  def draftModel(folder: String): Option[java.nio.file.Path] =
    Option(getClass.getResource(s"/fixtures/tiny/$folder/mtp.safetensors"))
      .map(_ => Fixtures.path(s"tiny/$folder/mtp.safetensors"))

  val Qwen3 = new TinyModelCase("qwen3")

  /** Qwen 3 as a text encoder, its worst hidden-state error relative to the
    * largest after 0 and 1 layers: the whole prompt, or its first 12 tokens
    * padded to 20 with id 7 (masked as keys, as FLUX.2 encodes).
    */
  def encodeError(ops: Ops, padded: Boolean): Double = {
    val model = drift.runner.models.Qwen3
      .open(ops, Fixtures.path("tiny/qwen3/model.safetensors"))
    try {
      val ids = Qwen3.ids
      val expected =
        Fixtures.withSafetensors("tiny/qwen3/expected.safetensors")(
          _(if (padded) "padded_hidden" else "hidden").decode()
        )
      val out =
        ops.allocate(DType.F32, Shape.of(2 * ids.length, model.config.hidden))
      try {
        if (padded) model.encode(ids.take(12), Seq(0, 1), out, 8, 7)
        else model.encode(ids, Seq(0, 1), out)
        val actual = ops.toFloats(out)
        val scale = expected.map(math.abs).max.toDouble
        expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
      } finally ops.release(out)
    } finally model.close()
  }

  /** Gemma 2 as a text encoder: the worst errors, relative to the largest, of
    * the residual stream after 0, 1 and 2 layers, and of the final normed
    * state.
    */
  def gemma2EncodeErrors(ops: Ops): (Double, Double) = {
    val model = drift.runner.models.Gemma2
      .open(ops, Fixtures.path("tiny/gemma2/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/gemma2/expected.safetensors") { golden =>
        val ids = golden("ids").decode().map(_.toInt)
        def error(actual: Array[Float], expected: Array[Float]) = {
          val scale = expected.map(math.abs).max.toDouble
          expected.indices
            .map(i => math.abs(actual(i) - expected(i)))
            .max / scale
        }
        val hidden = model.config.hidden.toLong
        val taps = ops.allocate(DType.F32, Shape.of(3 * ids.length, hidden))
        val normed = ops.allocate(DType.F32, Shape.of(ids.length, hidden))
        try {
          model.encode(ids, Seq(0, 1, 2), taps)
          model.lastHiddenState(ids, normed)
          (
            error(ops.toFloats(taps), golden("hidden").decode()),
            error(ops.toFloats(normed), golden("normed").decode())
          )
        } finally {
          ops.release(taps)
          ops.release(normed)
        }
      }
    finally model.close()
  }

  /** Mistral as a text encoder (FLUX.2 [dev]'s): the worst error, relative to
    * the largest, of the residual stream after 1 and 2 layers, read from
    * `file`: transformers' safetensors or llama.cpp's GGUF (q and k permuted).
    */
  def mistralEncodeError(ops: Ops, file: String): Double = {
    val model = drift.runner.models.Mistral
      .open(ops, Fixtures.path(s"tiny/mistral/$file"))
    try
      Fixtures.withSafetensors("tiny/mistral/expected.safetensors") { golden =>
        val ids = golden("ids").decode().map(_.toInt)
        val expected = golden("hidden").decode()
        val out = ops.allocate(
          DType.F32,
          Shape.of(2L * ids.length, model.config.hidden)
        )
        try {
          model.encode(ids, Seq(1, 2), out)
          val actual = ops.toFloats(out)
          val scale = expected.map(math.abs).max.toDouble
          expected.indices
            .map(i => math.abs(actual(i) - expected(i)))
            .max / scale
        } finally ops.release(out)
      }
    finally model.close()
  }

  val Qwen35 = new TinyModelCase("qwen35")
  val Qwen35Moe = new TinyModelCase("qwen35moe")

  /** Its sparse attention keeps 2 blocks of 4: positions up to 10 see every
    * token, and the runner's dense attention is exact there.
    */
  val Qwen4Exp = new TinyModelCase("qwen4exp")
  val Qwen4ExpDensePositions = 11
}
