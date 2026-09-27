package drift.runner.decode

import drift.runner.models.{CausalModel, GivenRows}
import drift.runner.ops.Ops
import drift.runner.state.Snapshot
import drift.runner.tensor.Tensor
import drift.runner.text.Tokenizer
import drift.runner.vision.PreparedImage

import scala.collection.mutable.ArrayBuffer

/** What one generation produced, and how long each phase took. */
final case class Generated(
    ids: Seq[Int],
    text: String,
    promptTokens: Int,
    /** The prompt's first tokens found already in the sequence, not run again.
      */
    reusedTokens: Int,
    promptSeconds: Double,
    decodeSeconds: Double,
    /** Tokens drafted, and how many of them the model accepted. */
    drafted: Int,
    accepted: Int
) {
  def promptTokensPerSecond: Double =
    (promptTokens - reusedTokens) / promptSeconds
  def tokensPerSecond: Double =
    if (ids.size > 1) (ids.size - 1) / decodeSeconds else 0
}

/** One sequence on one model: its caches (one per layer, `context` tokens), its
  * pages, and the loop that prefills a prompt then decodes, token by token or,
  * with a `speculation`, several tokens per step, drafted then verified.
  *
  * A prompt resumes from what it shares with the sequence instead of starting
  * over: from all it holds (the last prompt and its reply), or else from the
  * latest checkpoint within the shared part. Checkpoints are taken at a
  * prompt's end and before each token that opens a turn (`turnStarts`), so when
  * a template rewrites the last reply (Qwen 3.6 drops its empty reasoning) the
  * next turn resumes from that reply's start. A model without recurrent state
  * or drafter resumes from anywhere. Images count as held when their pictures
  * are the same.
  *
  * A prompt's images go through `vision` (their tokens' embeddings, F32
  * `[tokens, hidden]`, released once prefilled).
  */
final class Generator(
    ops: Ops,
    model: CausalModel,
    tokenizer: Tokenizer,
    context: Int,
    pageSize: Int,
    /** The longest prefill chunk: bounds the activations' memory. */
    prefillChunk: Int,
    speculation: Option[Speculation],
    vision: Option[PreparedImage => Tensor] = None,
    /** The tokens that open a turn (`<|im_start|>`). */
    turnStarts: Set[Int] = Set.empty
) extends AutoCloseable {

  private val sequence = model.newSequence(context, pageSize)
  private val drafter = speculation.map(_.drafter(sequence))

  /** Tokens in the sequence. */
  private var length = 0

  /** The ids at positions `0 until length`. */
  private val held = ArrayBuffer.empty[Int]

  /** The images among them. */
  private var heldImages = Seq.empty[PromptImage]

  /** The state after the first `position` held tokens. */
  final private class Checkpoint(val position: Int, snapshots: Seq[Snapshot])
      extends AutoCloseable {
    def restore(): Unit = snapshots.foreach(_.restore())
    def close(): Unit = snapshots.foreach(_.close())
  }

  /** By position, the latest `Generator.Checkpoints` at most. */
  private val checkpoints = ArrayBuffer.empty[Checkpoint]

  /** A checkpoint of the state after all held tokens. */
  private def keepCheckpoint(): Unit = {
    if (checkpoints.lastOption.exists(_.position == length))
      checkpoints.remove(checkpoints.size - 1).close()
    checkpoints += new Checkpoint(
      length,
      sequence.snapshot() +: drafter.map(_.snapshot()).toSeq
    )
    if (checkpoints.size > Generator.Checkpoints) checkpoints.remove(0).close()
  }

  /** Where the text after the prompt turns: slot `s` at `s + shift`. */
  private var shift = 0

  /** Places text tokens in slots `from until from + count`. */
  private def placeText(from: Int, count: Int): Unit = {
    val turns = Array.range(from + shift, from + shift + count)
    sequence.place(from, turns ++ turns ++ turns)
  }

  /** Runs `ids` from the current length, with `givenRows` the embeddings of the
    * tokens each chunk (its slots `from until end`) covers; returns the last
    * token's logits. With `checkpointing`, a checkpoint is kept before each
    * turn's start.
    */
  private def feed(
      ids: Array[Int],
      givenRows: (Int, Int) => Seq[GivenRows] = (_, _) => Nil,
      checkpointing: Boolean = false
  ): Tensor = {
    var logits = Option.empty[Tensor]
    val turns =
      if (checkpointing) ids.indices.filter(i => turnStarts(ids(i))) else Nil
    var at = 0
    (turns :+ ids.length).foreach { end =>
      ids
        .slice(at, end)
        .grouped(prefillChunk)
        .foreach(chunk => logits = Some(run(chunk, givenRows)))
      if (end < ids.length) keepCheckpoint()
      at = end
    }
    logits.get
  }

  /** One chunk of `feed`. */
  private def run(
      chunk: Array[Int],
      givenRows: (Int, Int) => Seq[GivenRows]
  ): Tensor = {
    sequence.reserve(length + chunk.length)
    val logits = model.forward(
      chunk,
      length,
      sequence,
      allLogits = false,
      givenRows(length, length + chunk.length)
    )
    drafter.foreach(_.observed(chunk, length, chunk.length))
    length += chunk.length
    held ++= chunk
    logits
  }

  /** Runs the prompt from slot `from`: its images through the vision tower
    * first, their rows given to the chunks they fall in.
    */
  private def prefill(prompt: Prompt, from: Int): Tensor = {
    sequence.place(from, prompt.positionsFrom(from))
    val fresh = prompt.images.filter(_.at >= from)
    if (fresh.nonEmpty && vision.isEmpty)
      throw new IllegalArgumentException("this model reads text only")
    val encoded = ArrayBuffer.empty[(PromptImage, Tensor)]
    try {
      vision.foreach(encode =>
        fresh.foreach(i => encoded += i -> encode(i.image))
      )
      val logits = feed(
        prompt.ids.drop(from),
        (start, end) =>
          encoded.toSeq.collect {
            case (image, rows) if image.at < end && image.end > start =>
              val first = math.max(image.at, start)
              val last = math.min(image.end, end)
              GivenRows(
                first - start,
                rows.rows(first - image.at, last - first)
              )
          },
        checkpointing = true
      )
      heldImages = prompt.images
      logits
    } finally encoded.foreach((_, rows) => ops.release(rows))
  }

  /** Keeps what the sequence holds of `prompt`'s beginning; returns how many of
    * its tokens are in place.
    */
  private def resume(prompt: Prompt): Int = {
    val shared = prompt.sharedPrefix(held, heldImages)
    if (shared == length) length
    else {
      val anywhere = sequence.recurrent.isEmpty && drafter.isEmpty
      val back =
        if (anywhere) shared
        else checkpoints.findLast(_.position <= shared).fold(0)(_.position)
      if (back == 0) reset()
      else {
        while (checkpoints.lastOption.exists(_.position > back))
          checkpoints.remove(checkpoints.size - 1).close()
        if (!anywhere) checkpoints.last.restore()
        length = back
        held.dropRightInPlace(held.length - back)
        heldImages = heldImages.filter(_.end <= back)
      }
      length
    }
  }

  /** Forgets everything: the next generation starts from an empty cache. */
  def reset(): Unit = {
    sequence.reset()
    drafter.foreach(_.reset())
    length = 0
    held.clear()
    heldImages = Nil
    checkpoints.foreach(_.close())
    checkpoints.clear()
  }

  /** Prefills `prompt`, then samples up to `maxTokens`, stopping after any of
    * `stops`; `onText` gets the text as it completes and answers whether to go
    * on.
    */
  def generate(
      prompt: Array[Int],
      maxTokens: Int,
      sampler: Sampler,
      stops: Set[Int],
      onText: String => Boolean
  ): Generated =
    generate(Prompt.text(prompt), maxTokens, sampler, stops, onText)

  def generate(
      prompt: Prompt,
      maxTokens: Int,
      sampler: Sampler,
      stops: Set[Int],
      onText: String => Boolean
  ): Generated = {
    val decoder = tokenizer.streamingDecoder(skipSpecial = true)
    val started = System.nanoTime()
    val reused = resume(prompt)
    shift = prompt.shift
    val logits = prefill(prompt, reused)
    keepCheckpoint()
    val prefilled = System.nanoTime()
    val out = Vector.newBuilder[Int]
    var (produced, drafted, accepted) = (0, 0, 0)

    /** Hands `token` on; answers whether to go on after it. */
    def emit(token: Int): Boolean = {
      out += token
      produced += 1
      val goOn = onText(decoder.next(token))
      !stops(token) && goOn && produced < maxTokens
    }

    // `token` is handed on but not run yet
    var token = if (maxTokens > 0) sample(sampler, logits).head else 0
    var going = maxTokens > 0 && emit(token)
    while (going) {
      val budget = speculation.fold(0)(s =>
        math.min(s.drafts, math.min(maxTokens - produced, context - length) - 1)
      )
      drafter.filter(_ => budget > 0) match {
        case None =>
          placeText(length, 1)
          token = sample(sampler, feed(Array(token))).head
          going = emit(token)
        case Some(drafter) =>
          sequence.reserve(length + 1 + budget)
          placeText(length, 1 + budget)
          val drafts = drafter.propose(token, length, budget)
          val ids = token +: drafts
          val rows = sample(sampler, model.verify(ids, length, sequence))
          // sampling the model's own distributions, in order: the same tokens
          // as without drafts, the drafts only saving the runs they match
          var matched = 0
          var next = rows.head
          while (going && matched < drafts.length && next == drafts(matched)) {
            going = emit(next)
            matched += 1
            if (going) next = rows(matched)
          }
          val kept = 1 + matched
          if (kept < ids.length) sequence.restore(kept - 1)
          drafter.observed(ids, length, kept)
          length += kept
          held ++= ids.take(kept)
          drafted += drafts.length
          accepted += matched
          if (going) {
            token = next
            going = emit(token)
          }
      }
    }
    onText(decoder.finish())
    val finished = System.nanoTime()
    val ids = out.result()
    Generated(
      ids,
      tokenizer.decode(ids, skipSpecial = true),
      prompt.ids.length,
      reused,
      (prefilled - started) / 1e9,
      (finished - prefilled) / 1e9,
      drafted,
      accepted
    )
  }

  /** Each row of `logits`' token, drawn in order: from the row's candidates,
    * the row itself coming back only when the sampler needs it.
    */
  private def sample(sampler: Sampler, logits: Tensor): LazyList[Int] = {
    val candidates = ops.candidates(logits, sampler.candidateCount)
    LazyList.tabulate(candidates.size)(row =>
      sampler.next(candidates(row), ops.toFloats(logits.rows(row, 1)))
    )
  }

  def close(): Unit = {
    checkpoints.foreach(_.close())
    drafter.foreach(_.close())
    sequence.close()
  }
}

object Generator {

  /** Checkpoints kept at most: a turn's copy of Qwen 3.8 Flash Next's recurrent
    * states is about 115 MB.
    */
  val Checkpoints = 8
}
