package drift.runner.decode

import drift.runner.models.{CausalModel, Drafting}
import drift.runner.ops.Ops
import drift.runner.state.{Sequence, Snapshot}
import drift.runner.tensor.{DType, Shape}

import scala.collection.mutable.ArrayBuffer

/** Guesses the tokens after the next for speculative decoding (`specs/42`, step
  * 9). The target model checks every guess, so a poor drafter costs speed,
  * never correctness.
  */
trait Drafter extends AutoCloseable {

  /** The target just ran `ids` at positions `start` on and keeps the first
    * `kept` of them; the model's hidden states are still theirs.
    */
  def observed(ids: Array[Int], start: Int, kept: Int): Unit

  /** Up to `count` tokens guessed to follow `last`, which sits at `position`
    * and has not run yet.
    */
  def propose(last: Int, position: Int, count: Int): Array[Int]

  /** Back to an empty sequence. */
  def reset(): Unit

  /** Where the drafter is, taken beside a snapshot of the sequence. */
  def snapshot(): Snapshot
}

/** How many tokens to draft per step, and the drafter for a sequence. */
final case class Speculation(drafts: Int, drafter: Sequence => Drafter)

object Speculation {

  /** Drafting through the model's own MTP layer, when its weights carry one. */
  def multiToken(
      ops: Ops,
      model: CausalModel,
      drafts: Int,
      prefillChunk: Int
  ): Option[Speculation] =
    model match {
      case drafting: Drafting if drafting.drafts && drafts > 0 =>
        Some(
          Speculation(
            drafts,
            sequence =>
              new MultiTokenDrafter(
                ops,
                drafting,
                sequence,
                drafts,
                prefillChunk
              )
          )
        )
      case _ => None
    }
}

/** Drafts with the model's MTP layer, greedily. Its cache entry at position `i`
  * pairs token `i + 1` with the target's hidden state at `i`, so an entry waits
  * for the token after its hidden state: those waiting are `pending`, run with
  * the next proposal (or, past `drafts + 1` of them, as a prefill goes, at
  * once). A chained draft pairs the token drafted with the MTP's own output
  * state.
  */
final class MultiTokenDrafter(
    ops: Ops,
    model: Drafting,
    sequence: Sequence,
    drafts: Int,
    prefillChunk: Int
) extends Drafter {

  private val capacity = prefillChunk + drafts + 1

  /** MTP entries `0 until written` are in its cache. */
  private var written = 0

  /** The target's length: its hidden states `written until known` wait. */
  private var known = 0

  /** The tokens at positions `written + 1 until known`: the waiting entries'
    * but the last's, whose token is not known yet.
    */
  private val waiting = ArrayBuffer.empty[Int]

  private val pending =
    ops.allocate(DType.F32, Shape.of(capacity, model.hiddenSize))
  private val chained = ops.allocate(DType.F32, Shape.of(1, model.hiddenSize))

  def observed(ids: Array[Int], start: Int, kept: Int): Unit = {
    require(
      start == known,
      s"observed tokens at $start; the drafter is at $known"
    )
    val rows = known - written
    ops.copy(model.hidden.rows(0, kept), pending.rows(rows, kept))
    (0 until kept).foreach(i => if (start + i > written) waiting += ids(i))
    known += kept
    if (known - written > drafts + 1) {
      // a prefill: the entries whose token is known run now
      val entries = known - written - 1
      model.draft(waiting.toArray, pending.rows(0, entries), written, sequence)
      ops.copy(pending.rows(entries, 1), pending.rows(0, 1))
      written += entries
      waiting.clear()
    }
  }

  def propose(last: Int, position: Int, count: Int): Array[Int] = {
    require(
      position == known && known > written,
      s"a draft at $position; the drafter is at $known"
    )
    val guesses = new Array[Int](count)
    var logits = model.draft(
      (waiting :+ last).toArray,
      pending.rows(0, known - written),
      written,
      sequence
    )
    written = known
    waiting.clear()
    guesses(0) = ops.argmax(logits)
    (1 until count).foreach { j =>
      ops.copy(model.draftHidden, chained)
      logits =
        model.draft(Array(guesses(j - 1)), chained, position + j - 1, sequence)
      guesses(j) = ops.argmax(logits)
    }
    guesses
  }

  def reset(): Unit = {
    written = 0
    known = 0
    waiting.clear()
  }

  def snapshot(): Snapshot = {
    val (w, k, tokens) = (written, known, waiting.toArray)
    // only the waiting entries' hidden states matter
    val rows = Option.when(k > w) {
      val copy = ops.allocate(DType.F32, Shape.of(k - w, model.hiddenSize))
      ops.copy(pending.rows(0, k - w), copy)
      copy
    }
    new Snapshot {
      def restore(): Unit = {
        written = w
        known = k
        waiting.clear()
        waiting ++= tokens
        rows.foreach(r => ops.copy(r, pending.rows(0, k - w)))
      }
      def close(): Unit = rows.foreach(ops.release)
    }
  }

  def close(): Unit = {
    ops.release(pending)
    ops.release(chained)
  }
}
