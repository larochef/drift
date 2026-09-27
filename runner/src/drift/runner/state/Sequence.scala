package drift.runner.state

import drift.runner.ops.Ops
import drift.runner.tensor.*

/** One sequence's memory in a model (`specs/42`, Chat requirements): the paged
  * key-value caches of its attention layers, one page table for all of them,
  * and the carried F32 states of its recurrent layers (the short convolution's
  * last inputs, the delta rule's matrix), zero when it starts.
  */
final class Sequence(
    ops: Ops,
    val context: Int,
    val pageSize: Int,
    attentionLayers: Int,
    kvHeads: Int,
    headDimension: Int,
    recurrentShapes: Seq[Shape]
) extends AutoCloseable {

  private val pageCount = (context + pageSize - 1) / pageSize
  private val pages = new SequencePages(new PageAllocator(pageCount), pageSize)

  val caches: Seq[KvCache] =
    (0 until attentionLayers).map(_ =>
      ops.allocateCache(pageCount, pageSize, kvHeads, headDimension)
    )

  val recurrent: Seq[Tensor] = recurrentShapes.map { shape =>
    val tensor = ops.allocate(DType.F32, shape)
    ops.zero(tensor)
    tensor
  }

  /** Each slot's rotary position on each axis (temporal, height, width),
    * axis-major: a text token's three are equal, an image's tokens sit on its
    * grid, and after an image a text token's position trails its slot (an image
    * takes fewer positions than tokens). Slot `s` sits at `s` until placed
    * otherwise.
    */
  private val rotary = Array.tabulate(3 * context)(_ % context)

  /** Places slots `start until start + n` at `positions` (`[3, n]`,
    * axis-major).
    */
  def place(start: Int, positions: Array[Int]): Unit = {
    val n = positions.length / 3
    (0 until 3).foreach(a =>
      System.arraycopy(positions, a * n, rotary, a * context + start, n)
    )
  }

  /** The positions of slots `start until start + count` on the first `axes`
    * axes, axis-major.
    */
  def positions(start: Int, count: Int, axes: Int): Array[Int] = {
    val out = new Array[Int](axes * count)
    (0 until axes).foreach(a =>
      System.arraycopy(rotary, a * context + start, out, a * count, count)
    )
    out
  }

  private val table: Tensor =
    ops.fromInts(Shape.of(pageCount), new Array[Int](pageCount))

  /** The page table the attention layers read. */
  def pageTable: Tensor = table

  /** Makes room for positions up to `until - 1`. */
  def reserve(until: Int): Unit = {
    if (until > context)
      throw new IllegalStateException(
        s"$until tokens exceed the context of $context"
      )
    val before = pages.table.length
    pages.reserve(until)
    if (pages.table.length != before)
      ops.writeInts(table, pages.table.padTo(pageCount, 0))
  }

  /** Each recurrent state after each token of a verified run, room for
    * `historyTokens`: speculative decoding rolls back to one of them.
    */
  private var kept = Seq.empty[Tensor]
  private var historyTokens = 0

  /** Room for `tokens` states per recurrent state, aligned with `recurrent`. */
  def history(tokens: Int): Seq[Tensor] = {
    if (tokens > historyTokens) {
      kept.foreach(ops.release)
      kept = recurrent.map(state =>
        ops.allocate(DType.F32, Shape(tokens.toLong +: state.shape.dimensions))
      )
      historyTokens = tokens
    }
    kept
  }

  /** The recurrent states back to what they were after token `index` of the
    * last run that kept a history.
    */
  def restore(index: Int): Unit = {
    require(
      recurrent.isEmpty || index < historyTokens,
      s"no state kept for token $index"
    )
    recurrent
      .zip(kept)
      .foreach((state, history) =>
        ops.copy(history.rows(index, 1).view(state.shape.dimensions*), state)
      )
  }

  /** A copy of the recurrent states as they are, taken at a turn's start or a
    * prompt's end: a later prompt that matches up to there resumes from it (the
    * caches before it are kept; the caller knows its length).
    */
  def snapshot(): Snapshot = {
    val copies = recurrent.map { state =>
      val copy = ops.allocate(DType.F32, state.shape)
      ops.copy(state, copy)
      copy
    }
    new Snapshot {
      def restore(): Unit =
        recurrent.zip(copies).foreach((state, copy) => ops.copy(copy, state))
      def close(): Unit = copies.foreach(ops.release)
    }
  }

  /** Back to empty: the pages go back and the recurrent states are zero. */
  def reset(): Unit = {
    pages.release()
    recurrent.foreach(ops.zero)
  }

  def close(): Unit = {
    kept.foreach(ops.release)
    caches.foreach(cache => {
      ops.release(cache.keys); ops.release(cache.values)
    })
    recurrent.foreach(ops.release)
    ops.release(table)
  }
}
