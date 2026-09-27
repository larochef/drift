package drift.runner.text

import java.util.PriorityQueue

/** Byte-pair merging over token ids: the pair of adjacent symbols with the
  * lowest merge rank is merged first, the leftmost among equals, until no
  * adjacent pair has a merge — HuggingFace's `Word::merge_all`.
  *
  * `merges` maps a pair `(left << 32) | right` to `(rank << 32) | merged`.
  */
final class Bpe(merges: java.util.HashMap[java.lang.Long, java.lang.Long]) {

  def mergeCount: Int = merges.size

  private def pair(left: Int, right: Int): Long =
    (left.toLong << 32) | (right & 0xffffffffL)

  /** Merges `symbols` (ids) in place of a copy; returns the merged ids. */
  def merge(symbols: Array[Int]): Array[Int] = {
    val count = symbols.length
    if (count < 2) return symbols
    val ids = symbols.clone()
    val previous = Array.tabulate(count)(_ - 1)
    val next = Array.tabulate(count)(i => if (i + 1 < count) i + 1 else -1)
    val alive = Array.fill(count)(true)
    // (rank, position, left id, right id): stale entries are skipped on pop
    val queue = new PriorityQueue[Array[Long]]((a, b) =>
      if (a(0) != b(0)) java.lang.Long.compare(a(0), b(0))
      else java.lang.Long.compare(a(1), b(1))
    )
    def offer(position: Int): Unit = {
      val right = next(position)
      if (right >= 0) {
        val found = merges.get(pair(ids(position), ids(right)))
        if (found != null)
          queue.add(
            Array(
              found >>> 32,
              position.toLong,
              ids(position).toLong,
              ids(right).toLong
            )
          )
      }
    }
    (0 until count - 1).foreach(offer)
    while (!queue.isEmpty) {
      val top = queue.poll()
      val position = top(1).toInt
      val right = next(position)
      // still the same pair at this position?
      if (
        alive(position) && right >= 0 && ids(position) == top(2).toInt && ids(
          right
        ) == top(3).toInt
      ) {
        val merged =
          (merges.get(pair(ids(position), ids(right))) & 0xffffffffL).toInt
        ids(position) = merged
        alive(right) = false
        next(position) = next(right)
        if (next(right) >= 0) previous(next(right)) = position
        if (previous(position) >= 0) offer(previous(position))
        offer(position)
      }
    }
    ids.indices.filter(alive).map(ids).toArray
  }
}
