package drift.runner.state

import scala.collection.mutable

/** Hands out a cache's pages to sequences. Host-side bookkeeping only: the
  * pages themselves stay where the cache allocated them.
  */
final class PageAllocator(pages: Int) {

  private val free = mutable.ArrayDeque.from(0 until pages)

  def available: Int = free.size

  def take(count: Int): Seq[Int] = {
    if (count > free.size)
      throw new IllegalStateException(s"$count pages wanted, ${free.size} free")
    Seq.fill(count)(free.removeHead())
  }

  def give(returned: Seq[Int]): Unit = free ++= returned
}

/** The pages of one sequence, in position order: its page table. */
final class SequencePages(allocator: PageAllocator, val pageSize: Int) {

  private val owned = mutable.ArrayBuffer.empty[Int]

  def table: Array[Int] = owned.toArray

  /** Makes room for positions up to `length - 1`. */
  def reserve(length: Int): Unit = {
    val needed = (length + pageSize - 1) / pageSize - owned.size
    if (needed > 0) owned ++= allocator.take(needed)
  }

  /** Forgets positions from `length` on (a rejected draft, a restart); whole
    * pages past it go back.
    */
  def truncate(length: Int): Unit = {
    val kept = (length + pageSize - 1) / pageSize
    if (kept < owned.size) {
      allocator.give(owned.drop(kept).toSeq)
      owned.dropRightInPlace(owned.size - kept)
    }
  }

  def release(): Unit = truncate(0)
}
