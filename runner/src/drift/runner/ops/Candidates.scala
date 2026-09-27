package drift.runner.ops

/** A row's largest values and their indices, largest first, the lower index
  * first among equals: what sampling draws from.
  */
final case class Candidates(ids: Array[Int], values: Array[Float]) {
  def size: Int = ids.length
}

object Candidates {

  /** A float's rank as a signed int: larger floats rank higher, -0 as 0. */
  def rank(value: Float): Int = {
    val bits = java.lang.Float.floatToRawIntBits(if (value == 0) 0f else value)
    if (bits < 0) bits ^ 0x7fffffff else bits
  }

  /** The indices `0 until count` of `values(from + i)`, largest first, the
    * lower index first among equals: one primitive sort, nothing boxed.
    */
  def order(values: Array[Float], from: Int, count: Int): Array[Int] = {
    // descending rank in the high half, the index in the low half
    val keys = Array.tabulate(count)(i =>
      (~rank(values(from + i))).toLong << 32 | i.toLong
    )
    java.util.Arrays.sort(keys)
    keys.map(_.toInt)
  }

  /** The `count` largest of each `width`-wide row of `values`. */
  def of(values: Array[Float], width: Int, count: Int): Seq[Candidates] =
    (0 until values.length / width).map { row =>
      val ids = order(values, row * width, width).take(count)
      Candidates(ids, ids.map(id => values(row * width + id)))
    }
}
