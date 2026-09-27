package drift.runner

import scala.util.Random

import drift.runner.ops.{Attention, Ops}
import drift.runner.state.{KvCache, PageAllocator, SequencePages}
import drift.runner.tensor.{DType, Shape, Tensor}

/** One attention call with seeded inputs, runnable on any backend: `keyCount`
  * cached keys written through a page table whose pages come in a shuffled
  * order, then `tokens` queries at the end of them.
  */
final case class AttentionCase(
    tokens: Int,
    keyCount: Int,
    qHeads: Int,
    kvHeads: Int,
    dimension: Int,
    causal: Boolean,
    window: Option[Int],
    softcap: Option[Float],
    sinks: Boolean,
    pageSize: Int,
    shuffledPages: Boolean,
    seed: Long,
    /** Replaces the seeded queries when set. */
    givenQueries: Option[Array[Float]] = None
) {
  def queryStart: Int = keyCount - tokens

  val pageCount: Int = (keyCount + pageSize - 1) / pageSize + 3

  /** The page table: positions' pages, shuffled or in order. */
  val table: Array[Int] = {
    val allocator = new PageAllocator(pageCount)
    val pages = new SequencePages(allocator, pageSize)
    pages.reserve(keyCount)
    val inOrder = pages.table
    if (shuffledPages) new Random(seed).shuffle(inOrder.toSeq).toArray
    else inOrder
  }

  val keys: Array[Float] =
    TestData.gaussian(seed, keyCount * kvHeads * dimension)
  val values: Array[Float] =
    TestData.gaussian(seed + 1, keyCount * kvHeads * dimension)
  val queries: Array[Float] =
    givenQueries.getOrElse(
      TestData.gaussian(seed + 2, tokens * qHeads * dimension)
    )
  val sinkValues: Array[Float] = TestData.gaussian(seed + 3, qHeads)

  /** The outputs `[tokens, qHeads, dimension]`, written into a fresh cache. */
  def run(ops: Ops): Array[Float] = {
    val cache = ops.allocateCache(pageCount, pageSize, kvHeads, dimension)
    val pageTable = ops.fromInts(Shape.of(table.length), table)
    write(ops, cache, pageTable, 0, keyCount)
    val out = ops.allocate(DType.F32, Shape.of(tokens, qHeads, dimension))
    ops.attention(
      ops.fromFloats(Shape.of(tokens, qHeads, dimension), queries),
      cache,
      pageTable,
      queryStart,
      keyCount,
      options(ops),
      out
    )
    ops.toFloats(out)
  }

  /** Writes keys and values of positions `from` until `until`. */
  def write(
      ops: Ops,
      cache: KvCache,
      pageTable: Tensor,
      from: Int,
      until: Int
  ): Unit = {
    val width = kvHeads * dimension
    val shape = Shape.of(until - from, kvHeads, dimension)
    ops.cacheWrite(
      ops.fromFloats(shape, keys.slice(from * width, until * width)),
      ops.fromFloats(shape, values.slice(from * width, until * width)),
      cache,
      pageTable,
      from
    )
  }

  def options(ops: Ops): Attention =
    Attention(
      scale = (1 / math.sqrt(dimension)).toFloat,
      causal = causal,
      window = window,
      softcap = softcap,
      sinks = Option.when(sinks)(ops.fromFloats(Shape.of(qHeads), sinkValues))
    )
}
