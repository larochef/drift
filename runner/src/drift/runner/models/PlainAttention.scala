package drift.runner.models

import drift.runner.ops.{Attention, Ops}
import drift.runner.state.KvCache
import drift.runner.tensor.*

/** Attention of whole sequences without a causal order, for the models that run
  * every token at each pass: queries of `heads` over keys and values of
  * `kvHeads`, heads of `headDimension`, scores scaled by `1 / √headDimension`.
  * The keys and values go through one page of an F16 cache, remade when a
  * longer sequence comes; values beyond the cache's range are scaled down by a
  * power of two before it and the output back (attention is linear in them).
  */
final class PlainAttention(
    ops: Ops,
    heads: Int,
    kvHeads: Int,
    headDimension: Int,
    /** The head width the scores are scaled for, when the heads are padded to a
      * wider kernel's.
      */
    scaledFor: Option[Int] = None
) extends AutoCloseable {

  private val ValueLimit = 8192f
  private val attention = Attention(
    (1 / math.sqrt(scaledFor.getOrElse(headDimension).toDouble)).toFloat,
    causal = false,
    None,
    None,
    None
  )
  private val pageTable = ops.fromInts(Shape.of(1), Array(0))
  private var cache = Option.empty[KvCache]

  private def cacheFor(tokens: Int): KvCache =
    cache.filter(_.pageSize >= tokens).getOrElse {
      release()
      val created =
        ops.allocateCache(1, (tokens + 15) / 16 * 16, kvHeads, headDimension)
      cache = Some(created)
      created
    }

  private def release(): Unit = cache.foreach { held =>
    ops.release(held.keys)
    ops.release(held.values)
  }

  /** `out` (`[queries, heads × headDimension]`) from `q` (the same), `k` and
    * `v` (`[keys, kvHeads × headDimension]`; `v` is left scaled when it
    * outgrows the cache). Every query sees every key, or with `sight` the
    * queries `[first, first + count)` see the first `keyCount` keys, range
    * after range.
    */
  def apply(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      out: Tensor,
      sight: Seq[(Int, Int, Int)] = Nil
  ): Unit = {
    val (queries, keys) =
      (q.shape.dimensions.head.toInt, k.shape.dimensions.head.toInt)
    val d = headDimension.toLong
    val held = cacheFor(keys)
    val largest = ops.maxAbs(v)
    val factor =
      if (largest <= ValueLimit) 1f
      else
        math
          .pow(2, math.ceil(math.log(largest / ValueLimit) / math.log(2)))
          .toFloat
    if (factor != 1f) ops.scale(v, 1 / factor, v)
    ops.cacheWrite(
      k.view(keys, kvHeads, d),
      v.view(keys, kvHeads, d),
      held,
      pageTable,
      0
    )
    val ranges = if (sight.isEmpty) Seq((0, queries, keys)) else sight
    ranges.foreach { (first, count, keyCount) =>
      ops.attention(
        q.rows(first, count).view(count, heads, d),
        held,
        pageTable,
        0,
        keyCount,
        attention,
        out.rows(first, count).view(count, heads, d)
      )
    }
    if (factor != 1f) ops.scale(out, factor, out)
  }

  def close(): Unit = {
    release()
    ops.release(pageTable)
  }
}
