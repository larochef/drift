package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*

/** The reference backend's attention, key-value cache and rotary positions.
  */
private[ops] trait CpuAttention extends CpuBase {

  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor]
  ): Unit = {
    val (s, b, heads, kvHeads, d) = Ops.checkShortAttention(q, k, v, out)
    bias.foreach(t => Ops.checkAttentionBias(t, heads, s))
    val biases = bias.map(reader)
    val (queries, keys, values, target) =
      (reader(q), reader(k), reader(v), segment(out))
    for {
      position <- 0 until s
      sequence <- 0 until b
      h <- 0 until heads
    } {
      val row = position.toLong * b + sequence
      val kv = h / (heads / kvHeads)
      def keyAt(other: Int) = (other.toLong * b + sequence) * kvHeads + kv
      val scores = (0 until s).map { other =>
        (0 until d)
          .map(i =>
            queries((row * heads + h) * d + i) * keys(keyAt(other) * d + i)
          )
          .sum * scale + biases.fold(0.0)(table =>
          table((h.toLong * s + position) * s + other)
        )
      }
      val largest = scores.max
      val weights = scores.map(score => math.exp(score - largest))
      (0 until d).foreach { i =>
        val value = (0 until s)
          .map(other => weights(other) * values(keyAt(other) * d + i))
          .sum / weights.sum
        target.setAtIndex(JAVA_FLOAT, (row * heads + h) * d + i, value.toFloat)
      }
    }
  }

  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean
  ): Unit = {
    val (_, heads, d, pairs, perHead) =
      Ops.checkRopeTable(x, cosines, sines, out)
    val (input, cosAt, sinAt) = (reader(x), reader(cosines), reader(sines))
    fill(out) { index =>
      val dim = (index % d).toInt
      // the table's row: the token's, or the token's head's
      val token =
        if (perHead) index / d else index / (heads.toLong * d)
      val (cos, sin) = (cosAt, sinAt)
      if (halves) {
        if (dim >= 2 * pairs) input(index)
        else {
          val m = dim % pairs
          val (c, s) = (cos(token * pairs + m), sin(token * pairs + m))
          val (a, b) = (input(index - dim + m), input(index - dim + m + pairs))
          if (dim < pairs) a * c - b * s else b * c + a * s
        }
      } else {
        val m = dim / 2
        if (m >= pairs) input(index)
        else {
          val (c, s) = (cos(token * pairs + m), sin(token * pairs + m))
          val (a, b) = (input(index - dim % 2), input(index - dim % 2 + 1))
          if (dim % 2 == 0) a * c - b * s else a * s + b * c
        }
      }
    }
  }

  /** The angle is computed in float, as transformers does (`inv_freq` in
    * float32, then `position × inv_freq`), and turned in double.
    */
  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit = {
    val (tokens, heads, dimension) = Ops.checkRope(x, positions, rope, out)
    val (input, target, position) =
      (reader(x), segment(out), segment(positions))
    val rotary = rope.rotaryDimensions
    val half = rotary / 2
    def section(pair: Int): Int = rope.sections match {
      case RopeSections.Single                          => 0
      case RopeSections.Contiguous(temporal, height, _) =>
        if (pair < temporal) 0 else if (pair < temporal + height) 1 else 2
      case RopeSections.Axes(pairs) =>
        pairs.scanLeft(0)(_ + _).tail.indexWhere(pair < _)
      case RopeSections.Interleaved(_, height, width) =>
        if (pair % 3 == 1 && pair < 3 * height) 1
        else if (pair % 3 == 2 && pair < 3 * width) 2
        else 0
    }
    for {
      t <- 0 until tokens
      h <- 0 until heads
    } {
      val base = (t.toLong * heads + h) * dimension
      for (d <- 0 until dimension) {
        val value =
          if (d >= rotary) input(base + d)
          else {
            val (pair, first, partner) = rope.layout match {
              case RopeLayout.Neox =>
                (d % half, d < half, if (d < half) d + half else d - half)
              case RopeLayout.Interleaved =>
                (d / 2, d % 2 == 0, if (d % 2 == 0) d + 1 else d - 1)
            }
            val inverseFrequency = rope.sections match {
              case RopeSections.Axes(pairs) =>
                val axis = section(pair)
                val (start, count) = (pairs.take(axis).sum, pairs(axis))
                1f / math
                  .pow(rope.theta, (pair - start).toDouble / count)
                  .toFloat
              case _ =>
                1f / math.pow(rope.theta, 2.0 * pair / rotary).toFloat
            }
            val p =
              position.getAtIndex(JAVA_INT, section(pair).toLong * tokens + t)
            val angle = (p.toFloat * inverseFrequency).toDouble
            val (a, b) =
              if (first) (input(base + d), input(base + partner))
              else (input(base + partner), input(base + d))
            if (first) a * math.cos(angle) - b * math.sin(angle)
            else a * math.sin(angle) + b * math.cos(angle)
          }
        target.setAtIndex(JAVA_FLOAT, base + d, value.toFloat)
      }
    }
  }

  def allocateCache(
      pages: Int,
      pageSize: Int,
      kvHeads: Int,
      headDimension: Int
  ): KvCache =
    KvCache(
      allocate(DType.F16, Shape.of(pages, pageSize, kvHeads, headDimension)),
      allocate(DType.F16, Shape.of(pages, kvHeads, headDimension, pageSize)),
      pageSize
    )

  /** Where position `position`'s key for `head`, dimension `d`, lies in the key
    * pool, and its value in the value pool, in elements.
    */
  private def cacheIndices(
      cache: KvCache,
      table: MemorySegment,
      position: Int,
      head: Int,
      d: Int
  ): (Long, Long) = {
    val page =
      table.getAtIndex(JAVA_INT, (position / cache.pageSize).toLong).toLong
    val offset = position % cache.pageSize
    val (heads, dimension, size) =
      (cache.kvHeads, cache.headDimension, cache.pageSize)
    (
      ((page * size + offset) * heads + head) * dimension + d,
      ((page * heads + head) * dimension + d) * size + offset
    )
  }

  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit = {
    val tokens = Ops.checkCacheWrite(k, v, cache, pageTable, start)
    val (keys, values, table) =
      (segment(cache.keys), segment(cache.values), segment(pageTable))
    val (key, value) = (reader(k), reader(v))
    for {
      t <- 0 until tokens
      h <- 0 until cache.kvHeads
      d <- 0 until cache.headDimension
    } {
      val source = (t.toLong * cache.kvHeads + h) * cache.headDimension + d
      val (keyAt, valueAt) = cacheIndices(cache, table, start + t, h, d)
      keys.setAtIndex(
        JAVA_SHORT,
        keyAt,
        java.lang.Float.floatToFloat16(key(source).toFloat)
      )
      values.setAtIndex(
        JAVA_SHORT,
        valueAt,
        java.lang.Float.floatToFloat16(value(source).toFloat)
      )
    }
  }

  /** Scores, softmax and weighted values in double, one query row at a time,
    * over the F16 keys and values as stored.
    */
  def attention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): Unit = {
    val (tokens, qHeads) = Ops.checkAttention(
      q,
      cache,
      pageTable,
      queryStart,
      keyCount,
      attention,
      out
    )
    val (keys, values, table, target) = (
      segment(cache.keys),
      segment(cache.values),
      segment(pageTable),
      segment(out)
    )
    val query = reader(q)
    val sinks = attention.sinks.map(reader)
    val dimension = cache.headDimension
    val group = qHeads / cache.kvHeads
    def half(pool: MemorySegment, index: Long) =
      java.lang.Float
        .float16ToFloat(pool.getAtIndex(JAVA_SHORT, index))
        .toDouble
    for {
      t <- 0 until tokens
      head <- 0 until qHeads
    } {
      val kvHead = head / group
      val position = queryStart + t
      val queryAt = (t.toLong * qHeads + head) * dimension
      val visible = (0 until keyCount).filter { key =>
        (!attention.causal || key <= position) && attention.window.forall(w =>
          key > position - w
        )
      }
      val scores = visible.map { key =>
        val dot = (0 until dimension).map { d =>
          query(queryAt + d) * half(
            keys,
            cacheIndices(cache, table, key, kvHead, d)._1
          )
        }.sum
        val score = dot * attention.scale
        attention.softcap.fold(score)(cap => cap * math.tanh(score / cap))
      }
      val sink = sinks.map(_(head.toLong))
      val largest = (scores ++ sink).maxOption.getOrElse(0.0)
      val weights = scores.map(s => math.exp(s - largest))
      val total = weights.sum + sink.fold(0.0)(s => math.exp(s - largest))
      (0 until dimension).foreach { d =>
        val sum = visible.indices.map { i =>
          weights(i) * half(
            values,
            cacheIndices(cache, table, visible(i), kvHead, d)._2
          )
        }.sum
        target.setAtIndex(
          JAVA_FLOAT,
          queryAt + d,
          (if (total > 0) sum / total else 0.0).toFloat
        )
      }
    }
  }
}
