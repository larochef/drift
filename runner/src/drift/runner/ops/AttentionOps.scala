package drift.runner.ops

import drift.runner.tensor.*

/** Attention and rotary positions: the key-value cache, the attentions over it
  * and over short sequences, and the rotary embeddings.
  */
trait AttentionOps { this: Ops =>

  /** Attention over many short sequences at once, no mask: `q` and `out`
    * `[S, B, heads, D]`, `k` and `v` `[S, B, kvHeads, D]` (row `s × B + b` is
    * position `s` of sequence `b`), scores `q·k × scale`, plus `bias` (F32
    * `[heads, S, S]`, by query then key position) when given: T5's relative
    * positions.
    */
  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor] = None
  ): Unit

  /** Rotary embedding from precomputed angles, pairs of adjacent values: pair
    * `m` of every head of token `t` (values `2m`, `2m + 1`) turns by the angle
    * whose cosine and sine are `cosines[t, m]` and `sines[t, m]`; the values
    * past the table's pairs are copied. `x` and `out` `[tokens, heads, D]`.
    * With `halves`, pair `m` is values `m` and `m + pairs` (transformers'
    * `rotate_half` over the head's first `2 × pairs` values). Tables of
    * `[tokens, heads, pairs]` give each head its own angles (LTX 2's "split"
    * RoPE).
    */
  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean = false
  ): Unit

  /** Rotary embedding of `x` `[tokens, heads, headDimension]`; `positions` is
    * I32 `[tokens]`, or `[3, tokens]` for mRoPE.
    */
  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit

  /** An empty cache of `pages` pages (`KvCache`). */
  def allocateCache(
      pages: Int,
      pageSize: Int,
      kvHeads: Int,
      headDimension: Int
  ): KvCache

  /** Writes the keys and values of `k`, `v` (F32 `[tokens, kvHeads, D]`) at
    * positions `start` on, through the sequence's `pageTable`.
    */
  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit

  /** A gated attention's inputs from its projections: each query head's half of
    * `queryAndGate` (`[t, heads, 2·D]`: the query, then the output gate into
    * `gate`) and `keys` (`[t, kvHeads, D]`) normed and turned as `norms` says,
    * the queries into `rotatedQueries`, the keys written to `cache` with
    * `values` at `start`. `queries` and `rotatedKeys` hold what lies between,
    * where the steps run apart. One launch where the backend fuses them.
    */
  def attentionInputs(
      queryAndGate: Tensor,
      keys: Tensor,
      values: Tensor,
      norms: QueryKeyNorms,
      positions: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int,
      queries: Tensor,
      gate: Tensor,
      rotatedKeys: Tensor,
      rotatedQueries: Tensor
  ): Unit = {
    val Seq(t, heads, d) = queries.shape.dimensions
    val kvHeads = keys.shape.dimensions(1)
    splitHalves(queryAndGate, queries, gate)
    rmsNorm(
      queries.view(t * heads, d),
      norms.queryNorm,
      norms.epsilon,
      norms.weightOffset,
      queries.view(t * heads, d)
    )
    rmsNorm(
      keys.view(t * kvHeads, d),
      norms.keyNorm,
      norms.epsilon,
      norms.weightOffset,
      keys.view(t * kvHeads, d)
    )
    rope(queries, positions, norms.rope, rotatedQueries)
    rope(keys, positions, norms.rope, rotatedKeys)
    cacheWrite(rotatedKeys, values, cache, pageTable, start)
  }

  /** Attention of `q` (F32 `[tokens, qHeads, D]`, token `t` at position
    * `queryStart + t`) over the sequence's first `keyCount` cached keys,
    * `qHeads / kvHeads` query heads per key head, into `out` (like `q`).
    */
  def attention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): Unit
}

/** The rules of `AttentionOps`.
  */
private[ops] trait AttentionChecks extends CommonChecks {

  /** Returns `(tokens, heads, D, pairs)`. */
  def checkRopeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int, Boolean) = {
    checkElementwise("ropeTable", x, out)
    requireF32("ropeTable", cosines, sines)
    require(
      x.shape.rank == 3,
      s"ropeTable takes [tokens, heads, D], not ${x.shape}"
    )
    val Seq(tokens, heads, d) = x.shape.dimensions.map(_.toInt)
    val pairs = cosines.shape.last.toInt
    val perHead = cosines.shape.rank == 3
    require(
      (cosines.shape == Shape.of(tokens, pairs) ||
        cosines.shape == Shape.of(tokens, heads, pairs)) &&
        sines.shape == cosines.shape && 2 * pairs <= d,
      s"ropeTable: tables ${cosines.shape}, ${sines.shape} for ${x.shape}"
    )
    (tokens, heads, d, pairs, perHead)
  }

  /** Returns `(S, B, heads, kvHeads, D)`. */
  def checkShortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int, Int) = {
    requireF32("shortAttention", q, k, v, out)
    requireSameShape("shortAttention", q, out)
    requireSameShape("shortAttention", k, v)
    require(
      q.shape.rank == 4 && k.shape.rank == 4,
      s"shortAttention takes [S, B, heads, D], not ${q.shape}"
    )
    val Seq(s, b, heads, d) = q.shape.dimensions.map(_.toInt)
    val Seq(ks, kb, kvHeads, kd) = k.shape.dimensions.map(_.toInt)
    require(
      ks == s && kb == b && kd == d && heads % kvHeads == 0 && d <= 512,
      s"shortAttention: q ${q.shape}, k ${k.shape}"
    )
    (s, b, heads, kvHeads, d)
  }

  def checkAttentionBias(bias: Tensor, heads: Int, s: Int): Unit = {
    requireF32("shortAttention bias", bias)
    require(
      bias.shape == Shape.of(heads, s, s),
      s"shortAttention: bias ${bias.shape} for $heads heads over $s positions"
    )
  }

  /** Returns `(tokens, heads, headDimension)`. */
  def checkRope(
      x: Tensor,
      positions: Tensor,
      rope: Rope,
      out: Tensor
  ): (Int, Int, Int) = {
    checkElementwise("rope", x, out)
    require(
      x.shape.rank == 3,
      s"rope takes [tokens, heads, headDimension], got ${x.shape}"
    )
    val Seq(tokens, heads, dimension) = x.shape.dimensions.map(_.toInt)
    require(
      positions.dtype == DType.I32,
      s"rope positions are I32, not ${positions.dtype}"
    )
    rope.sections match {
      case RopeSections.Axes(pairs) =>
        require(
          pairs.size <= 4 && pairs.forall(_ > 0) &&
            pairs.sum * 2 == rope.rotaryDimensions,
          s"rope: axes of $pairs pairs for ${rope.rotaryDimensions} rotary values"
        )
      case _ => ()
    }
    val expected = rope.sections.positionsPerToken match {
      case 1     => Shape.of(tokens)
      case count => Shape.of(count, tokens)
    }
    require(
      positions.shape == expected,
      s"rope positions ${positions.shape}, expected $expected"
    )
    require(
      rope.rotaryDimensions % 2 == 0 && rope.rotaryDimensions <= dimension,
      s"rope: ${rope.rotaryDimensions} rotary dimensions in heads of $dimension"
    )
    (tokens, heads, dimension)
  }

  def checkCacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Int = {
    checkElementwise("cacheWrite", k, v)
    require(
      k.shape.rank == 3 && k.shape.dimensions
        .drop(1) == Vector(cache.kvHeads.toLong, cache.headDimension.toLong),
      s"cacheWrite: ${k.shape} into a cache of ${cache.kvHeads} heads of ${cache.headDimension}"
    )
    val tokens = k.shape.dimensions.head.toInt
    checkPageTable(pageTable, cache, start + tokens)
    tokens
  }

  def checkPageTable(
      pageTable: Tensor,
      cache: KvCache,
      positions: Int
  ): Unit = {
    require(
      pageTable.dtype == DType.I32 && pageTable.shape.rank == 1,
      s"a page table is I32 [pages], not ${pageTable.dtype} ${pageTable.shape}"
    )
    require(
      pageTable.shape.elementCount * cache.pageSize >= positions,
      s"a page table of ${pageTable.shape.elementCount} pages for $positions positions"
    )
  }

  /** Returns `(tokens, qHeads)`. */
  def checkAttention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): (Int, Int) = {
    checkElementwise("attention", q, out)
    require(
      q.shape.rank == 3 && q.shape.last == cache.headDimension,
      s"attention: queries ${q.shape} against heads of ${cache.headDimension}"
    )
    val Seq(tokens, qHeads, _) = q.shape.dimensions.map(_.toInt)
    require(
      qHeads % cache.kvHeads == 0,
      s"attention: $qHeads query heads over ${cache.kvHeads} key heads"
    )
    // queries see keys by position only when causal or windowed; otherwise
    // they may lie past the keys (padding that attends to a prompt alone)
    require(
      queryStart >= 0 && (queryStart + tokens <= keyCount ||
        !attention.causal && attention.window.isEmpty),
      s"attention: queries at $queryStart..${queryStart + tokens - 1} past $keyCount keys"
    )
    checkPageTable(pageTable, cache, keyCount)
    attention.sinks.foreach { sinks =>
      requireF32("attention sinks", sinks)
      require(
        sinks.shape == Shape.of(qHeads),
        s"attention: sinks ${sinks.shape} for $qHeads heads"
      )
    }
    (tokens, qHeads)
  }
}
