package drift.runner.ops

import drift.runner.native.*
import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*

/** The GPU backend's attention, key-value cache and rotary positions.
  */
private[ops] trait HipAttention extends HipBase {

  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor]
  ): Unit = {
    val (s, b, heads, kvHeads, d) = Ops.checkShortAttention(q, k, v, out)
    bias.foreach(b => Ops.checkAttentionBias(b, heads, s))
    val waves = s.toLong * b * heads
    launch(
      kernel(attentionKernels, "short_attention_f32"),
      (waves + 7) / 8,
      256,
      Pointer(pointer(q)),
      Pointer(pointer(k)),
      Pointer(pointer(v)),
      Pointer(pointer(out)),
      I32(s),
      I32(b),
      I32(heads),
      I32(kvHeads),
      I32(d),
      F32(scale),
      Pointer(bias.fold(MemorySegment.NULL)(pointer))
    )
  }

  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean
  ): Unit = {
    val (tokens, heads, d, pairs, perHead) =
      Ops.checkRopeTable(x, cosines, sines, out)
    launch(
      kernel(
        ropeKernels,
        if (halves) "rope_table_halves_f32" else "rope_table_f32"
      ),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(cosines)),
      Pointer(pointer(sines)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(heads),
      I32(d),
      I32(pairs),
      I32(if (perHead) 1 else 0)
    )
  }

  /** `rope_f32`'s four section sizes. */
  private def ropeSections(rope: Rope): Seq[Int] = rope.sections match {
    case RopeSections.Single               => Seq(0, 0, 0, 0)
    case RopeSections.Contiguous(t, h, w)  => Seq(t, h, w, 0)
    case RopeSections.Interleaved(t, h, w) => Seq(t, h, w, 0)
    case RopeSections.Axes(pairs)          => pairs.padTo(4, 0)
  }

  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit = {
    val (tokens, heads, dimension) = Ops.checkRope(x, positions, rope, out)
    val Seq(s0, s1, s2, s3) = ropeSections(rope)
    val count = x.shape.elementCount
    launch(
      kernel(ropeKernels, "rope_f32"),
      (count + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(positions)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(heads),
      I32(dimension),
      I32(rope.rotaryDimensions),
      F32(rope.theta),
      I32(rope.layout.code),
      I32(rope.sections.code),
      I32(s0),
      I32(s1),
      I32(s2),
      I32(s3)
    )
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

  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit = {
    val tokens = Ops.checkCacheWrite(k, v, cache, pageTable, start)
    val count = k.shape.elementCount
    launch(
      kernel(attentionKernels, "cache_write"),
      math.min((count + 255) / 256, 1L << 20),
      256,
      Pointer(pointer(k)),
      Pointer(pointer(v)),
      Pointer(pointer(cache.keys)),
      Pointer(pointer(cache.values)),
      Pointer(pointer(pageTable)),
      I32(tokens),
      I32(cache.kvHeads),
      I32(cache.headDimension),
      I32(cache.pageSize),
      I32(start)
    )
  }

  override def attentionInputs(
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
    if (d > AttentionInputDimension)
      super.attentionInputs(
        queryAndGate,
        keys,
        values,
        norms,
        positions,
        cache,
        pageTable,
        start,
        queries,
        gate,
        rotatedKeys,
        rotatedQueries
      )
    else {
      Ops.checkSplitHalves(queryAndGate, queries, gate)
      Ops.checkRope(queries, positions, norms.rope, rotatedQueries)
      Ops.checkRope(keys, positions, norms.rope, rotatedKeys)
      Ops.checkCacheWrite(rotatedKeys, values, cache, pageTable, start)
      val rope = norms.rope
      val Seq(s0, s1, s2, s3) = ropeSections(rope)
      launch(
        kernel(attentionKernels, "attention_inputs"),
        t * (heads + kvHeads),
        256,
        Pointer(pointer(queryAndGate)),
        Pointer(pointer(keys)),
        Pointer(pointer(values)),
        Pointer(pointer(norms.queryNorm)),
        Pointer(pointer(norms.keyNorm)),
        F32(norms.epsilon),
        F32(norms.weightOffset),
        Pointer(pointer(positions)),
        I32(t.toInt),
        I32(heads.toInt),
        I32(kvHeads.toInt),
        I32(d.toInt),
        I32(rope.rotaryDimensions),
        F32(rope.theta),
        I32(rope.layout.code),
        I32(rope.sections.code),
        I32(s0),
        I32(s1),
        I32(s2),
        I32(s3),
        Pointer(pointer(rotatedQueries)),
        Pointer(pointer(gate)),
        Pointer(pointer(cache.keys)),
        Pointer(pointer(cache.values)),
        Pointer(pointer(pageTable)),
        I32(cache.pageSize),
        I32(start)
      )
    }
  }

  /** `attention.hip`'s ATTENTION_INPUT_DIMENSION. */
  private val AttentionInputDimension = 512

  /** The kernel's `AttentionArguments`, field for field. */
  private val attentionLayout = MemoryLayout.structLayout(
    (Seq(
      "q",
      "keys",
      "values",
      "page_table",
      "sinks",
      "out",
      "partial_o",
      "partial_ml"
    )
      .map(ADDRESS.withName(_)) ++
      Seq(
        "tokens",
        "q_heads",
        "kv_heads",
        "page_size",
        "query_start",
        "key_count"
      )
        .map(JAVA_INT.withName(_)) ++
      Seq(
        JAVA_FLOAT.withName("scale"),
        JAVA_INT.withName("causal"),
        JAVA_INT.withName("window"),
        JAVA_FLOAT.withName("softcap"),
        JAVA_INT.withName("splits"),
        JAVA_INT.withName("token_tiles"),
        JAVA_INT.withName("first_group")
      ))*
  )

  /** Each field's byte offset, resolved once: resolving a layout path per call
    * costs more than the launch.
    */
  private val attentionOffsets: Map[String, Long] =
    attentionLayout
      .memberLayouts()
      .asScala
      .flatMap(member =>
        member
          .name()
          .map[(String, Long)](name =>
            name -> attentionLayout
              .byteOffset(MemoryLayout.PathElement.groupElement(name))
          )
          .toScala
      )
      .toMap

  /** With `token_tiles`, how many splits a token's keys go into at most, and
    * the fewest blocks of 16 keys a split takes (`TOKEN_SPLITS`,
    * `MIN_SPLIT_BLOCKS`).
    */
  private val TokenSplits = 32
  private val MinimumSplitBlocks = 2

  /** Enough workgroups to fill the GPU: a decode step has few query rows, so
    * its keys are split (flash decoding), each split at least 16 blocks of 16
    * (fewer made the per-workgroup setup dominate at 4096 keys).
    */
  private def splitsFor(workgroups: Long, keyCount: Int): Int = {
    val target = 160L
    if (workgroups >= target) 1
    else
      math.max(
        1,
        math
          .min((target + workgroups - 1) / workgroups, (keyCount / 256).toLong)
          .toInt
      )
  }

  /** Query rows (tokens × the group's heads) from which the tiled kernel runs,
    * and its waves per workgroup (`attention.hip`'s `TILED_WAVES`).
    */
  private val TiledMinimumRows = 256
  private val TiledWaves = 8

  /** The most one launch of the tiled kernel is given, in query rows × keys ×
    * key heads: about 50 ms on the Radeon 8060S. PiD's attention over a 4096²
    * tile was one launch of 2.1 s, 56 times a tile, and the desktop's frames
    * waited behind each; cut in launches this size the GPU is free for them in
    * between, and the tile takes the same time.
    */
  private val TiledLaunchWork = 1.5e9

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
    val dimension = cache.headDimension
    require(
      Attention.KernelHeadWidths.contains(dimension),
      s"no attention kernel for heads of $dimension yet"
    )
    val group = qHeads / cache.kvHeads
    val rows = tokens.toLong * group
    // a decode step or a verification: a tile per token and splits of fixed
    // runs of keys (`token_tiles`), so each token attends as it would alone
    val tokenTiles = tokens <= MaxMatVecRows
    val tiles =
      if (tokenTiles) tokens.toLong * ((group + 15) / 16) else (rows + 15) / 16
    require(
      tiles <= Int.MaxValue && rows <= Int.MaxValue,
      s"attention over $rows rows"
    )
    // many rows: the tiled kernel, its waves sharing the keys and values
    val tiled = Option.when(
      !tokenTiles && rows >= TiledMinimumRows && dimension <= 128
    )(
      (s"attention_tiled_d$dimension", TiledWaves)
    )
    val splits =
      if (tiled.isDefined) 1
      else if (tokenTiles) {
        // as many as the tile with the most can use: each cuts its blocks
        // into up to TokenSplits runs of at least MinimumSplitBlocks
        val keys =
          attention.window.fold(keyCount)(window =>
            math.min(keyCount, window + 16)
          )
        val blocks = (keys + 15) / 16
        math.min(
          TokenSplits,
          (blocks + MinimumSplitBlocks - 1) / MinimumSplitBlocks
        )
      } else splitsFor(tiles * cache.kvHeads, keyCount)
    val partials = 4L * cache.kvHeads * rows * splits
    val partialO =
      if (splits > 1) scratch(0, partials * (dimension + 2))
      else MemorySegment.NULL
    val partialMl =
      if (splits > 1) offset(partialO, partials * dimension)
      else MemorySegment.NULL
    val arena = Arena.ofConfined()
    try {
      val arguments = arena.allocate(attentionLayout)
      def at(name: String) = attentionOffsets(name)
      def address(name: String, value: MemorySegment) =
        arguments.set(ADDRESS, at(name), value)
      def int(name: String, value: Int) =
        arguments.set(JAVA_INT, at(name), value)
      def float(name: String, value: Float) =
        arguments.set(JAVA_FLOAT, at(name), value)
      address("q", pointer(q))
      address("keys", pointer(cache.keys))
      address("values", pointer(cache.values))
      address("page_table", pointer(pageTable))
      address("sinks", attention.sinks.fold(MemorySegment.NULL)(pointer))
      address("out", pointer(out))
      address("partial_o", partialO)
      address("partial_ml", partialMl)
      int("tokens", tokens)
      int("q_heads", qHeads)
      int("kv_heads", cache.kvHeads)
      int("page_size", cache.pageSize)
      int("query_start", queryStart)
      int("key_count", keyCount)
      float("scale", attention.scale)
      int("causal", if (attention.causal) 1 else 0)
      int("window", attention.window.getOrElse(0))
      float("softcap", attention.softcap.getOrElse(0f))
      int("splits", splits)
      int("token_tiles", if (tokenTiles) 1 else 0)
      val (name, waves) = tiled.getOrElse((s"attention_d$dimension", 1))
      val groups = ((tiles + waves - 1) / waves).toInt
      // the tiled kernel's rows in launches short enough for the desktop
      val launches =
        if (tiled.isEmpty) 1
        else
          math
            .ceil(rows.toDouble * keyCount * cache.kvHeads / TiledLaunchWork)
            .toInt
            .max(1)
            .min(groups)
      val each = (groups + launches - 1) / launches
      (0 until groups by each).foreach { first =>
        int("first_group", first)
        hip.launch(
          kernel(attentionKernels, name),
          Dim3(math.min(each, groups - first), cache.kvHeads, splits),
          Dim3(32 * waves),
          0,
          MemorySegment.NULL,
          KernelArgument.Struct(arguments)
        )
      }
      if (splits > 1)
        hip.launch(
          kernel(attentionKernels, "attention_combine"),
          Dim3(rows.toInt, cache.kvHeads, 1),
          Dim3(dimension),
          0,
          MemorySegment.NULL,
          KernelArgument.Struct(arguments),
          I32(dimension)
        )
    } finally arena.close()
  }
}
