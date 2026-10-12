package drift.runner.ops

import drift.runner.native.KernelArgument.{I32, Pointer}
import drift.runner.tensor.*

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.collection.mutable

/** The GPU backend's candidates, recurrent states, residual streams and n-gram
  * rows.
  */
private[ops] trait HipSequence extends HipBase {

  def argmax(values: Tensor): Int = {
    Ops.requireF32("argmax", values)
    launch(
      kernel(normKernels, "argmax_f32"),
      1,
      1024,
      Pointer(pointer(values)),
      Pointer(pointer(argmaxIndex)),
      I32(values.shape.elementCount.toInt)
    )
    hip.copy(intStaging, pointer(argmaxIndex), 4)
    intStaging.get(JAVA_INT, 0)
  }

  /** `candidates`' ids and values, `[rows, count]` pairs of ints. */
  private var candidateBuffer = Option.empty[Tensor]

  /** What each chunk of `candidates`' rows keeps, and the chunks each row
    * finished (the last leaves it at 0): `norm.hip`'s `candidates_f32`.
    */
  private var candidateKept = Option.empty[Tensor]
  private lazy val finishedCandidates = {
    val counters = allocate(DType.I32, Shape.of(65536))
    zero(counters)
    counters
  }
  private val CandidateChunk = 4096

  def candidates(values: Tensor, count: Int): Seq[Candidates] = {
    Ops.requireCandidates(values, count)
    val width = values.shape.last.toInt
    val rows = (values.shape.elementCount / width).toInt
    val kept = math.min(count, width)
    val pairs = 2L * rows * count
    require(pairs <= 65536, s"$rows rows of $count candidates")
    val buffer = candidateBuffer
      .filter(_.shape.elementCount >= pairs)
      .getOrElse {
        candidateBuffer.foreach(release)
        val fresh = allocate(DType.I32, Shape.of(math.max(pairs, 2048L)))
        candidateBuffer = Some(fresh)
        fresh
      }
    // each row's chunks keep their own candidates, then the last one to finish
    // selects among them
    val chunks = (width + CandidateChunk - 1) / CandidateChunk
    val keptValues = 2L * rows * chunks * count
    val chunkCandidates = candidateKept
      .filter(_.shape.elementCount >= keptValues)
      .getOrElse {
        candidateKept.foreach(release)
        val fresh = allocate(DType.I32, Shape.of(keptValues))
        candidateKept = Some(fresh)
        fresh
      }
    launch(
      kernel(normKernels, "candidates_f32"),
      rows.toLong * chunks,
      1024,
      Pointer(pointer(values)),
      Pointer(pointer(buffer)),
      I32(width),
      I32(count),
      Pointer(pointer(chunkCandidates)),
      Pointer(pointer(finishedCandidates))
    )
    hip.copy(intStaging, pointer(buffer), 4 * pairs)
    val all = intStaging.asSlice(0, 4 * pairs).toArray(JAVA_INT)
    (0 until rows).map { row =>
      val at = 2 * row * count
      Candidates(
        Array.tabulate(kept)(c => all(at + 2 * c + 1)),
        Array.tabulate(kept)(c =>
          java.lang.Float.intBitsToFloat(all(at + 2 * c))
        )
      )
    }
  }

  def causalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val (tokens, channels, taps) =
      Ops.checkCausalConv(x, weight, dilation, state, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    launch(
      kernel(linearAttentionKernels, "causal_conv_silu"),
      (channels + 255L) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(weight)),
      Pointer(pointer(state)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(channels),
      I32(taps),
      I32(dilation),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
  }

  def gatedDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val tokens = Ops.checkDeltaRule(qkv, a, b, decay, dtBias, state, rule, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    require(
      Set(32, 64, 128).contains(rule.dimension),
      s"no delta-rule kernel for heads of ${rule.dimension}"
    )
    launch(
      kernel(linearAttentionKernels, s"delta_rule_d${rule.dimension}"),
      rule.valueHeads.toLong * rule.dimension / 32, // COLUMNS per workgroup
      128,
      Pointer(pointer(qkv)),
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(decay)),
      Pointer(pointer(dtBias)),
      Pointer(pointer(state)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(rule.keyHeads),
      I32(rule.valueHeads),
      I32(if (rule.tiledHeads) 1 else 0),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
  }

  override def convolvedDeltaRule(
      qkv: Tensor,
      convWeight: Tensor,
      convState: Tensor,
      convHistory: Option[Tensor],
      convolved: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val (tokens, _, taps) =
      Ops.checkCausalConv(qkv, convWeight, 1, convState, convolved)
    if (taps != 4 || rule.dimension != 128 || rule.keyHeads > 128)
      super.convolvedDeltaRule(
        qkv,
        convWeight,
        convState,
        convHistory,
        convolved,
        a,
        b,
        decay,
        dtBias,
        state,
        rule,
        history,
        out
      )
    else {
      Ops.checkDeltaRule(qkv, a, b, decay, dtBias, state, rule, out)
      convHistory.foreach(Ops.checkHistory(_, tokens, convState))
      history.foreach(Ops.checkHistory(_, tokens, state))
      launch(
        kernel(linearAttentionKernels, s"conv_delta_rule_d${rule.dimension}"),
        rule.valueHeads.toLong * rule.dimension / 32, // COLUMNS per workgroup
        128,
        Pointer(pointer(qkv)),
        Pointer(pointer(convWeight)),
        Pointer(pointer(convState)),
        Pointer(convHistory.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(a)),
        Pointer(pointer(b)),
        Pointer(pointer(decay)),
        Pointer(pointer(dtBias)),
        Pointer(pointer(state)),
        Pointer(pointer(out)),
        I32(tokens),
        I32(rule.keyHeads),
        I32(rule.valueHeads),
        I32(if (rule.tiledHeads) 1 else 0),
        Pointer(history.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(deltaCounters))
      )
    }
  }

  /** `conv_delta_rule`'s counters: zero between launches. */
  private lazy val deltaCounters = {
    val counters = allocate(DType.I32, Shape.of(256))
    zero(counters)
    counters
  }

  def streamsMix(
      gates: Tensor,
      normed: Tensor,
      streams: Int,
      out: Tensor
  ): Unit = {
    val (tokens, found, width) =
      Ops.checkStreams("streamsMix", Seq(gates, normed), out)
    require(found == streams, s"streamsMix: $found streams, expected $streams")
    launch(
      kernel(streamsKernels, "streams_mix"),
      (tokens.toLong * width + 255) / 256,
      256,
      Pointer(pointer(gates)),
      Pointer(pointer(normed)),
      Pointer(pointer(out)),
      I32(tokens),
      I32(streams),
      I32(width)
    )
  }

  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit = {
    val (tokens, streams, width) = Ops.checkStreams("streamsCombine", Seq(x), y)
    Ops.requireF32("streamsCombine", logits)
    require(
      logits.shape == Shape.of(tokens, streams),
      s"streamsCombine: logits ${logits.shape}"
    )
    launch(
      kernel(streamsKernels, "streams_combine"),
      (tokens.toLong * streams * width + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(logits)),
      I32(tokens),
      I32(streams),
      I32(width)
    )
  }

  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit = {
    val (tokens, streams, width) =
      Ops.checkStreams("pleGate", Seq(key, query, out), value)
    launch(
      kernel(streamsKernels, "ple_gate"),
      tokens.toLong * streams,
      256,
      Pointer(pointer(key)),
      Pointer(pointer(query)),
      Pointer(pointer(value)),
      Pointer(pointer(out)),
      I32(streams),
      I32(width)
    )
  }

  /** Each hash's multipliers, sizes and offsets on the device, as 64-bit words
    * (two ints each), uploaded once.
    */
  private val hashTables = mutable.Map.empty[NgramHash, Seq[Tensor]]

  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit = {
    val count = Ops.checkNgramRows(tokens, state, hash, rows)
    history.foreach(Ops.checkHistory(_, count, state))
    val Seq(multipliers, sizes, offsets) = hashTables.getOrElseUpdate(
      hash,
      Seq(hash.multipliers, hash.sizes, hash.offsets).map(longs =>
        fromInts(
          Shape.of(2L * longs.size),
          longs.flatMap(l => Seq(l.toInt, (l >>> 32).toInt)).toArray
        )
      )
    )
    launch(
      kernel(streamsKernels, "ngram_rows"),
      (count.toLong * hash.heads + 255) / 256,
      256,
      Pointer(pointer(tokens)),
      Pointer(pointer(state)),
      Pointer(pointer(multipliers)),
      Pointer(pointer(sizes)),
      Pointer(pointer(offsets)),
      Pointer(pointer(rows)),
      I32(count),
      I32(hash.ngramSize),
      I32(hash.headsPerNgram),
      I32(hash.eos),
      Pointer(history.fold(MemorySegment.NULL)(pointer))
    )
    launch(
      kernel(streamsKernels, "ngram_state"),
      1,
      32,
      Pointer(pointer(tokens)),
      Pointer(pointer(state)),
      I32(count),
      I32(hash.kept)
    )
  }
}
