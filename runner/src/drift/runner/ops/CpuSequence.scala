package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The reference backend's candidates, recurrent states, residual streams and
  * n-gram rows.
  */
private[ops] trait CpuSequence extends CpuBase {

  def argmax(values: Tensor): Int = {
    Ops.requireF32("argmax", values)
    val read = reader(values)
    var best = 0L
    var i = 1L
    while (i < values.shape.elementCount) {
      if (read(i) > read(best)) best = i
      i += 1
    }
    best.toInt
  }

  def candidates(values: Tensor, count: Int): Seq[Candidates] = {
    Ops.requireCandidates(values, count)
    Candidates.of(toFloats(values), values.shape.last.toInt, count)
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
    val span = (taps - 1) * dilation
    val (input, w, target, memory) =
      (reader(x), reader(weight), segment(out), segment(state))
    (0 until channels).foreach { c =>
      // the inputs this channel has seen, oldest first: the state, then x
      val seen = (0 until span).map(i =>
        memory.getAtIndex(JAVA_FLOAT, i.toLong * channels + c).toDouble
      ) ++
        (0 until tokens).map(t => input(t.toLong * channels + c))
      (0 until tokens).foreach { t =>
        // tap i reads (taps − 1 − i) × dilation tokens back
        val sum = (0 until taps)
          .map(i =>
            w(c.toLong * taps + i) * seen(t + span - (taps - 1 - i) * dilation)
          )
          .sum
        target.setAtIndex(
          JAVA_FLOAT,
          t.toLong * channels + c,
          (sum / (1 + math.exp(-sum))).toFloat
        )
      }
      (0 until span).foreach(i =>
        memory.setAtIndex(
          JAVA_FLOAT,
          i.toLong * channels + c,
          seen(tokens + i).toFloat
        )
      )
      history.map(segment).foreach { kept =>
        for {
          t <- 0 until tokens
          i <- 0 until span
        } kept.setAtIndex(
          JAVA_FLOAT,
          (t.toLong * span + i) * channels + c,
          seen(t + 1 + i).toFloat
        )
      }
    }
  }

  /** transformers' `torch_recurrent_gated_delta_rule`, in double. */
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
    val (row, as, bs, decays, biases) =
      (reader(qkv), reader(a), reader(b), reader(decay), reader(dtBias))
    val (memory, target) = (segment(state), segment(out))
    val (d, keys, values, width) =
      (rule.dimension, rule.keyHeads, rule.valueHeads, rule.qkvWidth.toLong)
    (0 until values).foreach { h =>
      val kh = rule.keyHeadOf(h)
      val s = Array.tabulate(d, d)((i, j) =>
        memory.getAtIndex(JAVA_FLOAT, (h.toLong * d + i) * d + j).toDouble
      )
      (0 until tokens).foreach { t =>
        def vector(offset: Long) =
          Array.tabulate(d)(i => row(t * width + offset + i))
        def normalized(v: Array[Double]) = {
          val n = 1 / math.sqrt(v.map(x => x * x).sum + 1e-6); v.map(_ * n)
        }
        val q = normalized(vector(kh.toLong * d)).map(_ / math.sqrt(d))
        val k = normalized(vector(keys.toLong * d + kh.toLong * d))
        val v = vector(2L * keys * d + h.toLong * d)
        val x = as(t.toLong * values + h) + biases(h)
        val g =
          math.exp(decays(h) * (if (x > 20) x else math.log1p(math.exp(x))))
        val beta = 1 / (1 + math.exp(-bs(t.toLong * values + h)))
        for {
          i <- 0 until d
          j <- 0 until d
        } s(i)(j) *= g
        val delta = Array.tabulate(d)(j =>
          (v(j) - (0 until d).map(i => s(i)(j) * k(i)).sum) * beta
        )
        for {
          i <- 0 until d
          j <- 0 until d
        } s(i)(j) += k(i) * delta(j)
        (0 until d).foreach { j =>
          val o = (0 until d).map(i => s(i)(j) * q(i)).sum
          target.setAtIndex(
            JAVA_FLOAT,
            (t.toLong * values + h) * d + j,
            o.toFloat
          )
        }
        history.map(segment).foreach { kept =>
          for {
            i <- 0 until d
            j <- 0 until d
          } kept.setAtIndex(
            JAVA_FLOAT,
            ((t.toLong * values + h) * d + i) * d + j,
            s(i)(j).toFloat
          )
        }
      }
      for {
        i <- 0 until d
        j <- 0 until d
      } memory.setAtIndex(
        JAVA_FLOAT,
        (h.toLong * d + i) * d + j,
        s(i)(j).toFloat
      )
    }
  }

  def streamsMix(
      gates: Tensor,
      normed: Tensor,
      streams: Int,
      out: Tensor
  ): Unit = {
    val (_, found, width) =
      Ops.checkStreams("streamsMix", Seq(gates, normed), out)
    require(found == streams, s"streamsMix: $found streams, expected $streams")
    val (g, n) = (reader(gates), reader(normed))
    fill(out) { index =>
      val (t, c) = (index / width, index % width)
      (0 until streams).map { j =>
        val at = (t * streams + j) * width + c
        n(at) / (1 + math.exp(-g(at)))
      }.sum / streams
    }
  }

  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit = {
    val (_, streams, width) = Ops.checkStreams("streamsCombine", Seq(x), y)
    val (before, added, logit) = (reader(x), reader(y), reader(logits))
    val values = Array.tabulate(x.shape.elementCount.toInt) { index =>
      val tj = index / width
      val weight = 2 / (1 + math.exp(-logit(tj) / streams))
      before(index) + weight * added(tj / streams * width + index % width)
    }
    fill(x)(index => values(index.toInt))
  }

  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit = {
    val (tokens, streams, width) =
      Ops.checkStreams("pleGate", Seq(key, query, out), value)
    val (k, q, v) = (reader(key), reader(query), reader(value))
    val gates = Array.tabulate(tokens * streams) { tj =>
      val s = (0L until width)
        .map(c => k(tj.toLong * width + c) * q(tj.toLong * width + c))
        .sum /
        math.sqrt(width)
      val root = math.sqrt(math.max(math.abs(s), 1e-6))
      1 / (1 + math.exp(-(if (s < 0) -root else root)))
    }
    fill(out) { index =>
      val tj = (index / width).toInt
      gates(tj) * v(tj / streams * width.toLong + index % width)
    }
  }

  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit = {
    val count = Ops.checkNgramRows(tokens, state, hash, rows)
    history.foreach(Ops.checkHistory(_, count, state))
    val (ids, memory, target) = (segment(tokens), segment(state), segment(rows))
    // the stored predecessors, oldest first, then the call's tokens
    val seen = (0 until hash.kept).map { i =>
      val stored = memory.getAtIndex(JAVA_FLOAT, i.toLong).toInt
      if (stored == 0) hash.eos else stored - 1
    } ++ (0 until count).map(t => ids.getAtIndex(JAVA_INT, t.toLong))
    (0 until count).foreach { t =>
      val at = t + hash.kept
      val context = (0 until hash.ngramSize)
        .scanLeft((seen(at), false)) { case ((_, cut), k) =>
          if (k == 0) (seen(at), false)
          else {
            val previous = seen(at - k)
            val nowCut = cut || previous == hash.eos
            (if (nowCut) hash.eos else previous, nowCut)
          }
        }
        .tail
        .map(_._1)
      (0 until hash.heads).foreach { h =>
        val n = 2 + h / hash.headsPerNgram
        val mixed = (0 until n)
          .map(j => context(j).toLong * hash.multipliers(j))
          .reduce(_ ^ _)
        val row = java.lang.Long.remainderUnsigned(mixed, hash.sizes(h)) +
          hash.offsets(h)
        target.setAtIndex(JAVA_INT, t.toLong * hash.heads + h, row.toInt)
      }
      history.map(segment).foreach { kept =>
        (0 until hash.kept).foreach(i =>
          kept.setAtIndex(
            JAVA_FLOAT,
            t.toLong * hash.kept + i,
            (seen(at - (hash.kept - 1 - i)) + 1).toFloat
          )
        )
      }
    }
    (0 until hash.kept).foreach(i =>
      memory.setAtIndex(
        JAVA_FLOAT,
        i.toLong,
        (seen(count + i) + 1).toFloat
      )
    )
  }
}
