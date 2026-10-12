package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The reference backend's mixtures of experts.
  */
private[ops] trait CpuExperts extends CpuBase {

  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit = {
    val (tokens, experts, k) = Ops.checkRoute(logits, ids, weights)
    val (scores, chosen, weight) =
      (reader(logits), segment(ids), segment(weights))
    (0 until tokens).foreach { t =>
      val top = (0 until experts)
        .sortBy(e => (-scores(t.toLong * experts + e), e))
        .take(k)
      val largest = scores(t.toLong * experts + top.head)
      val exponentials =
        top.map(e => math.exp(scores(t.toLong * experts + e) - largest))
      top.zipWithIndex.foreach { (e, j) =>
        chosen.setAtIndex(JAVA_INT, t.toLong * k + j, e)
        weight.setAtIndex(
          JAVA_FLOAT,
          t.toLong * k + j,
          (exponentials(j) / exponentials.sum).toFloat
        )
      }
    }
  }

  def expertsLinear(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, weight, ids, out)
    val (input, weights, chosen, target) =
      (segment(x), segment(weight), segment(ids), segment(out))
    val rowBytes = weight.dtype.byteSize(k.toLong)
    (0 until slots).foreach { s =>
      val expert = chosen.getAtIndex(JAVA_INT, s.toLong)
      (0 until n).foreach { row =>
        val decoded = weight.dtype.decode(
          weights.asSlice((expert.toLong * n + row) * rowBytes, rowBytes),
          k
        )
        val sum = (0 until k)
          .map(c =>
            decoded(c).toDouble * input
              .getAtIndex(JAVA_FLOAT, (s / divisor).toLong * k + c)
          )
          .sum
        target.setAtIndex(JAVA_FLOAT, s.toLong * n + row, sum.toFloat)
      }
    }
  }

  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("expertsGatedLinear", gate, up)
    val (gated, raised) =
      (allocate(DType.F32, out.shape), allocate(DType.F32, out.shape))
    expertsLinear(x, gate, ids, gated)
    expertsLinear(x, up, ids, raised)
    this.gated(Activation.Silu, gated, raised, out)
  }

  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit = {
    val Seq(tokens, hidden) = out.shape.dimensions
    val k = weights.shape.elementCount / tokens
    val (experts, weight) = (reader(expertOut), reader(weights))
    val gates = shared.map { expert =>
      val (input, router) = (reader(expert.input), reader(expert.router))
      Array.tabulate(tokens.toInt)(t =>
        1 / (1 + math.exp(
          -(0L until hidden).map(c => input(t * hidden + c) * router(c)).sum
        ))
      )
    }
    val sharedOutput = shared.map(expert => reader(expert.output))
    val before =
      residual.map(r => CpuOps.copy(reader(r), out.shape.elementCount))
    fill(out) { i =>
      val (t, c) = (i / hidden, i % hidden)
      val routed = (0L until k)
        .map(j => weight(t * k + j) * experts((t * k + j) * hidden + c))
        .sum
      before.fold(0.0)(_(i.toInt)) + routed + gates.fold(0.0)(g =>
        g(t.toInt) * sharedOutput.get(i)
      )
    }
  }
}
