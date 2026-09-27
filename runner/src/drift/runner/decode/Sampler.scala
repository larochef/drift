package drift.runner.decode

import drift.runner.ops.{Candidates, Ops}

import scala.util.Random

/** How the next token is picked from the logits. A temperature of 0 is greedy
  * (the most likely token, the first among equals). Otherwise, in transformers'
  * order: divide by the temperature, keep the `topK` most likely (0: all), keep
  * the smallest set reaching `topP` of the probability, drop what is below
  * `minP` × the top probability, then draw.
  */
final case class Sampling(
    temperature: Float,
    topK: Int,
    topP: Float,
    minP: Float,
    seed: Long
)

object Sampling {
  val Greedy: Sampling = Sampling(0f, 0, 1f, 0f, 0L)
}

/** Draws from a row's candidates, its largest logits sorted on the GPU, so the
  * whole row rarely comes back: with a top-k the candidates are all that is
  * kept. Without one, the row comes back for the weight of the rest, and a
  * nucleus reaching past the candidates falls back to sorting it all. The same
  * arithmetic in the same order as over the sorted row: the same draws.
  */
final class Sampler(sampling: Sampling) {

  private val random = new Random(sampling.seed)

  /** The candidates a row needs: the one greedy takes, the top-k kept, or else
    * `Sampler.NucleusCandidates`.
    */
  val candidateCount: Int =
    if (sampling.temperature <= 0) 1
    else if (sampling.topK > 0)
      math.min(sampling.topK, Ops.MaximumCandidates)
    else Sampler.NucleusCandidates

  /** The next token from a row's `candidates` (`candidateCount`, or all the
    * row's when it is narrower); `row` brings the whole row back when they are
    * not enough.
    */
  def next(candidates: Candidates, row: => Array[Float]): Int =
    if (sampling.temperature <= 0) candidates.ids(0)
    else if (candidates.size < candidateCount)
      draw(candidates.ids, candidates.values, candidates.size, 0, true).get
    else if (sampling.topK > 0) {
      if (candidates.size >= sampling.topK)
        draw(candidates.ids, candidates.values, sampling.topK, 0, true).get
      else next(row)
    } else {
      val logits = row
      val largest = candidates.values(0).toDouble
      val candidate = new Array[Boolean](logits.length)
      candidates.ids.foreach(candidate(_) = true)
      var rest = 0.0
      var i = 0
      while (i < logits.length) {
        if (!candidate(i))
          rest += math.exp((logits(i) - largest) / sampling.temperature)
        i += 1
      }
      draw(candidates.ids, candidates.values, candidates.size, rest, false)
        .getOrElse(next(logits))
    }

  /** The next token from a whole row of logits. */
  def next(logits: Array[Float]): Int =
    if (sampling.temperature <= 0) greedy(logits)
    else {
      val order = Candidates.order(logits, 0, logits.length)
      val kept =
        if (sampling.topK > 0) math.min(sampling.topK, order.length)
        else order.length
      draw(
        order,
        Array.tabulate(kept)(i => logits(order(i))),
        kept,
        0,
        true
      ).get
    }

  /** Draws among the first `count` of `ids`, the most likely tokens in order
    * (`values` their logits), `rest` the weight of the kept tokens after them
    * (`complete` when there are none); None when the draw could reach those.
    */
  private def draw(
      ids: Array[Int],
      values: Array[Float],
      count: Int,
      rest: Double,
      complete: Boolean
  ): Option[Int] = {
    val largest = values(0).toDouble
    val weights = new Array[Double](count)
    var total = 0.0
    var i = 0
    while (i < count) {
      weights(i) = math.exp((values(i) - largest) / sampling.temperature)
      total += weights(i)
      i += 1
    }
    total += rest
    val probabilities = weights.map(_ / total)
    // nucleus: the smallest prefix reaching topP, at least one token
    var cumulative = 0.0
    var nucleus = 0
    while (nucleus < count && (nucleus == 0 || cumulative < sampling.topP)) {
      cumulative += probabilities(nucleus)
      nucleus += 1
    }
    val floor = sampling.minP * probabilities.head
    // the probabilities fall: min-p keeps a prefix of the nucleus
    var chosen = 0
    while (chosen < nucleus && probabilities(chosen) >= floor) chosen += 1
    val cut = complete || nucleus < count || chosen < count
    if (!cut) None
    else {
      var mass = 0.0
      (0 until chosen).foreach(i => mass += probabilities(i))
      var drawn = random.nextDouble() * mass
      var at = 0
      while (at < chosen - 1 && { drawn -= probabilities(at); drawn > 0 })
        at += 1
      Some(ids(at))
    }
  }

  private def greedy(logits: Array[Float]): Int = {
    var best = 0
    var i = 1
    while (i < logits.length) {
      if (logits(i) > logits(best)) best = i
      i += 1
    }
    best
  }
}

object Sampler {

  /** Candidates fetched without a top-k: the nucleus of most rows. */
  val NucleusCandidates = 256
}
