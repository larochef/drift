package drift.runner

import utest.*

import drift.runner.decode.{Sampler, Sampling}
import drift.runner.ops.Candidates

import scala.util.Random

object SamplerTests extends TestSuite {

  private val logits = Array(1f, 3f, 2f, 3.5f, -1f, 0f)

  private def draws(sampling: Sampling, count: Int): Seq[Int] = {
    val sampler = new Sampler(sampling)
    Seq.fill(count)(sampler.next(logits))
  }

  /** The sampler before candidates: every row sorted whole, boxed. */
  final private class FullSortSampler(sampling: Sampling) {
    private val random = new Random(sampling.seed)
    def next(logits: Array[Float]): Int = {
      val order = logits.indices.sortBy(i => -logits(i)).toArray
      val kept = if (sampling.topK > 0) order.take(sampling.topK) else order
      val largest = logits(kept.head).toDouble
      val weights =
        kept.map(i => math.exp((logits(i) - largest) / sampling.temperature))
      val total = weights.sum
      val probabilities = weights.map(_ / total)
      var cumulative = 0.0
      val nucleus = probabilities.indices.takeWhile { i =>
        val inside = i == 0 || cumulative < sampling.topP
        cumulative += probabilities(i)
        inside
      }.size
      val floor = sampling.minP * probabilities.head
      val chosen = (0 until nucleus).filter(i => probabilities(i) >= floor)
      val mass = chosen.map(probabilities).sum
      var draw = random.nextDouble() * mass
      chosen
        .find { i =>
          draw -= probabilities(i)
          draw <= 0
        }
        .map(kept)
        .getOrElse(kept(chosen.last))
    }
  }

  /** Rows of `width` logits like a model's: a few likely tokens, a long tail,
    * ties; `spread` scales them (small: a flat row, a wide nucleus).
    */
  private def rows(seed: Long, count: Int, width: Int, spread: Float) =
    Seq.tabulate(count)(r =>
      TestData
        .gaussian(seed + r, width)
        .map(v => math.round(v * 8) / 8f * spread + 0.01f)
    )

  /** The candidate sampler's draws against the full sort's, same seed. */
  private def sameDraws(sampling: Sampling, rows: Seq[Array[Float]]): Unit = {
    val sampler = new Sampler(sampling)
    val reference = new FullSortSampler(sampling)
    var wholeRows = 0
    val drawn = rows.map { row =>
      val candidates =
        Candidates.of(row, row.length, sampler.candidateCount).head
      sampler.next(candidates, { wholeRows += 1; row })
    }
    assert(drawn == rows.map(reference.next))
    if (sampling.topK > 0 && sampling.topK <= sampler.candidateCount)
      assert(wholeRows == 0)
  }

  val tests = Tests {
    test("candidates: the largest first, the lower index among equals") {
      rows(7, 3, 3000, 1f).foreach { row =>
        val expected = row.indices.sortBy(i => -row(i)).take(300)
        val candidates = Candidates.of(row, row.length, 300).head
        assert(candidates.ids.toSeq == expected)
        assert(candidates.values.toSeq == expected.map(row))
      }
    }
    test("candidates draw what the full sort draws") {
      val wide = rows(11, 300, 20000, 1.5f)
      sameDraws(Sampling(0.8f, 40, 0.95f, 0.05f, 21), wide)
      sameDraws(Sampling(1f, 20, 0.8f, 0f, 22), wide)
      sameDraws(Sampling(0.6f, 1024, 1f, 0f, 23), wide)
      sameDraws(Sampling(0.7f, 0, 0.9f, 0.05f, 24), wide)
      sameDraws(Sampling(1f, 0, 1f, 0.1f, 25), wide)
      // a few likely tokens: the nucleus within the candidates
      val peaked =
        wide.map(_.zipWithIndex.map((v, i) => if (i % 97 == 3) v + 12 else v))
      sameDraws(Sampling(0.7f, 0, 0.9f, 0f, 28), peaked)
      sameDraws(Sampling(1f, 0, 0.95f, 0.05f, 29), peaked)
    }
    test("a nucleus past the candidates falls back to the whole row") {
      val flat = rows(13, 40, 20000, 0.05f)
      val sampling = Sampling(1f, 0, 0.95f, 0f, 26)
      val sampler = new Sampler(sampling)
      var wholeRows = 0
      val drawn = flat.map { row =>
        sampler.next(
          Candidates.of(row, row.length, sampler.candidateCount).head,
          { wholeRows += 1; row }
        )
      }
      assert(wholeRows == flat.size)
      assert(drawn == flat.map(new FullSortSampler(sampling).next))
      // more than the top-k candidates can hold: the whole row too
      sameDraws(Sampling(1f, 2000, 1f, 0f, 27), rows(14, 30, 20000, 0.05f))
    }
    test("greedy takes the first candidate") {
      val row = rows(15, 1, 20000, 1f).head
      val candidates = Candidates.of(row, row.length, 1).head
      assert(
        new Sampler(Sampling.Greedy).next(candidates, ???) ==
          new Sampler(Sampling.Greedy).next(row)
      )
    }
    test("greedy takes the largest, the first among equals") {
      assert(new Sampler(Sampling.Greedy).next(logits) == 3)
      assert(new Sampler(Sampling.Greedy).next(Array(1f, 5f, 5f)) == 1)
    }
    test("top-k never leaves the k most likely") {
      assert(draws(Sampling(1f, 2, 1f, 0f, 1), 500).toSet == Set(1, 3))
    }
    test("top-p keeps the smallest set reaching p") {
      // probabilities at temperature 1: 3.5 → 0.52, 3 → 0.32, 2 → 0.12, …
      assert(draws(Sampling(1f, 0, 0.8f, 0f, 2), 500).toSet == Set(1, 3))
      assert(draws(Sampling(1f, 0, 0.3f, 0f, 2), 200).toSet == Set(3))
    }
    test("min-p drops what is below p × the top probability") {
      assert(draws(Sampling(1f, 0, 1f, 0.5f, 3), 500).toSet == Set(1, 3))
    }
    test("a seed repeats its draws; frequencies follow the probabilities") {
      val sampling = Sampling(1f, 0, 1f, 0f, 4)
      val first = draws(sampling, 4000)
      assert(first == draws(sampling, 4000))
      val weights = logits.map(l => math.exp(l - 3.5))
      val expected = weights(3) / weights.sum
      val observed = first.count(_ == 3) / 4000.0
      assert(math.abs(observed - expected) < 0.03)
    }
  }
}
