package drift.runner

import utest.*

import drift.runner.tensor.{Comparison, Tolerance}

object ComparisonTests extends TestSuite {

  val tests = Tests {
    test("equal values pass an exact comparison") {
      assert(
        Comparison.of(Array(1f, -2f), Array(1f, -2f), Tolerance.Exact).passed
      )
    }
    test("the tolerance is absolute plus relative to the reference") {
      val tolerance = Tolerance(absolute = 0.1, relative = 0.01)
      // allowed: 0.1 + 0.01 × 100 = 1.1
      assert(Comparison.of(Array(100f), Array(101f), tolerance).passed)
      assert(!Comparison.of(Array(100f), Array(101.2f), tolerance).passed)
    }
    test("it reports the first mismatch and the worst errors") {
      val comparison =
        Comparison.of(Array(1f, 2f, 3f), Array(1f, 2.5f, 5f), Tolerance.Exact)
      assert(comparison.mismatches == 2)
      assert(comparison.firstMismatch == Some(1))
      assert(comparison.maxAbsolute == 2.0)
    }
    test("NaN passes only against NaN") {
      assert(
        Comparison
          .of(Array(Float.NaN), Array(Float.NaN), Tolerance.Exact)
          .passed
      )
      assert(
        !Comparison.of(Array(1f), Array(Float.NaN), Tolerance(1e9, 1e9)).passed
      )
      assert(
        !Comparison.of(Array(Float.NaN), Array(1f), Tolerance(1e9, 1e9)).passed
      )
    }
    test("an infinity passes only against the same infinity") {
      val infinite = Float.PositiveInfinity
      assert(
        Comparison.of(Array(infinite), Array(infinite), Tolerance.Exact).passed
      )
      assert(
        !Comparison
          .of(Array(infinite), Array(-infinite), Tolerance(1e9, 1e9))
          .passed
      )
    }
  }
}
