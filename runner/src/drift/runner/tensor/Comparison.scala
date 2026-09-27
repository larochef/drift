package drift.runner.tensor

/** An element passes when `|actual − expected| ≤ absolute + relative ×
  * |expected|`.
  */
final case class Tolerance(absolute: Double, relative: Double)

object Tolerance {
  val Exact: Tolerance = Tolerance(0, 0)
}

/** How far one result is from its reference. A NaN or an infinity passes only
  * where the reference has the same one.
  */
final case class Comparison(
    count: Int,
    mismatches: Int,
    firstMismatch: Option[Int],
    maxAbsolute: Double,
    maxRelative: Double
) {
  def passed: Boolean = mismatches == 0

  def summary: String =
    s"$mismatches/$count mismatches" +
      firstMismatch.fold("")(index => s", first at $index") +
      f", max absolute $maxAbsolute%.3g, max relative $maxRelative%.3g"
}

object Comparison {

  def of(
      expected: Array[Float],
      actual: Array[Float],
      tolerance: Tolerance
  ): Comparison = {
    require(
      expected.length == actual.length,
      s"comparing ${actual.length} values with ${expected.length}"
    )
    var mismatches = 0
    var firstMismatch = Option.empty[Int]
    var maxAbsolute = 0.0
    var maxRelative = 0.0
    var index = 0
    while (index < expected.length) {
      val reference = expected(index).toDouble
      val value = actual(index).toDouble
      val passes =
        if (reference.isNaN || value.isNaN) reference.isNaN && value.isNaN
        else if (reference.isInfinite || value.isInfinite) reference == value
        else {
          val absolute = math.abs(value - reference)
          maxAbsolute = math.max(maxAbsolute, absolute)
          if (reference != 0)
            maxRelative = math.max(maxRelative, absolute / math.abs(reference))
          absolute <= tolerance.absolute + tolerance.relative * math.abs(
            reference
          )
        }
      if (!passes) {
        mismatches += 1
        if (firstMismatch.isEmpty) firstMismatch = Some(index)
      }
      index += 1
    }
    Comparison(
      expected.length,
      mismatches,
      firstMismatch,
      maxAbsolute,
      maxRelative
    )
  }
}
