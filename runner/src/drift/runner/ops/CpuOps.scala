package drift.runner.ops

/** The reference backend: plain loops, accumulating in `Double`. Slow on
  * purpose; it is what every kernel is checked against, so it stays simple
  * enough to be read as the maths.
  */
final class CpuOps
    extends CpuBase
    with CpuMemory
    with CpuElementwise
    with CpuNormalisation
    with CpuLinear
    with CpuExperts
    with CpuAttention
    with CpuConvolution
    with CpuLayout
    with CpuSequence

object CpuOps {

  /** A tensor's values, read before `out` (which may be the same) is written.
    */
  def copy(read: Long => Double, count: Long): Array[Double] =
    Array.tabulate(count.toInt)(i => read(i.toLong))

  /** The error function, to about 1e-15: its Taylor series near zero, the
    * continued fraction of erfc beyond.
    */
  def erf(x: Double): Double =
    if (x < 0) -erf(-x)
    else if (x < 2.5) {
      var term = x
      var sum = x
      var n = 0
      while (math.abs(term) > 1e-17 * math.abs(sum)) {
        n += 1
        term *= -x * x / n
        sum += term / (2 * n + 1)
      }
      2 / math.sqrt(math.Pi) * sum
    } else {
      // erfc(x) = exp(−x²)/√π · 1/(x + 1/2/(x + 1/(x + 3/2/(x + …))))
      var fraction = 0.0
      var n = 60
      while (n >= 1) {
        fraction = (n / 2.0) / (x + fraction)
        n -= 1
      }
      1 - math.exp(-x * x) / math.sqrt(math.Pi) / (x + fraction)
    }

  /** A float to bfloat16 bits, rounding to nearest even; NaN stays NaN. */
  def toBfloat16(value: Float): Short = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    if (value.isNaN) ((bits >>> 16) | 0x40).toShort
    else ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16).toShort
  }
}
