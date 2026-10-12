package drift.runner.tensor

/** Single precision rounded to BF16, to nearest with ties to even: what a model
  * computed in BF16 keeps of a value, and how host floats become BF16 weights.
  */
object Bf16 {

  private def high(value: Float): Int = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16
  }

  /** `value` as the float its BF16 rounding is. */
  def round(value: Float): Float =
    java.lang.Float.intBitsToFloat(high(value) << 16)

  /** `values` as BF16, little-endian. */
  def bytes(values: Array[Float]): Array[Byte] = {
    val out = new Array[Byte](2 * values.length)
    values.indices.foreach { i =>
      val rounded = high(values(i))
      out(2 * i) = rounded.toByte
      out(2 * i + 1) = (rounded >>> 8).toByte
    }
    out
  }
}
