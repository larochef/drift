package drift.runner.tensor

import java.lang.foreign.MemorySegment

/** ConvRot's rotated basis (`ComfyQuant`): every `Group` columns of a weight
  * times a normalized regular Hadamard matrix, symmetric and its own inverse.
  */
object ConvRot {

  /** The only rotation checked against the official weights: 256 columns, the
    * Kronecker fourth power of the 4 × 4 regular Hadamard matrix.
    */
  val Group = 256

  /** `length` values from `from`, every `group` of them times the normalized
    * regular Hadamard matrix of that size: the Kronecker power of `[[1, 1, 1,
    * −1], [1, 1, −1, 1], [1, −1, 1, 1], [−1, 1, 1, 1]] / 2`, one factor a
    * base-4 digit of the column.
    */
  def rotate(
      values: Array[Float],
      from: Int,
      length: Int,
      group: Int
  ): Unit = {
    var start = from
    while (start < from + length) {
      var stride = 1
      while (stride < group) {
        var block = start
        while (block < start + group) {
          var i = block
          while (i < block + stride) {
            val a = values(i)
            val b = values(i + stride)
            val c = values(i + 2 * stride)
            val d = values(i + 3 * stride)
            val half = (a + b + c + d) * 0.5f
            values(i) = half - d
            values(i + stride) = half - c
            values(i + 2 * stride) = half - b
            values(i + 3 * stride) = half - a
            i += 1
          }
          block += 4 * stride
        }
        stride *= 4
      }
      start += group
    }
  }

  /** A rotated linear read where the file holds it, undecoded (`bugs/53`): the
    * codes alone, which only a backend that was given their scales
    * (`Ops.rotated`) can decode — each product does. `Codes` are
    * `int8_tensorwise`'s, a value a byte; `Nibbles` `asym_w4a8_int8`'s, two
    * values a byte.
    */
  sealed abstract class Stored(label: String, valuesPerByte: Int)
      extends DType(label, valuesPerByte, 1, None, None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = throw new UnsupportedOperationException(
      s"$label codes are decoded with their scales, by the backend that holds them"
    )
  }
  object Codes extends Stored("CONVROT_I8", 1)
  object Nibbles extends Stored("CONVROT_W4A8", 2)

  def stored(dtype: DType): Boolean = dtype == Codes || dtype == Nibbles
}
