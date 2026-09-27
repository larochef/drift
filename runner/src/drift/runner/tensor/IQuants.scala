package drift.runner.tensor

import drift.runner.tensor.LittleEndian.*

import java.lang.foreign.MemorySegment

/** GGML's non-linear 4-bit quants: each nibble indexes a fixed table of 16
  * values (`kvalues_iq4nl`) instead of standing for itself. Element `j` of a
  * 32-element run is the low nibble of byte `j` for `j < 16`, the high nibble
  * of byte `j − 16` after. The arithmetic follows gguf-py's
  * `dequantize_blocks`, in float.
  */
object IQuants {

  private val Values: Array[Float] =
    Array(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89,
      113).map(_.toFloat)

  /** 32 elements of `scale × Values(q)` from the 16 bytes at `at`. */
  private def decodeRun(
      source: MemorySegment,
      at: Long,
      scale: Float,
      target: Array[Float],
      targetOffset: Int
  ): Unit = {
    var j = 0
    while (j < 16) {
      val byte = uint8(source, at + j)
      target(targetOffset + j) = scale * Values(byte & 0xf)
      target(targetOffset + j + 16) = scale * Values(byte >> 4)
      j += 1
    }
  }

  /** `d`, then 16 bytes of nibbles. */
  object IQ4_NL extends DType("IQ4_NL", 32, 18, Some(20), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit =
      decodeRun(
        source,
        offset + 2,
        float16(source, offset),
        target,
        targetOffset
      )
  }

  /** `d`, `scales_h` (u16), `scales_l[4]`, then 128 bytes of nibbles: eight
    * runs of 32, run `i` scaled by `d × (ls − 32)` with a six-bit `ls` (low
    * four bits from `scales_l`, high two from `scales_h`).
    */
  object IQ4_XS extends DType("IQ4_XS", 256, 136, Some(23), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      val high = uint16(source, offset + 2)
      var i = 0
      while (i < 8) {
        val low = (uint8(source, offset + 4 + i / 2) >> (4 * (i % 2))) & 0xf
        val ls = low | (((high >> (2 * i)) & 3) << 4)
        decodeRun(
          source,
          offset + 8 + 16 * i,
          d * (ls - 32).toFloat,
          target,
          targetOffset + 32 * i
        )
        i += 1
      }
    }
  }
}
