package drift.runner.tensor

import drift.runner.tensor.LittleEndian.*

import java.lang.foreign.MemorySegment

/** GGML's original 32-element quants. Element `j` of a block is the low nibble
  * of byte `j` for `j < 16`, the high nibble of byte `j − 16` after. The
  * arithmetic follows gguf-py's `dequantize_blocks`, in float.
  */
object GgmlQuants {

  /** `d × (q − 8)`; `d` f16, 16 bytes of nibbles. */
  object Q4_0 extends DType("Q4_0", 32, 18, Some(2), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      var j = 0
      while (j < 16) {
        val byte = uint8(source, offset + 2 + j)
        target(targetOffset + j) = d * ((byte & 0xf) - 8).toFloat
        target(targetOffset + j + 16) = d * ((byte >> 4) - 8).toFloat
        j += 1
      }
    }
  }

  /** `d × q + m`; `d`, `m` f16. */
  object Q4_1 extends DType("Q4_1", 32, 20, Some(3), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      val m = float16(source, offset + 2)
      var j = 0
      while (j < 16) {
        val byte = uint8(source, offset + 4 + j)
        target(targetOffset + j) = d * (byte & 0xf).toFloat + m
        target(targetOffset + j + 16) = d * (byte >> 4).toFloat + m
        j += 1
      }
    }
  }

  /** The fifth bit of element `j` is bit `j` of a 32-bit `qh`. */
  private def fiveBit(
      source: MemorySegment,
      offset: Long,
      qhAt: Long,
      qsAt: Long,
      j: Int
  ): Int = {
    val qh = uint32(source, offset + qhAt)
    val nibbles = uint8(source, offset + qsAt + (j % 16))
    val low = if (j < 16) nibbles & 0xf else nibbles >> 4
    low | (((qh >> j) & 1).toInt << 4)
  }

  /** `d × (q − 16)`, q of five bits. */
  object Q5_0 extends DType("Q5_0", 32, 22, Some(6), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      var j = 0
      while (j < 32) {
        target(targetOffset + j) =
          d * (fiveBit(source, offset, 2, 6, j) - 16).toFloat
        j += 1
      }
    }
  }

  /** `d × q + m`, q of five bits. */
  object Q5_1 extends DType("Q5_1", 32, 24, Some(7), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      val m = float16(source, offset + 2)
      var j = 0
      while (j < 32) {
        target(targetOffset + j) =
          d * fiveBit(source, offset, 4, 8, j).toFloat + m
        j += 1
      }
    }
  }

  /** `q × d`, q a signed byte. */
  object Q8_0 extends DType("Q8_0", 32, 34, Some(8), None) {
    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val d = float16(source, offset)
      var j = 0
      while (j < 32) {
        target(targetOffset + j) = int8(source, offset + 2 + j).toFloat * d
        j += 1
      }
    }
  }
}
