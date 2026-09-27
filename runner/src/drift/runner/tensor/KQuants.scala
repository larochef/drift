package drift.runner.tensor

import drift.runner.tensor.LittleEndian.*

import java.lang.foreign.MemorySegment

/** GGML's K-quants: super-blocks of 256 elements in sub-blocks of 16 or 32,
  * each with its own scale. Element `e` of a block is decoded on its own, with
  * the bit positions and the float arithmetic of gguf-py's `dequantize_blocks`.
  */
object KQuants {

  private val Elements = 256

  /** A K-quant block, decoded one element at a time. */
  abstract private class KQuant(
      name: String,
      bytes: Int,
      ggmlId: Int
  ) extends DType(name, Elements, bytes, Some(ggmlId), None) {

    def element(source: MemorySegment, offset: Long, e: Int): Float

    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      var e = 0
      while (e < Elements) {
        target(targetOffset + e) = element(source, offset, e)
        e += 1
      }
    }
  }

  /** The two bits of element `e` in a 64-byte field packed by 128-element
    * halves: four 2-bit planes of 32 bytes each.
    */
  private def twoBits(source: MemorySegment, at: Long, e: Int): Int = {
    val half = e / 128
    val plane = (e % 128) / 32
    (uint8(source, at + half * 32 + e % 32) >> (2 * plane)) & 3
  }

  /** The nibble of element `e` in a 128-byte field of 64-element chunks: the
    * low nibbles of 32 bytes, then their high nibbles.
    */
  private def nibble(source: MemorySegment, at: Long, e: Int): Int = {
    val chunk = e / 32
    (uint8(source, at + (chunk / 2) * 32 + e % 32) >> (4 * (chunk % 2))) & 0xf
  }

  /** Q4_K's and Q5_K's eight 6-bit scales and mins, packed in 12 bytes. */
  private def scaleAndMin(
      source: MemorySegment,
      at: Long,
      j: Int
  ): (Int, Int) = {
    def byte(i: Int) = uint8(source, at + i)
    if (j < 4) (byte(j) & 63, byte(j + 4) & 63)
    else
      (
        (byte(j + 4) & 0xf) | ((byte(j - 4) >> 2) & 0x30),
        (byte(j + 4) >> 4) | ((byte(j) >> 2) & 0x30)
      )
  }

  /** scales[16] (4-bit scale, 4-bit min), qs[64], d, dmin:
    * `d × sc × q − dmin × m`.
    */
  val Q2_K: DType = new KQuant("Q2_K", 84, 10) {
    def element(source: MemorySegment, offset: Long, e: Int): Float = {
      val scale = uint8(source, offset + e / 16)
      val dl = float16(source, offset + 80) * (scale & 0xf).toFloat
      val ml = float16(source, offset + 82) * (scale >> 4).toFloat
      dl * twoBits(source, offset + 16, e).toFloat - ml
    }
  }

  /** hmask[32], qs[64], scales[12] (6-bit, offset 32), d:
    * `d × (sc − 32) × (q − 4 × !hbit)`.
    */
  val Q3_K: DType = new KQuant("Q3_K", 110, 11) {
    def element(source: MemorySegment, offset: Long, e: Int): Float = {
      val i = e / 16
      val low =
        if (i < 8) uint8(source, offset + 96 + i) & 0xf
        else uint8(source, offset + 96 + i - 8) >> 4
      val high = (uint8(source, offset + 104 + i % 4) >> (2 * (i / 4))) & 3
      val scale = (low | (high << 4)) - 32
      val dl = float16(source, offset + 108) * scale.toFloat
      val highBit = (uint8(source, offset + e % 32) >> (e / 32)) & 1
      val q = twoBits(source, offset + 32, e) - ((highBit ^ 1) << 2)
      dl * q.toFloat
    }
  }

  /** d, dmin, scales[12], qs[128]: `d × sc × q − dmin × m`, per 32. */
  val Q4_K: DType = new KQuant("Q4_K", 144, 12) {
    def element(source: MemorySegment, offset: Long, e: Int): Float = {
      val (scale, min) = scaleAndMin(source, offset + 4, e / 32)
      val d = float16(source, offset) * scale.toFloat
      val m = float16(source, offset + 2) * min.toFloat
      d * nibble(source, offset + 16, e).toFloat - m
    }
  }

  /** d, dmin, scales[12], qh[32], qs[128]: Q4_K with a fifth bit. */
  val Q5_K: DType = new KQuant("Q5_K", 176, 13) {
    def element(source: MemorySegment, offset: Long, e: Int): Float = {
      val (scale, min) = scaleAndMin(source, offset + 4, e / 32)
      val d = float16(source, offset) * scale.toFloat
      val m = float16(source, offset + 2) * min.toFloat
      val highBit = (uint8(source, offset + 16 + e % 32) >> (e / 32)) & 1
      d * (nibble(source, offset + 48, e) | (highBit << 4)).toFloat - m
    }
  }

  /** ql[128], qh[64], scales[16] (int8), d: `d × sc × (q − 32)`, q of six bits.
    */
  val Q6_K: DType = new KQuant("Q6_K", 210, 14) {
    def element(source: MemorySegment, offset: Long, e: Int): Float = {
      val half = e / 128
      val r = e % 128
      val low =
        (uint8(source, offset + half * 64 + r % 64) >> (4 * (r / 64))) & 0xf
      val q = (low | (twoBits(source, offset + 128, e) << 4)) - 32
      val d = float16(source, offset + 208) *
        int8(source, offset + 192 + e / 16).toFloat
      d * q.toFloat
    }
  }
}
