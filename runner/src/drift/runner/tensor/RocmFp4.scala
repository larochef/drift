package drift.runner.tensor

import drift.runner.tensor.LittleEndian.*

import java.lang.foreign.MemorySegment

/** ROCmFP4 (`specs/42`, Weight formats): an unofficial 4-bit format for gfx1151
  * from the rocmfp4-llama fork. 32 nibbles in 16 bytes, element `j` the low
  * nibble of byte `j` for `j < 16` and the high nibble of byte `j − 16` after,
  * each a code into a signed "Codebook10", times half an unsigned E4M3 scale.
  * `Dual` has one scale per 16-element half, `Fast` one for the block. Decoding
  * follows `rocmfp4_dequantize_row_q4_0` in the fork's
  * `ggml/rocmfp4/rocmfp4.c`.
  *
  * Its GGUF type ids (100, 101) are the fork's and may one day collide with an
  * official type, so a file claiming them is also checked: the tensor spans
  * must fit these block sizes (`GgufFile`) and the scale bytes must be finite.
  */
object RocmFp4 {

  private val Codebook =
    Array[Float](0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)

  /** The largest finite unsigned E4M3 byte; 0x7f is NaN, the top bit unused. */
  private val MaxScale = 0x7e

  /** Half the unsigned E4M3 value of a scale byte; 0 for an invalid byte, as
    * the reference does.
    */
  def halfScale(byte: Int): Float =
    if (byte > MaxScale) 0f
    else {
      val exponent = byte >> 3
      val mantissa = byte & 0x7
      if (exponent == 0) java.lang.Math.scalb(mantissa.toFloat, -10)
      else (8 + mantissa) * java.lang.Math.scalb(1f, exponent - 11)
    }

  /** How many blocks `rejects` samples from the start of a tensor. */
  private val SampledBlocks = 4096

  abstract private class Layout(name: String, bytes: Int, ggmlId: Int)
      extends DType(name, 32, bytes, Some(ggmlId), None) {

    /** The offsets of the scale bytes within a block. */
    def scaleOffsets: Seq[Int]

    def decodeBlock(
        source: MemorySegment,
        offset: Long,
        target: Array[Float],
        targetOffset: Int
    ): Unit = {
      val low = halfScale(uint8(source, offset + scaleOffsets.head))
      val high = halfScale(uint8(source, offset + scaleOffsets.last))
      var j = 0
      while (j < 16) {
        val byte = uint8(source, offset + j)
        target(targetOffset + j) = Codebook(byte & 0xf) * low
        target(targetOffset + j + 16) = Codebook(byte >> 4) * high
        j += 1
      }
    }

    override def rejects(bytes: MemorySegment): Option[String] = {
      val blocks = math.min(bytes.byteSize() / blockBytes, SampledBlocks)
      (0L until blocks).iterator
        .flatMap(block => scaleOffsets.map(block * blockBytes + _))
        .find(at => uint8(bytes, at) > MaxScale)
        .map(at =>
          f"scale byte 0x${uint8(bytes, at)}%02x at byte $at is not a finite unsigned E4M3, so this is not $name"
        )
    }
  }

  /** `Q4_0_ROCMFP4`: two scales, 4.5 bits per weight. */
  val Dual: DType = new Layout("Q4_0_ROCMFP4", 18, 100) {
    val scaleOffsets: Seq[Int] = Seq(16, 17)
  }

  /** `Q4_0_ROCMFP4_FAST`: one scale, 4.25 bits per weight. */
  val Fast: DType = new Layout("Q4_0_ROCMFP4_FAST", 17, 101) {
    val scaleOffsets: Seq[Int] = Seq(16)
  }
}
