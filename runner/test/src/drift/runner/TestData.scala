package drift.runner

import scala.util.Random

import drift.runner.tensor.DType

/** Seeded inputs, so a failure reproduces. */
object TestData {

  def gaussian(seed: Long, count: Int): Array[Float] = {
    val random = new Random(seed)
    Array.fill(count)(random.nextGaussian().toFloat)
  }

  /** Where each quantized type keeps its f16 scales, which must be finite, and
    * its E4M3 scale bytes, which must be at most 0x7e.
    */
  private val halfScales = Map(
    "Q8_0" -> Seq(0),
    "Q4_0" -> Seq(0),
    "Q4_K" -> Seq(0, 2),
    "Q6_K" -> Seq(208),
    "Q5_1" -> Seq(0, 2),
    "Q5_K" -> Seq(0, 2),
    "IQ4_NL" -> Seq(0),
    "IQ4_XS" -> Seq(0)
  )
  private val e4m3Scales =
    Map("Q4_0_ROCMFP4" -> Seq(16, 17), "Q4_0_ROCMFP4_FAST" -> Seq(16))

  /** `rows × cols` random weights of a quantized type: random bytes, so every
    * code occurs, with plausible scales.
    */
  def quantized(dtype: DType, rows: Int, cols: Int, seed: Long): Array[Byte] = {
    val random = new Random(seed)
    if (dtype.blockElements == 1) return dense(dtype, rows * cols, random)
    val bytes = new Array[Byte](dtype.byteSize(rows.toLong * cols).toInt)
    random.nextBytes(bytes)
    val buffer =
      java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    for (block <- 0 until bytes.length / dtype.blockBytes) {
      val start = block * dtype.blockBytes
      halfScales.getOrElse(dtype.name, Nil).foreach { at =>
        val scale = (random.nextGaussian() * 0.01).toFloat
        buffer.putShort(start + at, java.lang.Float.floatToFloat16(scale))
      }
      e4m3Scales.getOrElse(dtype.name, Nil).foreach { at =>
        bytes(start + at) = random.nextInt(0x7f).toByte
      }
    }
    bytes
  }

  /** Gaussian values stored in a dense type. */
  private def dense(dtype: DType, count: Int, random: Random): Array[Byte] = {
    val buffer = java.nio.ByteBuffer
      .allocate(count * dtype.blockBytes)
      .order(java.nio.ByteOrder.LITTLE_ENDIAN)
    (0 until count).foreach { _ =>
      val value = (random.nextGaussian() * 0.05).toFloat
      dtype match {
        case DType.F32 => buffer.putFloat(value)
        case DType.F16 => buffer.putShort(java.lang.Float.floatToFloat16(value))
        case _ => buffer.putShort(drift.runner.ops.CpuOps.toBfloat16(value))
      }
    }
    buffer.array()
  }
}
