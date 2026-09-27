package drift.runner.tensor

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.nio.ByteOrder

/** Reads the little-endian values model files are made of, at any alignment.
  */
object LittleEndian {

  private val Short = JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)
  private val Int = JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)
  private val Long = JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)
  private val Float = JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)
  private val Double = JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)

  def int8(segment: MemorySegment, offset: Long): Int =
    segment.get(JAVA_BYTE, offset).toInt

  def uint8(segment: MemorySegment, offset: Long): Int =
    segment.get(JAVA_BYTE, offset) & 0xff

  def int16(segment: MemorySegment, offset: Long): Int =
    segment.get(Short, offset).toInt

  def uint16(segment: MemorySegment, offset: Long): Int =
    segment.get(Short, offset) & 0xffff

  def int32(segment: MemorySegment, offset: Long): Int =
    segment.get(Int, offset)

  def uint32(segment: MemorySegment, offset: Long): Long =
    segment.get(Int, offset) & 0xffffffffL

  /** Signed; an unsigned value above `Long.MaxValue` comes out negative. */
  def int64(segment: MemorySegment, offset: Long): Long =
    segment.get(Long, offset)

  def float32(segment: MemorySegment, offset: Long): Float =
    segment.get(Float, offset)

  def float64(segment: MemorySegment, offset: Long): Double =
    segment.get(Double, offset)

  def float16(segment: MemorySegment, offset: Long): Float =
    java.lang.Float.float16ToFloat(segment.get(Short, offset))
}
