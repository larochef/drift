package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.{Arena, MemorySegment}
import java.lang.foreign.ValueLayout.*

/** What every part of the reference backend builds on: the arena its tensors
  * live in, their segments, and the loops over values and rows.
  */
private[ops] trait CpuBase extends Ops {

  protected val arena = Arena.ofShared()

  def name: String = "cpu"

  /** `out(i) = f(i)` over every element. */
  protected def fill(out: Tensor)(f: Long => Double): Unit = {
    val target = segment(out)
    var index = 0L
    while (index < out.shape.elementCount) {
      target.setAtIndex(JAVA_FLOAT, index, f(index).toFloat)
      index += 1
    }
  }

  protected def reader(tensor: Tensor): Long => Double = {
    val source = segment(tensor)
    index => source.getAtIndex(JAVA_FLOAT, index).toDouble
  }

  /** Rows `first until first + count` of a rotated linear (`rotated`), decoded
    * as `ComfyQuant` decodes them: a code times its scales, each group of 256
    * rotated back.
    */
  protected def rotatedRows(
      weight: Tensor,
      first: Long,
      count: Int
  ): Array[Float] = {
    val parts = partsOf(weight)
    val k = weight.shape.last.toInt
    val source = segment(weight)
    val scales = reader(parts.scales)
    val single = parts.scales.shape.elementCount == 1
    val out = new Array[Float](count * k)
    val level: (Long, Int) => Float = parts.levels match {
      case None =>
        (row, column) => source.get(JAVA_BYTE, row * k + column).toFloat
      case Some(l) =>
        val codebook = reader(l.codebook)
        val groups = k / l.groupSize
        val relative = DType.F8E4M3.decode(
          segment(l.relative).asSlice(first * groups, count.toLong * groups),
          count * groups
        )
        (row, column) => {
          val byte = source.get(JAVA_BYTE, row * (k / 2) + column / 2)
          val code = if (column % 2 == 0) byte & 0xf else (byte >> 4) & 0xf
          val scale =
            relative(((row - first) * groups + column / l.groupSize).toInt)
          math
            .max(
              -127.0,
              math.min(127.0, math.rint(codebook(code).toFloat * scale))
            )
            .toFloat
        }
    }
    var row = 0
    while (row < count) {
      val scale = scales(if (single) 0 else first + row).toFloat
      var column = 0
      while (column < k) {
        out(row * k + column) = level(first + row, column) * scale
        column += 1
      }
      row += 1
    }
    ConvRot.rotate(out, 0, out.length, ConvRot.Group)
    out
  }

  /** The host view of a tensor; a registered mapping is readable here too. */
  protected def segment(tensor: Tensor): MemorySegment = {
    val whole = tensor.storage match {
      case Storage.Host(segment)       => segment
      case Storage.Registered(host, _) => host
      case Storage.Device(_, _)        =>
        throw new IllegalArgumentException(
          "the cpu backend cannot read device memory; download the tensor first"
        )
    }
    whole.asSlice(tensor.byteOffset, tensor.byteSize)
  }

  def close(): Unit = arena.close()
}
