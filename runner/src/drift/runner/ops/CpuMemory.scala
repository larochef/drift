package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*

/** The reference backend's memory and transfers.
  */
private[ops] trait CpuMemory extends CpuBase {

  def allocate(dtype: DType, shape: Shape): Tensor = {
    val bytes = dtype.byteSize(shape.elementCount)
    Tensor(
      dtype,
      shape,
      Storage.Host(arena.allocate(math.max(bytes, 1), 64)),
      0
    )
  }

  def release(tensor: Tensor): Unit = () // the arena frees everything at close

  def mapFile(path: java.nio.file.Path): MappedWeights = {
    val mapped = new drift.runner.tensor.MappedFile(path)
    new MappedWeights(
      mapped.segment,
      Storage.Host(mapped.segment),
      () => mapped.close()
    )
  }

  def fromFloats(shape: Shape, values: Array[Float]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    val tensor = allocate(DType.F32, shape)
    MemorySegment.copy(values, 0, segment(tensor), JAVA_FLOAT, 0, values.length)
    tensor
  }

  def fromInts(shape: Shape, values: Array[Int]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    val tensor = allocate(DType.I32, shape)
    MemorySegment.copy(values, 0, segment(tensor), JAVA_INT, 0, values.length)
    tensor
  }

  def writeInts(tensor: Tensor, values: Array[Int]): Unit = {
    require(
      tensor.dtype == DType.I32 && values.length <= tensor.shape.elementCount,
      s"${values.length} ints into $tensor"
    )
    MemorySegment.copy(values, 0, segment(tensor), JAVA_INT, 0, values.length)
  }

  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor = {
    Ops.checkBytes(dtype, shape, bytes)
    val tensor = allocate(dtype, shape)
    MemorySegment.copy(
      MemorySegment.ofArray(bytes),
      0,
      segment(tensor),
      0,
      bytes.length
    )
    tensor
  }

  def toFloats(tensor: Tensor): Array[Float] = {
    Ops.requireF32("toFloats", tensor)
    segment(tensor).toArray(JAVA_FLOAT)
  }

  def toInts(tensor: Tensor): Array[Int] = {
    require(tensor.dtype == DType.I32, s"toInts of ${tensor.dtype}")
    segment(tensor).toArray(JAVA_INT)
  }

  def convert(x: Tensor, out: Tensor): Unit = {
    Ops.checkConvert(x, out)
    val (source, target) = (segment(x), segment(out))
    val count = x.shape.elementCount
    if (out.dtype == DType.F32) {
      val values = x.dtype.decode(source, count.toInt)
      MemorySegment.copy(values, 0, target, JAVA_FLOAT, 0, values.length)
    } else if (x.dtype != DType.F32) {
      val values = rotatedRows(x, 0, x.shape.dimensions.head.toInt)
      var index = 0
      while (index < count) {
        target.setAtIndex(JAVA_SHORT, index, CpuOps.toBfloat16(values(index)))
        index += 1
      }
    } else {
      var index = 0L
      while (index < count) {
        val value = source.getAtIndex(JAVA_FLOAT, index)
        val bits =
          if (out.dtype == DType.F16) java.lang.Float.floatToFloat16(value)
          else CpuOps.toBfloat16(value)
        target.setAtIndex(JAVA_SHORT, index, bits)
        index += 1
      }
    }
  }

  def zero(tensor: Tensor): Unit = segment(tensor).fill(0.toByte)

  def copy(from: Tensor, to: Tensor): Unit = {
    Ops.checkCopy(from, to)
    MemorySegment.copy(segment(from), 0, segment(to), 0, from.byteSize)
  }
}
