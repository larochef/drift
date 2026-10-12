package drift.runner.ops

import drift.runner.native.*
import drift.runner.tensor.*

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.util.control.NonFatal

/** The GPU backend's memory and transfers: buffers taken from the spares or the
  * device, uploads and read-backs staged in host memory, conversions.
  */
private[ops] trait HipMemory extends HipBase {

  def allocate(dtype: DType, shape: Shape): Tensor = {
    val bytes = math.max(dtype.byteSize(shape.elementCount), 1)
    val pointer = takeSpare(bytes).getOrElse(
      try hip.allocate(bytes)
      catch {
        // No room on the device: the spares are room.
        case NonFatal(_) if spare.nonEmpty =>
          while (spare.nonEmpty) freeOldestSpare()
          hip.allocate(bytes)
      }
    )
    allocations(pointer.address()) = pointer
    Tensor(dtype, shape, Storage.Device(pointer, bytes), 0)
  }

  def release(tensor: Tensor): Unit = tensor.storage match {
    case Storage.Device(pointer, bytes) if tensor.byteOffset == 0 =>
      allocations
        .remove(pointer.address())
        .foreach(owned => keepSpare(owned, bytes))
    case _ => () // views, registered mappings: not ours to free
  }

  def mapFile(path: java.nio.file.Path): MappedWeights = {
    val registered = new drift.runner.native.RegisteredFile(hip, path)
    new MappedWeights(
      registered.host,
      Storage.Registered(registered.host, registered.device),
      () => registered.close()
    )
  }

  /** A new tensor filled from host memory by `write`. */
  private def upload(dtype: DType, shape: Shape)(
      write: MemorySegment => Unit
  ): Tensor = {
    val tensor = allocate(dtype, shape)
    val arena = Arena.ofConfined()
    try {
      val staging = arena.allocate(math.max(tensor.byteSize, 1), 64)
      write(staging)
      hip.copy(pointer(tensor), staging, tensor.byteSize)
      deviceWrites += 1
    } finally arena.close()
    tensor
  }

  def fromFloats(shape: Shape, values: Array[Float]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    upload(DType.F32, shape)(
      MemorySegment.copy(values, 0, _, JAVA_FLOAT, 0, values.length)
    )
  }

  def fromInts(shape: Shape, values: Array[Int]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    upload(DType.I32, shape)(
      MemorySegment.copy(values, 0, _, JAVA_INT, 0, values.length)
    )
  }

  def writeInts(tensor: Tensor, values: Array[Int]): Unit = {
    require(
      tensor.dtype == DType.I32 && values.length <= tensor.shape.elementCount,
      s"${values.length} ints into ${tensor.shape}"
    )
    // as many copies as the staging takes: a 2048² picture with a reference
    // of its size has more positions than one holds
    val target = pointer(tensor)
    var written = 0
    while (written < values.length) {
      val count = math.min(HipBase.StagedInts, values.length - written)
      MemorySegment.copy(values, written, intStaging, JAVA_INT, 0, count)
      hip.copy(offset(target, 4L * written), intStaging, 4L * count)
      written += count
    }
    deviceWrites += 1
  }

  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor = {
    Ops.checkBytes(dtype, shape, bytes)
    upload(dtype, shape)(
      MemorySegment.copy(MemorySegment.ofArray(bytes), 0, _, 0, bytes.length)
    )
  }

  def toFloats(tensor: Tensor): Array[Float] = {
    Ops.requireF32("toFloats", tensor)
    download(tensor)(_.toArray(JAVA_FLOAT))
  }

  def toInts(tensor: Tensor): Array[Int] = {
    require(tensor.dtype == DType.I32, s"toInts of ${tensor.dtype}")
    download(tensor)(_.toArray(JAVA_INT))
  }

  /** `read` of a tensor's bytes, staged in host memory. */
  private def download[A](tensor: Tensor)(read: MemorySegment => A): A = {
    val arena = Arena.ofConfined()
    try {
      val staging = arena.allocate(math.max(tensor.byteSize, 1), 64)
      hip.copy(staging, pointer(tensor), tensor.byteSize)
      read(staging.asSlice(0, tensor.byteSize))
    } finally arena.close()
  }

  def convert(x: Tensor, out: Tensor): Unit = {
    Ops.checkConvert(x, out)
    if (ConvRot.stored(x.dtype))
      return dequantizeRotated(x, pointer(out))
    convertAt(
      pointer(x),
      pointer(out),
      x.shape.elementCount,
      conversionCodes(x.dtype -> out.dtype)
    )
  }

  def zero(tensor: Tensor): Unit = {
    hip.zero(pointer(tensor), tensor.byteSize)
    deviceWrites += 1
  }

  def copy(from: Tensor, to: Tensor): Unit = {
    Ops.checkCopy(from, to)
    hip.copyAsync(pointer(to), pointer(from), from.byteSize)
    deviceWrites += 1
  }
}
