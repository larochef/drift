package drift.runner.formats

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment

/** A tensor as a model file stores it: its bytes are a slice of the file's
  * mapping, at `fileOffset`, never copied.
  */
final case class StoredTensor(
    name: String,
    dtype: DType,
    shape: Shape,
    /** Where the bytes start in the whole file. */
    fileOffset: Long,
    file: MemorySegment
) {
  def byteSize: Long = dtype.byteSize(shape.elementCount)

  def bytes: MemorySegment = file.asSlice(fileOffset, byteSize)

  /** The values, decoded on the CPU onto the heap: for tests and small tensors.
    */
  def decode(): Array[Float] = {
    require(
      shape.elementCount <= Int.MaxValue,
      s"$name has too many values to decode onto the heap"
    )
    dtype.decode(bytes, shape.elementCount.toInt)
  }

  /** The same bytes as a `Tensor` over `storage`, the file's mapping or its
    * registration with the GPU.
    */
  def tensor(storage: Storage): Tensor =
    Tensor(dtype, shape, storage, fileOffset)
}

/** A model file's tensors by name. `unsupported` names tensors whose type the
  * runner cannot decode, with the reason; they only fail when asked for.
  */
trait TensorSource {
  def tensors: Map[String, StoredTensor]
  def unsupported: Map[String, String]

  def apply(name: String): StoredTensor =
    tensors.getOrElse(
      name,
      throw new NoSuchElementException(
        unsupported
          .get(name)
          .fold(s"no tensor $name")(reason => s"$name: $reason")
      )
    )
}

final class FormatException(message: String) extends RuntimeException(message)
