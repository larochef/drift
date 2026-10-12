package drift.runner.ops

import drift.runner.tensor.*

import java.nio.file.Path

/** Memory and transfers: tensors allocated, released, mapped from files,
  * uploaded, read back, copied and converted between storage types.
  */
trait MemoryOps {

  def allocate(dtype: DType, shape: Shape): Tensor

  /** Gives back a tensor `allocate` returned; its memory may be reused. */
  def release(tensor: Tensor): Unit

  /** Maps a weights file where this backend reads it in place: registered with
    * the GPU for `Hip`, a plain mapping for `Cpu`. The formats' readers parse
    * `segment`; `storage` is what tensors over it live in.
    */
  def mapFile(path: Path): MappedWeights

  /** A new F32 tensor holding `values`. */
  def fromFloats(shape: Shape, values: Array[Float]): Tensor

  /** A new I32 tensor holding `values`: positions, token ids. */
  def fromInts(shape: Shape, values: Array[Int]): Tensor

  /** Writes `values` into the start of an I32 tensor that exists already:
    * per-step inputs, without an allocation (a free waits for the GPU).
    */
  def writeInts(tensor: Tensor, values: Array[Int]): Unit

  /** A new tensor holding `bytes` as they are: quantized blocks, say. */
  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor

  /** An F32 tensor's values, back on the heap: for tests and small results. */
  def toFloats(tensor: Tensor): Array[Float]

  /** An I32 tensor's values, back on the heap: for tests. */
  def toInts(tensor: Tensor): Array[Int]

  /** Between F32 and F16 or BF16, rounding to nearest even; fp8 E4M3 to F32; a
    * rotated linear (`rotated`) to the BF16 weight it holds.
    */
  def convert(x: Tensor, out: Tensor): Unit

  /** Sets every byte of a tensor to zero: a fresh recurrent state. */
  def zero(tensor: Tensor): Unit

  /** Copies `from` into `to`, byte for byte (same type and size): a rollback, a
    * kept row. Ordered with the kernels; the host does not wait.
    */
  def copy(from: Tensor, to: Tensor): Unit
}

/** The rules of `MemoryOps`.
  */
private[ops] trait MemoryChecks extends CommonChecks {

  def checkCopy(from: Tensor, to: Tensor): Unit =
    require(
      from.dtype == to.dtype && from.byteSize == to.byteSize,
      s"copy: ${from.dtype} ${from.shape} into ${to.dtype} ${to.shape}"
    )

  /** The conversions `convert` knows, by (from, to). */
  val Conversions: Set[(DType, DType)] = Set(
    DType.F32 -> DType.F16,
    DType.F32 -> DType.BF16,
    DType.F16 -> DType.F32,
    DType.BF16 -> DType.F32,
    DType.F8E4M3 -> DType.F32,
    ConvRot.Codes -> DType.BF16,
    ConvRot.Nibbles -> DType.BF16
  )

  def checkConvert(x: Tensor, out: Tensor): Unit = {
    requireSameShape("convert", x, out)
    require(
      Conversions.contains(x.dtype -> out.dtype),
      s"convert: no conversion from ${x.dtype} to ${out.dtype}"
    )
  }

  def checkBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Unit =
    require(
      bytes.length == dtype.byteSize(shape.elementCount),
      s"${bytes.length} bytes for $dtype $shape"
    )
}
