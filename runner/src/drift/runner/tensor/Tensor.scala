package drift.runner.tensor

import java.lang.foreign.MemorySegment

final case class Shape(dimensions: Vector[Long]) {
  require(dimensions.forall(_ >= 0), s"negative dimension in $dimensions")

  def rank: Int = dimensions.size
  def elementCount: Long = dimensions.product
  def last: Long = dimensions.last

  override def toString: String = dimensions.mkString("[", "×", "]")
}

object Shape {
  def of(dimensions: Long*): Shape = Shape(dimensions.toVector)
}

/** Where a tensor's bytes live. Nothing lives on the JVM heap. */
enum Storage {

  /** Host memory only: what the `Cpu` backend computes on. */
  case Host(segment: MemorySegment)

  /** Device memory only (`hipMalloc`); `pointer` is never dereferenced. */
  case Device(pointer: MemorySegment, length: Long)

  /** The same pages seen by both: a registered mapping (`RegisteredFile`). */
  case Registered(host: MemorySegment, device: MemorySegment)

  def byteSize: Long = this match {
    case Host(segment)       => segment.byteSize()
    case Device(_, byteSize) => byteSize
    case Registered(host, _) => host.byteSize()
  }
}

/** A descriptor over off-heap bytes: contiguous, row-major for now. */
final case class Tensor(
    dtype: DType,
    shape: Shape,
    storage: Storage,
    byteOffset: Long
) {
  def byteSize: Long = dtype.byteSize(shape.elementCount)

  require(
    byteOffset >= 0 && byteOffset + byteSize <= storage.byteSize,
    s"$shape $dtype at $byteOffset overruns its storage of ${storage.byteSize} bytes"
  )

  /** The same bytes seen with another shape of as many elements. */
  def view(dimensions: Long*): Tensor = {
    val reshaped = Shape(dimensions.toVector)
    require(
      reshaped.elementCount == shape.elementCount,
      s"a view of $shape as $reshaped"
    )
    copy(shape = reshaped)
  }

  /** The first elements of these bytes seen with a shape of no more of them: a
    * workspace buffer holding a smaller operand.
    */
  def prefix(dimensions: Long*): Tensor = {
    val reshaped = Shape(dimensions.toVector)
    require(
      reshaped.elementCount <= shape.elementCount,
      s"a prefix of $shape as $reshaped"
    )
    copy(shape = reshaped)
  }

  /** Rows `from until from + count` of the first dimension, without a copy. */
  def rows(from: Long, count: Long): Tensor = {
    val first = shape.dimensions.head
    require(
      from >= 0 && count >= 0 && from + count <= first,
      s"rows $from..${from + count} of $shape"
    )
    val rowElements = shape.elementCount / math.max(first, 1)
    copy(
      shape = Shape(count +: shape.dimensions.tail),
      byteOffset = byteOffset + dtype.byteSize(from * rowElements)
    )
  }
}
