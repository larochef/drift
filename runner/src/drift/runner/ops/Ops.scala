package drift.runner.ops

import drift.runner.tensor.*

/** The operations blocks and models are written against (`specs/42`). Two
  * backends implement them: `CpuOps`, the slow and obviously correct reference,
  * and `HipOps`, the kernels. The same model code runs on both, which is how
  * the kernels are tested.
  *
  * Tensors belong to the backend that allocated them, or are registered
  * mappings both can read. `out` arguments are written, never read, and may be
  * the same tensor as an input for elementwise operations. Activations are F32;
  * weights may be any storage type.
  */
trait Ops
    extends AutoCloseable
    with MemoryOps
    with ElementwiseOps
    with NormalisationOps
    with LinearOps
    with ExpertsOps
    with AttentionOps
    with ConvolutionOps
    with LayoutOps
    with SequenceOps {

  def name: String

  /** Called as the backend is given each piece of work (a kernel launched): a
    * step's are counted to say how far a later one has got. The server's to set
    * around a generation.
    */
  var onWork: () => Unit = () => ()
}

/** The shape and dtype rules both backends enforce, so a mistake fails the same
  * way on either.
  */
object Ops
    extends CommonChecks
    with MemoryChecks
    with ElementwiseChecks
    with NormalisationChecks
    with LinearChecks
    with ExpertsChecks
    with AttentionChecks
    with ConvolutionChecks
    with LayoutChecks
    with SequenceChecks

/** The rules several groups of operations share.
  */
private[ops] trait CommonChecks {

  def requireF32(operation: String, tensors: Tensor*): Unit =
    tensors.foreach { tensor =>
      require(
        tensor.dtype == DType.F32,
        s"$operation takes F32, not ${tensor.dtype}"
      )
    }

  def requireSameShape(operation: String, tensors: Tensor*): Unit =
    require(
      tensors.map(_.shape).distinct.size == 1,
      s"$operation needs equal shapes, got ${tensors.map(_.shape).mkString(", ")}"
    )

  /** Same-shape F32 operands. */
  def checkElementwise(operation: String, tensors: Tensor*): Unit = {
    requireF32(operation, tensors*)
    requireSameShape(operation, tensors*)
  }

  /** Returns the row count. */
  def checkRowWise(
      operation: String,
      x: Tensor,
      out: Tensor,
      rowVectors: Tensor*
  ): Long = {
    checkElementwise(operation, x, out)
    require(x.shape.rank >= 1, s"$operation needs at least one dimension")
    rowVectors.foreach { vector =>
      requireF32(operation, vector)
      require(
        vector.shape == Shape.of(x.shape.last),
        s"$operation: ${vector.shape} for rows of ${x.shape.last}"
      )
    }
    x.shape.elementCount / x.shape.last
  }
}
