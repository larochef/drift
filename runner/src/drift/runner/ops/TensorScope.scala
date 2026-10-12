package drift.runner.ops

import drift.runner.tensor.*

import scala.collection.mutable

/** The tensors one piece of work allocates — a step, a frame, a model's buffers
  * — released together when it closes, in the order they were made. A scope
  * that has closed is empty and can be filled again.
  */
final class TensorScope(ops: Ops) extends AutoCloseable {

  private val held = mutable.ArrayBuffer.empty[Tensor]

  /** `tensor`, now the scope's to release. */
  def keep(tensor: Tensor): Tensor = {
    held += tensor
    tensor
  }

  def allocate(dtype: DType, shape: Shape): Tensor =
    keep(ops.allocate(dtype, shape))

  def fromFloats(shape: Shape, values: Array[Float]): Tensor =
    keep(ops.fromFloats(shape, values))

  /** How many tensors it holds. */
  def size: Int = held.size

  def close(): Unit = {
    held.foreach(ops.release)
    held.clear()
  }
}

object TensorScope {

  /** `body` with a scope of its own, closed when it ends, however it ends. */
  def use[A](ops: Ops)(body: TensorScope => A): A = {
    val scope = new TensorScope(ops)
    try body(scope)
    finally scope.close()
  }
}
