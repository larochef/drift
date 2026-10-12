package drift.runner.models

import drift.runner.ops.*
import drift.runner.tensor.*

import scala.collection.mutable

/** A model's work buffers for sequences of up to `tokens`: the model lists the
  * tensors it needs and their sizes, in the order they are allocated; they are
  * released together, the tensors in that order and then the pages of keys and
  * values.
  */
abstract class WorkBuffers(ops: Ops) {

  /** The longest sequence these buffers hold. */
  def tokens: Int

  private val held = new TensorScope(ops)
  private val pages = mutable.ArrayBuffer.empty[KvCache]

  /** A flat tensor, of which each use takes the `prefix` it needs. */
  protected def flat(dtype: DType, elements: Long): Tensor =
    held.allocate(dtype, Shape.of(elements))

  protected def rows(dtype: DType, count: Long, columns: Long): Tensor =
    held.allocate(dtype, Shape.of(count, columns))

  protected def fromFloats(shape: Shape, values: Array[Float]): Tensor =
    held.fromFloats(shape, values)

  /** Attention's keys and values of `kvHeads` heads, one page for a whole
    * sequence of `length` (pages hold a multiple of 16 tokens).
    */
  protected def page(length: Int, kvHeads: Int, headDimension: Int): KvCache = {
    val cache =
      ops.allocateCache(1, (length + 15) / 16 * 16, kvHeads, headDimension)
    pages += cache
    cache
  }

  /** The page table of a cache of one page. */
  protected def onePage(): Tensor =
    held.keep(ops.fromInts(Shape.of(1), Array(0)))

  def release(): Unit = {
    held.close()
    pages.foreach { cache =>
      ops.release(cache.keys)
      ops.release(cache.values)
    }
    pages.clear()
  }
}

/** A model's buffers kept for its next call, remade for the call's `tokens`
  * when it needs more than they hold.
  */
final class GrownBuffers[B <: WorkBuffers](create: Int => B) {

  private var held = Option.empty[B]

  def apply(tokens: Int): B =
    held.filter(_.tokens >= tokens).getOrElse {
      held.foreach(_.release())
      val created = create(tokens)
      held = Some(created)
      created
    }

  def release(): Unit = held.foreach(_.release())
}
