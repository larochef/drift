package drift.runner.diffusion

import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Tensor}

/** The sampling loop the flow models share: from a noise level to the next, the
  * latents move along the velocity the model gives — guided or not, under a
  * mask or not, drawn for a preview or not. A pipeline says what its model's
  * velocity is; the loop is written here once.
  */
object FlowSampler {

  /** Classifier-free guidance: its scale, and the velocity of the prompt to
    * move away from, written into the tensor given for step `i`.
    */
  final case class Guidance(scale: Float, velocityAt: (Int, Tensor) => Unit)

  /** An img2img's start: `x`, the noise, becomes σ × noise + (1 − σ) × the
    * picture's latents, which a mask remembers first. `encoded` is used up —
    * scaled in place — and stays the caller's to release.
    */
  def started(
      ops: Ops,
      x: Tensor,
      encoded: Tensor,
      sigma: Float,
      inpainting: Option[Inpainting]
  ): Unit = {
    inpainting.foreach(_.remember(encoded))
    ops.scale(x, sigma, x)
    ops.scale(encoded, 1 - sigma, encoded)
    ops.add(x, encoded, x)
  }

  /** An Euler step: x moves by the velocity times the change of level. */
  def euler(ops: Ops, sigmas: IndexedSeq[Float])(
      i: Int,
      x: Tensor,
      velocity: Tensor
  ): Unit = {
    ops.scale(velocity, sigmas(i + 1) - sigmas(i), velocity)
    ops.add(x, velocity, x)
  }

  /** Runs the steps `first` until `last` over `x`, in place.
    *
    *   - `velocityAt(i, into)` writes the model's velocity at step `i`.
    *   - With `guidance`, the velocity is the unconditional one plus the scale
    *     times what the prompt adds to it.
    *   - `onVelocity(step, steps, velocity, sigma)` is given each step's
    *     velocity before it is used, for a preview: `step` counts from 1 among
    *     the `steps` run.
    *   - `move` is the step itself, Euler's unless said.
    *   - Under a mask, what it keeps is put back at each new level.
    *   - `progress(done, steps)` is called before the first step and after
    *     each.
    */
  def run(
      ops: Ops,
      x: Tensor,
      sigmas: IndexedSeq[Float],
      first: Int,
      last: Int,
      velocityAt: (Int, Tensor) => Unit,
      guidance: Option[Guidance],
      inpainting: Option[Inpainting],
      onVelocity: (Int, Int, Tensor, Float) => Unit,
      progress: (Int, Int) => Unit,
      move: Option[(Int, Tensor, Tensor) => Unit] = None
  ): Unit = {
    val velocity = ops.allocate(DType.F32, x.shape)
    val other = guidance.map(_ => ops.allocate(DType.F32, x.shape))
    val step = move.getOrElse(euler(ops, sigmas))
    try {
      val steps = last - first
      progress(0, steps)
      (first until last).foreach { i =>
        velocityAt(i, velocity)
        for {
          guided <- guidance
          v <- other
        } {
          guided.velocityAt(i, v)
          // v = uncond + scale × (cond − uncond)
          ops.scale(v, 1 - guided.scale, v)
          ops.scale(velocity, guided.scale, velocity)
          ops.add(velocity, v, velocity)
        }
        onVelocity(i + 1 - first, steps, velocity, sigmas(i))
        step(i, x, velocity)
        inpainting.foreach(_.restore(x, sigmas(i + 1)))
        progress(i + 1 - first, steps)
      }
    } finally {
      ops.release(velocity)
      other.foreach(ops.release)
    }
  }
}
