package drift.runner.diffusion

/** A flow-matching schedule (diffusers' `FlowMatchEulerDiscreteScheduler` with
  * a fixed shift): `steps` noise levels from `linspace(1, 1/steps, steps)`,
  * each shifted by `σ' = e^μ / (e^μ + 1/σ − 1)`, then 0. The model sees σ as
  * its timestep; an Euler step moves `x += (σ_next − σ) × v`.
  */
object FlowSchedule {

  /** img2img's first step run (diffusers' `get_timesteps`): `steps − ⌊steps ×
    * strength⌋`, the floor taken before the subtraction, so 4 steps at 0.4 run
    * one.
    */
  def firstStep(steps: Int, strength: Float): Int =
    steps - math.min((steps * strength).toInt, steps)

  def sigmas(steps: Int, shift: Double): IndexedSeq[Float] = {
    require(steps >= 1, s"$steps steps")
    val e = math.exp(shift)
    val levels = (0 until steps).map { i =>
      val sigma =
        if (steps == 1) 1.0 else 1.0 - i * (1.0 - 1.0 / steps) / (steps - 1)
      (e / (e + (1 / sigma - 1))).toFloat
    }
    levels :+ 0f
  }

  /** FLUX.2's μ for `imageTokens` (the target's packed tokens) and `steps`
    * (diffusers' `compute_empirical_mu`): a line in the tokens fitted at 10 and
    * 200 steps, interpolated in the steps; past 4300 tokens the 200-step line
    * alone.
    */
  def flux2Shift(imageTokens: Int, steps: Int): Double = {
    val m200 = 0.00016927 * imageTokens + 0.45666666
    if (imageTokens > 4300) m200
    else {
      val m10 = 8.73809524e-5 * imageTokens + 1.89833333
      val a = (m200 - m10) / 190
      a * steps + (m200 - 200 * a)
    }
  }

  /** Qwen Image 2.1's μ for `imageTokens` (its scheduler's configuration): the
    * line through 256 tokens at 0.5 and 8192 at 0.9.
    */
  def qwenImage21Shift(imageTokens: Int): Double =
    0.5 + (imageTokens - 256) * (0.9 - 0.5) / (8192 - 256)

  /** `sigmas` (as `sigmas` gives them, ending in 0) stretched so that the last
    * level before 0 is `terminal` (diffusers' `shift_terminal`): `σ' = 1 − (1 −
    * σ) × (1 − terminal) / (1 − σ_last)`. A single level of 1 stays as it is.
    */
  def stretched(
      sigmas: IndexedSeq[Float],
      terminal: Double
  ): IndexedSeq[Float] = {
    val levels = sigmas.init
    val last = 1.0 - levels.last
    if (last == 0) sigmas
    else
      levels.map(s => (1 - (1 - s) * (1 - terminal) / last).toFloat) :+ 0f
  }
}
