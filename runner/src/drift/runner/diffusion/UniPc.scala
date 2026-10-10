package drift.runner.diffusion

/** A weighted sum of a UniPC update's four tensors: the sample, the clean image
  * the model gave one step back (`latest`) and two back (`earlier`), and the
  * one it gives now (`current`).
  */
final case class UniPcBlend(
    sample: Double,
    latest: Double,
    earlier: Double,
    current: Double
)

/** UniPC for flow matching (Wan's `FlowUniPCMultistepScheduler`, which HiDream
  * O1's full checkpoint samples with): order 2, `bh2`, on the predicted clean
  * image, the order lowered at both ends (`lower_order_final`), the corrector
  * on every step but the first. A step `i` runs the model at `sigmas(i)`,
  * corrects the sample it was given with what the model now says (`corrector`),
  * then predicts the sample at `sigmas(i + 1)` (`predictor`); the last step
  * lands on the clean image itself.
  *
  * Every update is a weighted sum, so the weights are all there is here: the
  * pipeline applies them to its tensors. `first` is the step a run starts at
  * (img2img), which has no history before it.
  */
final class UniPc(sigmas: IndexedSeq[Float], first: Int = 0) {
  require(sigmas.size >= 2, "UniPC: no step")
  require(sigmas.last == 0f, "UniPC: the noise levels end at 0")

  private val steps = sigmas.size - 1

  /** λ = log α − log σ of a level, α = 1 − σ; a level of 1 has no λ, and is
    * read just under it.
    */
  private def lambda(sigma: Double): Double =
    math.log(math.max(1 - sigma, UniPc.MinimumAlpha)) - math.log(sigma)

  /** The predictor's order at `step`: 2, but 1 on the run's first step and on
    * its last.
    */
  def order(step: Int): Int =
    math.min(math.min(UniPc.Order, steps - step), step - first + 1)

  /** `step`'s corrected sample, from the sample the model was run on at the
    * step before (`sample`), the clean images of the steps before (`latest`,
    * `earlier`) and this step's (`current`); none on the run's first step.
    */
  def corrector(step: Int): Option[UniPcBlend] =
    Option.when(step > first) {
      val (sigma, before) = (sigmas(step).toDouble, sigmas(step - 1).toDouble)
      val alpha = 1 - sigma
      val h = lambda(sigma) - lambda(before)
      // bh2: B(h) = e^(−h) − 1, which is hφ₁(h) too
      val phi = math.expm1(-h)
      if (order(step - 1) == 1)
        UniPcBlend(
          sample = sigma / before,
          latest = -alpha * phi + alpha * phi * 0.5,
          earlier = 0,
          current = -alpha * phi * 0.5
        )
      else {
        val r = (lambda(sigmas(step - 2)) - lambda(before)) / h
        val b1 = (phi / -h - 1) / phi
        val b2 = ((phi / -h - 1) / -h - 0.5) * 2 / phi
        // ρ₁ + ρ₂ = b₁, r ρ₁ + ρ₂ = b₂
        val rho1 = (b1 - b2) / (1 - r)
        val rho2 = b1 - rho1
        UniPcBlend(
          sample = sigma / before,
          latest = -alpha * phi + alpha * phi * (rho1 / r + rho2),
          earlier = -alpha * phi * rho1 / r,
          current = -alpha * phi * rho2
        )
      }
    }

  /** The sample at `sigmas(step + 1)`, from `step`'s (corrected) sample, its
    * clean image (`latest`) and the one before (`earlier`); `current` is not
    * read.
    */
  def predictor(step: Int): UniPcBlend = {
    val (next, sigma) = (sigmas(step + 1).toDouble, sigmas(step).toDouble)
    if (next == 0) UniPcBlend(0, 1, 0, 0)
    else {
      val alpha = 1 - next
      val h = lambda(next) - lambda(sigma)
      val phi = math.expm1(-h)
      if (order(step) == 1) UniPcBlend(next / sigma, -alpha * phi, 0, 0)
      else {
        val r = (lambda(sigmas(step - 1)) - lambda(sigma)) / h
        // order 2's simplified ρ = ½
        UniPcBlend(
          sample = next / sigma,
          latest = -alpha * phi + alpha * phi * 0.5 / r,
          earlier = -alpha * phi * 0.5 / r,
          current = 0
        )
      }
    }
  }
}

object UniPc {

  /** The solver's order (`solver_order`). */
  val Order = 2

  /** α's floor under the logarithm: a level of exactly 1 (a custom list's
    * first) has none.
    */
  private val MinimumAlpha = 1e-4

  /** The scheduler's `steps` noise levels for a `shift`, then 0: evenly spaced
    * from its highest trained level — 0.999 shifted — down, then shifted again
    * (`set_timesteps` shifts levels the constructor already shifted).
    */
  def sigmas(steps: Int, shift: Double): IndexedSeq[Float] = {
    require(steps >= 1, s"$steps steps")
    def shifted(sigma: Double) = shift * sigma / (1 + (shift - 1) * sigma)
    // float32, as the scheduler keeps its highest level
    val highest = shifted(1.0 - 1.0 / 1000).toFloat.toDouble
    (0 until steps).map(i =>
      shifted(highest * (1 - i.toDouble / steps)).toFloat
    ) :+ 0f
  }
}
