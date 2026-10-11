package drift.runner.tensor

import java.util.SplittableRandom

/** Random draws the pipelines and the models share. */
object Noise {

  /** A standard normal value: Box–Muller on two uniforms in (0, 1]. */
  def gaussian(random: SplittableRandom): Float = {
    val u = 1 - random.nextDouble()
    val v = random.nextDouble()
    (math.sqrt(-2 * math.log(u)) * math.cos(2 * math.Pi * v)).toFloat
  }
}
