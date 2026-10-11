package drift.runner.models

/** A timestep as the vector a model reads it by: cosines and sines of the value
  * times `half` frequencies falling from 1 to 1 / `maxPeriod`.
  *
  * The families differ in what they multiply and in how wide: `value` comes
  * already scaled (σ × 1000 where the model was trained on it), the sines come
  * first or the cosines, and the frequencies are kept in single precision or
  * not — which decides the last bit of the angle, and a model is compared with
  * its reference to the bit.
  */
object TimestepSinusoid {

  def apply(
      value: Float,
      half: Int,
      maxPeriod: Double = 10000,
      sinesFirst: Boolean = false,
      singleFrequencies: Boolean = false
  ): Array[Float] =
    Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(maxPeriod) * (i % half) / half)
      val angle: Double =
        if (singleFrequencies) (value * frequency.toFloat).toDouble
        else value * frequency
      val cosine = (i < half) != sinesFirst
      (if (cosine) math.cos(angle) else math.sin(angle)).toFloat
    }
}
