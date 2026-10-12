package drift.runner.models

import drift.runner.tensor.Bf16

/** A timestep as the vector a model reads it by: cosines and sines of the value
  * times `half` frequencies falling from 1 to 1 / `maxPeriod`.
  *
  * The families differ in what they multiply and in how wide: `value` comes
  * already scaled (σ × 1000 where the model was trained on it), the sines come
  * first or the cosines, and the frequencies are kept in single precision or
  * not — which decides the last bit of the angle, and a model is compared with
  * its reference to the bit. A model released in BF16 rounds on the way: its
  * frequencies (`bf16Frequencies`), what it reads (`bf16`); and some multiply
  * the angle once it is formed (`after`). Both are single precision's.
  */
object TimestepSinusoid {

  def apply(
      value: Double,
      half: Int,
      maxPeriod: Double = 10000,
      sinesFirst: Boolean = false,
      singleFrequencies: Boolean = false,
      bf16Frequencies: Boolean = false,
      after: Float = 1f,
      bf16: Boolean = false
  ): Array[Float] =
    Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(maxPeriod) * (i % half) / half)
      val angle: Double =
        if (singleFrequencies) {
          val single = frequency.toFloat
          val kept = if (bf16Frequencies) Bf16.round(single) else single
          (value.toFloat * kept * after).toDouble
        } else value * frequency
      val cosine = (i < half) != sinesFirst
      val read = (if (cosine) math.cos(angle) else math.sin(angle)).toFloat
      if (bf16) Bf16.round(read) else read
    }
}
