package drift.runner.diffusion

/** PyTorch's CPU generator (`torch.Generator().manual_seed(seed)`): its
  * Mersenne twister and `torch.randn`'s two paths, so a seed draws the noise
  * diffusers and ComfyUI draw (sd-cpp's `rng_mt19937.hpp`, after ATen's
  * `MT19937RNGEngine`, `normal_fill` and `normal_distribution`).
  *   - 16 values or more: one float uniform per value (24 bits), then
  *     Box–Muller over blocks of 16 (value j with value j + 8); a tail short of
  *     16 draws 16 fresh uniforms for the last 16 values.
  *   - Fewer: double Box–Muller from 53-bit uniforms, the pair's second value
  *     kept for the next draw.
  */
final class TorchRandom(seed: Long) {

  private val N = 624
  private val M = 397
  private val state = new Array[Int](N)
  private var left = 1
  private var next = 0
  private var nextGauss = Option.empty[Double]

  state(0) = seed.toInt
  (1 until N).foreach { j =>
    val previous = state(j - 1)
    state(j) = 1812433253 * (previous ^ (previous >>> 30)) + j
  }

  private def twist(u: Int, v: Int): Int = {
    val mixed = (u & 0x80000000) | (v & 0x7fffffff)
    (mixed >>> 1) ^ (if ((v & 1) != 0) 0x9908b0df else 0)
  }

  private def nextState(): Unit = {
    left = N
    next = 0
    (0 until N - M).foreach(j =>
      state(j) = state(j + M) ^ twist(state(j), state(j + 1))
    )
    (N - M until N - 1).foreach(j =>
      state(j) = state(j + M - N) ^ twist(state(j), state(j + 1))
    )
    state(N - 1) = state(M - 1) ^ twist(state(N - 1), state(0))
  }

  /** The next 32 random bits, unsigned in a Long. */
  private def nextUnsigned(): Long = {
    left -= 1
    if (left == 0) nextState()
    var y = state(next)
    next += 1
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18
    y.toLong & 0xffffffffL
  }

  private def floatUniform(): Float =
    (nextUnsigned() & ((1L << 24) - 1)).toFloat * (1f / (1 << 24))

  private def doubleUniform(): Double = {
    val bits = (nextUnsigned() << 32) | nextUnsigned()
    (bits & ((1L << 53) - 1)).toDouble * (1.0 / (1L << 53))
  }

  private def normalDouble(): Double = nextGauss match {
    case Some(value) =>
      nextGauss = None
      value
    case None =>
      val (u1, u2) = (doubleUniform(), doubleUniform())
      val r = math.sqrt(-2.0 * math.log1p(-u2))
      val theta = 2.0 * math.Pi * u1
      nextGauss = Some(r * math.sin(theta))
      r * math.cos(theta)
  }

  /** Box–Muller over `data(from until from + 16)` of uniforms, in place. */
  private def fill16(data: Array[Float], from: Int): Unit =
    (0 until 8).foreach { j =>
      val u1 = 1f - data(from + j)
      val u2 = data(from + j + 8)
      val radius = math.sqrt(-2.0 * math.log(u1)).toFloat
      val theta = (2.0 * math.Pi * u2).toFloat
      data(from + j) = radius * math.cos(theta).toFloat
      data(from + j + 8) = radius * math.sin(theta).toFloat
    }

  /** `torch.randn(count, generator=…)`, values in the tensor's order. */
  def normal(count: Int): Array[Float] = {
    val data = new Array[Float](count)
    if (count >= 16) {
      data.indices.foreach(i => data(i) = floatUniform())
      (0 to count - 16 by 16).foreach(fill16(data, _))
      if (count % 16 != 0) {
        (count - 16 until count).foreach(i => data(i) = floatUniform())
        fill16(data, count - 16)
      }
    } else data.indices.foreach(i => data(i) = normalDouble().toFloat)
    data
  }
}
