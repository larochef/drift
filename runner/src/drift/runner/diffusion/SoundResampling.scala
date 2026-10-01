package drift.runner.diffusion

/** torchaudio's `functional.resample` (the Hann-windowed sinc, 6 zero
  * crossings, a cut at 99% of the lower Nyquist), which ComfyUI runs on a sound
  * before its audio VAE: the rates reduced by their common divisor to `from` →
  * `to`, output sample `j × to + i` the kernel `i` over the input from
  * `j × from − width` on, the input zero-padded.
  */
object SoundResampling {

  private val ZeroCrossings = 6
  private val Rolloff = 0.99

  /** One channel's `samples` from `fromRate` to `toRate`: `⌈to × length /
    * from⌉` samples.
    */
  def resampled(
      samples: Array[Float],
      fromRate: Int,
      toRate: Int
  ): Array[Float] =
    if (fromRate == toRate) samples
    else {
      val divisor = BigInt(fromRate).gcd(BigInt(toRate)).toInt
      val (from, to) = (fromRate / divisor, toRate / divisor)
      val base = math.min(from, to) * Rolloff
      val width = math.ceil(ZeroCrossings * from / base).toInt
      val taps = 2 * width + from
      val kernels = Array.tabulate(to, taps) { (i, k) =>
        val t = math.max(
          -ZeroCrossings.toDouble,
          math.min(
            ZeroCrossings.toDouble,
            (-i.toDouble / to + (k - width).toDouble / from) * base
          )
        )
        val window = math.pow(math.cos(t * math.Pi / ZeroCrossings / 2), 2)
        val x = t * math.Pi
        val sinc = if (x == 0) 1.0 else math.sin(x) / x
        (sinc * window * base / from).toFloat
      }
      val length = samples.length
      val frames = (length + 2 * width + from - taps) / from + 1
      val target = math.ceil(to.toDouble * length / from).toInt
      val out = new Array[Float](math.min(target, frames * to))
      out.indices.foreach { index =>
        val (j, i) = (index / to, index % to)
        var sum = 0.0
        var k = 0
        while (k < taps) {
          val at = j * from + k - width
          if (at >= 0 && at < length) sum += kernels(i)(k) * samples(at)
          k += 1
        }
        out(index) = sum.toFloat
      }
      out
    }
}
