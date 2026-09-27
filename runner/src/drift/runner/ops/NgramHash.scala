package drift.runner.ops

/** How Qwen 3.8 Flash Next's n-gram embedding picks its rows (`Ops.ngramRows`):
  * each token with its `ngramSize − 1` predecessors (cut at `eos`), hashed as
  * `(t0·m0) ^ (t1·m1) [^ …]` in wrapping 64-bit arithmetic, then
  * `mod sizes(h) + offsets(h)` for each of the
  * `(ngramSize − 1) × headsPerNgram` heads (the bigram heads first).
  */
final case class NgramHash(
    ngramSize: Int,
    headsPerNgram: Int,
    eos: Int,
    multipliers: Vector[Long],
    sizes: Vector[Long],
    offsets: Vector[Long]
) {
  require(ngramSize >= 2 && ngramSize <= 8, s"n-grams of $ngramSize")
  require(multipliers.size == ngramSize, s"${multipliers.size} multipliers")
  def heads: Int = (ngramSize - 1) * headsPerNgram
  require(
    sizes.size == heads && offsets.size == heads,
    s"${sizes.size} sizes and ${offsets.size} offsets for $heads heads"
  )

  /** The predecessors a sequence keeps between calls. */
  def kept: Int = ngramSize - 1
}
