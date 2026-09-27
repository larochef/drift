package drift.runner.ops

/** How pairs of a head's values are formed: `Neox` pairs value `i` with
  * `i + rotary/2` (transformers' `rotate_half`), `Interleaved` pairs `2i` with
  * `2i + 1` (GPT-J).
  */
enum RopeLayout(val code: Int) {
  case Neox extends RopeLayout(0)
  case Interleaved extends RopeLayout(1)
}

/** Which position each pair turns by. `Single`: one position per token. mRoPE
  * takes three per token (temporal, height, width): `Contiguous` gives the
  * first `temporal` pairs the first, the next `height` the second, the rest the
  * third; `Interleaved` (Qwen3-VL and on) deals them out in turn — pair `i`
  * takes height when `i % 3 == 1 && i < 3 × height`, width when
  * `i % 3 == 2 && i < 3 × width`, temporal otherwise. `Axes` (Krea 2's three,
  * FLUX.2's four) splits the pairs in turn among up to four positions per
  * token, `pairs(a)` to axis `a`, and each axis turns on its own frequencies:
  * its pair `j` of `n` by `theta^(−j / n)`.
  */
enum RopeSections(val code: Int) {
  case Single extends RopeSections(0)
  case Contiguous(temporal: Int, height: Int, width: Int)
      extends RopeSections(1)
  case Interleaved(temporal: Int, height: Int, width: Int)
      extends RopeSections(2)
  case Axes(pairs: Seq[Int]) extends RopeSections(3)

  def positionsPerToken: Int = this match {
    case Single      => 1
    case Axes(pairs) => pairs.size
    case _           => 3
  }
}

/** A rotary embedding: the first `rotaryDimensions` values of each head turn
  * (partial rotary leaves the rest), pair `i` by
  * `position × theta^(−2i / rotaryDimensions)`.
  */
final case class Rope(
    theta: Float,
    rotaryDimensions: Int,
    layout: RopeLayout,
    sections: RopeSections
)
