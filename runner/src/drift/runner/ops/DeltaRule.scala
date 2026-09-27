package drift.runner.ops

/** A gated-DeltaNet layer's heads (`Ops.gatedDeltaRule`): `keyHeads` query and
  * key heads, `valueHeads` value heads (a multiple), all of `dimension`. With
  * `tiledHeads` value head `h` reads key head `h % keyHeads` (llama.cpp's GGUF
  * order); otherwise `h / (valueHeads / keyHeads)` (transformers').
  */
final case class DeltaRule(
    keyHeads: Int,
    valueHeads: Int,
    dimension: Int,
    tiledHeads: Boolean
) {
  require(
    valueHeads % keyHeads == 0,
    s"$valueHeads value heads over $keyHeads key heads"
  )
  def keyHeadOf(valueHead: Int): Int =
    if (tiledHeads) valueHead % keyHeads
    else valueHead / (valueHeads / keyHeads)
  def qkvWidth: Int = 2 * keyHeads * dimension + valueHeads * dimension
}
