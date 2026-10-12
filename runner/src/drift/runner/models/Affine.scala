package drift.runner.models

import drift.runner.lora.LoraUpdates
import drift.runner.ops.*
import drift.runner.tensor.*

/** A linear layer and its bias: `weight` (`[outputs, inputs]`) and `bias` (F32,
  * `[outputs]`). `site` is its name in the file, a LoRA's target, where the
  * model takes LoRAs.
  *
  * Each model loads its own (the weight as the file holds it or remade, the
  * bias as floats); what is shared is the layer and the way it runs.
  */
final case class Affine(weight: Tensor, bias: Tensor, site: String = "") {

  /** `out = x · weightᵀ + bias`; with `updates`, the active LoRAs' of its site
    * are added before the bias.
    */
  def apply(
      ops: Ops,
      x: Tensor,
      out: Tensor,
      updates: Option[LoraUpdates] = None
  ): Unit = {
    ops.linear(x, weight, out)
    updates.foreach(_(x, site, out))
    ops.addRow(out, bias, out)
  }

  /** As above, for a model whose LoRAs are always there. */
  def apply(ops: Ops, x: Tensor, out: Tensor, updates: LoraUpdates): Unit =
    apply(ops, x, out, Some(updates))
}
