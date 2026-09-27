package drift.runner.models

import drift.runner.state.Sequence
import drift.runner.tensor.Tensor

/** A decoder-only language model: tokens in, next-token logits out, what it
  * must remember of a sequence kept in that sequence's `Sequence`.
  */
trait CausalModel extends AutoCloseable {

  def vocabulary: Int

  /** A new, empty sequence of up to `context` tokens. */
  def newSequence(context: Int, pageSize: Int): Sequence

  /** Runs `ids` in slots `start` on, remembering them in `sequence` (which must
    * hold room for them; its `positions` say where each slot turns). The
    * `givenRows` replace some tokens' embeddings (an image's, for a model that
    * sees). The logits are those of the last token, or of every token with
    * `allLogits`: F32 `[1 or tokens, vocabulary]`, valid until the next call.
    */
  def forward(
      ids: Array[Int],
      start: Int,
      sequence: Sequence,
      allLogits: Boolean,
      givenRows: Seq[GivenRows] = Nil
  ): Tensor

  /** `forward` with every token's logits, keeping every recurrent state after
    * each token in `sequence.history`: a speculative run, which
    * `sequence.restore` cuts back to the tokens accepted.
    */
  def verify(ids: Array[Int], start: Int, sequence: Sequence): Tensor
}

/** Embeddings given instead of looked up: `rows` (F32 `[n, hidden]`) for a
  * run's tokens `at until at + n`, an image's tokens from its vision tower.
  */
final case class GivenRows(at: Int, rows: Tensor) {
  def count: Int = rows.shape.dimensions.head.toInt
}

/** Images a language model reads as it encodes: every token's mRoPE positions
  * (`[3, tokens]`, axis-major), the embeddings of the images' tokens, and
  * Qwen3-VL's deepstack rows, `deepstack(k)` added to the output of layer `k`
  * (0-based).
  */
final case class SeenImages(
    positions: Array[Int],
    rows: Seq[GivenRows],
    deepstack: Seq[Seq[GivenRows]]
)

/** A model that drafts the tokens after the next itself (multi-token
  * prediction), from the hidden states its forwards leave.
  */
trait Drafting extends CausalModel {

  /** Whether these weights carry the layers that draft. */
  def drafts: Boolean

  def hiddenSize: Int

  /** The last forward's or verify's final-norm hidden states, F32 `[tokens,
    * hiddenSize]`, valid until the next call.
    */
  def hidden: Tensor

  /** Runs the drafting layer on entries (`ids(i)`, `hidden(i)`) at positions
    * `start + i`: each token with the hidden state of the one before it,
    * remembered in `sequence`. Returns the last entry's logits for the token
    * after it, F32 `[1, vocabulary]`, valid until the next call.
    */
  def draft(
      ids: Array[Int],
      hidden: Tensor,
      start: Int,
      sequence: Sequence
  ): Tensor

  /** The last draft's output hidden state `[1, hiddenSize]`, which a chained
    * draft pairs with the token drafted.
    */
  def draftHidden: Tensor
}
