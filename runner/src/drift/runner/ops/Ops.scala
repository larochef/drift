package drift.runner.ops

import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path

/** The operations blocks and models are written against (`specs/42`). Two
  * backends implement them: `CpuOps`, the slow and obviously correct reference,
  * and `HipOps`, the kernels. The same model code runs on both, which is how
  * the kernels are tested.
  *
  * Tensors belong to the backend that allocated them, or are registered
  * mappings both can read. `out` arguments are written, never read, and may be
  * the same tensor as an input for elementwise operations. Activations are F32;
  * weights may be any storage type.
  */
trait Ops extends AutoCloseable {

  def name: String

  def allocate(dtype: DType, shape: Shape): Tensor

  /** Gives back a tensor `allocate` returned; its memory may be reused. */
  def release(tensor: Tensor): Unit

  /** Maps a weights file where this backend reads it in place: registered with
    * the GPU for `Hip`, a plain mapping for `Cpu`. The formats' readers parse
    * `segment`; `storage` is what tensors over it live in.
    */
  def mapFile(path: Path): MappedWeights

  /** A new F32 tensor holding `values`. */
  def fromFloats(shape: Shape, values: Array[Float]): Tensor

  /** A new I32 tensor holding `values`: positions, token ids. */
  def fromInts(shape: Shape, values: Array[Int]): Tensor

  /** Writes `values` into the start of an I32 tensor that exists already:
    * per-step inputs, without an allocation (a free waits for the GPU).
    */
  def writeInts(tensor: Tensor, values: Array[Int]): Unit

  /** A new tensor holding `bytes` as they are: quantized blocks, say. */
  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor

  /** An F32 tensor's values, back on the heap: for tests and small results. */
  def toFloats(tensor: Tensor): Array[Float]

  /** An I32 tensor's values, back on the heap: for tests. */
  def toInts(tensor: Tensor): Array[Int]

  /** The index of an F32 tensor's largest value, the first among equals: greedy
    * decoding without bringing the logits back.
    */
  def argmax(values: Tensor): Int

  /** Each row's `count` largest values (F32 `[rows, width]`, at most
    * `Ops.MaximumCandidates`), largest first, the lower index first among
    * equals: what sampling draws from without bringing the logits back.
    */
  def candidates(values: Tensor, count: Int): Seq[Candidates]

  /** The largest magnitude among `x`'s values (NaNs skipped), on the host. */
  def maxAbs(x: Tensor): Float

  /** `out = a + b`, elementwise. */
  def add(a: Tensor, b: Tensor, out: Tensor): Unit

  /** `out = a × b`, elementwise. */
  def mul(a: Tensor, b: Tensor, out: Tensor): Unit

  /** `out = x × factor`. */
  def scale(x: Tensor, factor: Float, out: Tensor): Unit

  /** `out[t] = table[ids[t]]`: rows of an embedding table in any storage type,
    * as F32. `ids` is I32 `[tokens]`, `out` `[tokens, D]`.
    */
  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit

  /** `out = x + row`, `row` added to every row of `x` (a bias). */
  def addRow(x: Tensor, row: Tensor, out: Tensor): Unit

  def activation(kind: Activation, x: Tensor, out: Tensor): Unit =
    scaledActivation(kind, 1f, x, out)

  /** `out = kind(inputScale × x)`, elementwise. */
  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit

  /** `out = kind(gate) × up`: SwiGLU with `Silu`, GeGLU with a GELU. */
  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit

  /** Between F32 and F16 or BF16, rounding to nearest even; fp8 E4M3 to F32. */
  def convert(x: Tensor, out: Tensor): Unit

  /** RMS norm over the last dimension:
    * `out = x / sqrt(mean(x²) + epsilon) × (weight + weightOffset)`;
    * `weightOffset` is 1 for Gemma, whose norm weights are stored minus one.
    */
  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit

  /** The residual stream's update and the next block's input: `x += y`, then
    * `norm`'s RMS norm of `x` into its `out`. One launch where the backend
    * fuses them.
    */
  def addRmsNorm(x: Tensor, y: Tensor, norm: RowNorm): Unit = {
    add(x, y, x)
    rmsNorm(x, norm.weight, norm.epsilon, norm.weightOffset, norm.out)
  }

  /** RMS norm of each of a row's `groups` equal parts on its own, with `weight`
    * spanning the whole row (`[groups × width]`): the residual streams of Qwen
    * 3.8 Flash Next, each normed with its own weights.
    */
  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit

  /** AdaLN's modulation: `out = x × (1 + scale) + shift` per row of `x`,
    * `scale` and `shift` F32 `[cols]`.
    */
  def modulate(x: Tensor, scale: Tensor, shift: Tensor, out: Tensor): Unit

  /** A gated residual: `x += gate × y`, `gate` F32 `[cols]` for every row. */
  def gatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit

  /** Attention over many short sequences at once, no mask: `q` and `out`
    * `[S, B, heads, D]`, `k` and `v` `[S, B, kvHeads, D]` (row `s × B + b` is
    * position `s` of sequence `b`), scores `q·k × scale`, plus `bias` (F32
    * `[heads, S, S]`, by query then key position) when given: T5's relative
    * positions.
    */
  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor] = None
  ): Unit

  /** A 3×3 convolution, channels-last: `x` `[H, W, in]`, `weight` `[out, in ×
    * 9]` (a `[out][in][3][3]` kernel flattened; BF16 on the GPU, whose patches
    * and sums are BF16, or F16, whose patches are F16 and sums F32), `bias` F32
    * `[out]`, `out` `[H / stride, W / stride, out]`. Stride 1 pads by 1 on
    * every side; stride 2 (even H and W) by 1 at the bottom and right only, as
    * diffusers' downsamplers do. `replicate` pads with the nearest edge pixel
    * instead of zeros (PyTorch's `padding_mode="replicate"`; stride 1),
    * `reflect` with the reflection (`"reflect"`: −1 reads 1; either stride).
    */
  def conv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int = 1,
      replicate: Boolean = false,
      reflect: Boolean = false
  ): Unit

  /** A 1-D convolution over time, channels-last: `x` `[T, in]`, `weight` `[out,
    * in × taps]` (a `[out][in][taps]` kernel flattened; BF16 or F32 on the GPU,
    * its patches of the same type, the sums F32), taps `dilation` samples
    * apart, zeros `padLeft` before and `padRight` after `x`, `bias` F32 `[out]`
    * when given, `out` `[(T + padLeft + padRight − dilation × (taps − 1) − 1) /
    * stride + 1, out]`.
    */
  def conv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): Unit

  /** A transposed 1-D convolution's overlap-add: `columns` `[T, out × taps]`
    * (the input times the weight regrouped as `[out × taps, in]`), `out`
    * `[(T − 1) × stride − 2 × pad + taps, out]` with `out[t × stride + k − pad,
    * o] = Σ columns[t, o × taps + k] + bias[o]`.
    */
  def overlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): Unit

  /** BigVGAN's anti-aliased SnakeBeta over time, channels-last (`x` and `out`
    * `[T, C]`): each channel upsampled ×2 (a transposed convolution by
    * `upFilter`, times 2, replicate-padded by 5 then cropped by 15 at both
    * ends), then `u + inverseMagnitude × sin²(frequency × u)`, then filtered by
    * `downFilter` at stride 2 (replicate-padded by 5 before, 6 after). The two
    * filters F32 `[12]`, `frequency` and `inverseMagnitude` F32 `[C]`.
    */
  def antiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): Unit

  /** DAC's Snake over time, channels-last (`x` and `out` `[T, C]`): `x + sin²(α
    * x) / (α + 10⁻⁹)`, `alpha` F32 `[C]` as stored (not log-scale).
    */
  def snake(x: Tensor, alpha: Tensor, out: Tensor): Unit

  /** Rotary embedding from precomputed angles, pairs of adjacent values: pair
    * `m` of every head of token `t` (values `2m`, `2m + 1`) turns by the angle
    * whose cosine and sine are `cosines[t, m]` and `sines[t, m]`; the values
    * past the table's pairs are copied. `x` and `out` `[tokens, heads, D]`.
    * With `halves`, pair `m` is values `m` and `m + pairs` (transformers'
    * `rotate_half` over the head's first `2 × pairs` values). Tables of
    * `[tokens, heads, pairs]` give each head its own angles (LTX 2's "split"
    * RoPE).
    */
  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean = false
  ): Unit

  /** Whether the products of quantized weights over many rows run in BF16
    * rather than F16 (a backend that converts them): F32's range, for
    * activations past F16's (MiniMax H3's MLP). One setting for the process,
    * set by the pipeline that needs it.
    */
  @volatile var wideProducts: Boolean = false

  /** A channels-last image `[H, W, C]` as rows of patch-ordered pixels
    * `[L × patch², C]`: row `l × patch² + py × patch + px` is pixel `(hs ×
    * patch + py, ws × patch + px)` of patch `l = hs × W / patch + ws`.
    * `patchesToPixels` is its inverse.
    */
  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit
  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit

  /** AdaLN from a table of chunks per row: `x`, `out` `[rows, C]`, `table`
    * `[rows, chunks × C]`; `out = x × (1 + table chunk scale) + table chunk
    * shift`.
    */
  def modulateChunks(
      x: Tensor,
      table: Tensor,
      shift: Int,
      scale: Int,
      out: Tensor
  ): Unit

  /** `x += table chunk gate × y`, the tables as `modulateChunks`'. */
  def gatedAddChunk(x: Tensor, y: Tensor, table: Tensor, gate: Int): Unit

  /** `x[r] += gate[r] × y[r]`: one gate per row (`gate` `[rows]`). */
  def rowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit

  /** Group norm over rows of channels (`[pixels, C]`): group `g` is the
    * channels `[g C / groups, (g + 1) C / groups)` of every row, normalized by
    * its mean and variance over all of them, then `× weight + bias`.
    */
  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit

  /** `unpackPatches`' inverse: a channels-last image `[gridH × patch, gridW ×
    * patch, C]` into packed patches `[gridH × gridW, C × patch²]`.
    */
  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit

  /** `out` (`[rows, Σ columns]`) as `parts` (each `[rows, columns]`) side by
    * side.
    */
  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit

  /** Nearest-neighbour upsampling ×2, channels-last: `x` `[H, W, C]`, `out`
    * `[2H, 2W, C]`.
    */
  def upsample2x(x: Tensor, out: Tensor): Unit

  /** Wan 2.2's up-shortcut on one frame (diffusers' `DupUp3D`, the last of its
    * `frames` kept), channels-last: `x` `[H, W, C]` into `out` `[2H, 2W, O]`,
    * output `(y, x, o)` reading input channel `j / (O × frames × 4 / C)` at
    * `(y / 2, x / 2)`, where `j = ((o × frames + frames − 1) × 2 + y % 2) × 2 +
    * x % 2`.
    */
  def duplicateUp(x: Tensor, frames: Int, out: Tensor): Unit

  /** Wan 2.2's down-shortcut on one frame (diffusers' `AvgDown3D`, zero frames
    * before it to make `frames`), channels-last: `x` `[H, W, C]` into `out`
    * `[H / 2, W / 2, O]`, output channel `o` the mean of the `G = C × frames ×
    * 4 / O` values `k = o × G + g`, where value `k = ((c × frames + t) × 2 +
    * dy) × 2 + dx` is input `(2y + dy, 2x + dx, c)` for the last frame `t` and
    * zero for the others.
    */
  def averageDown(x: Tensor, frames: Int, out: Tensor): Unit

  /** `out[c, r] = x[r, c]`: F32, or F16 or BF16 as they are stored. */
  def transpose(x: Tensor, out: Tensor): Unit

  /** A weighted scatter of rows: `out[ids[s]] += weights[s] × rows[s]` for
    * every slot `s`, `rows` `[slots, D]`, `ids` I32 `[slots]`, `weights` F32
    * `[slots]`, `out` `[tokens, D]`. A token may take any number of slots (a
    * mixture of experts where the experts choose their tokens).
    */
  def scatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): Unit

  /** A depthwise 3×3 convolution, channels-last, zeros around the image: `x`
    * and `out` `[H, W, C]`, `weight` F32 `[C, 9]` (each channel's own kernel,
    * tap `ky × 3 + kx`), `bias` F32 `[C]`.
    */
  def depthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): Unit

  /** Each column's mean over the rows: `x` `[rows, C]`, `out` `[C]` (a global
    * average pool of rows of pixels).
    */
  def columnMean(x: Tensor, out: Tensor): Unit

  /** Packed `patch × patch` patches (`[gridH × gridW, C × patch²]`, feature
    * `c × patch² + py × patch + px`) back to a channels-last image `[gridH ×
    * patch, gridW × patch, C]`.
    */
  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit

  /** Layer norm over the last dimension, `weight` and `bias` optional. */
  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit

  /** `out = softmax(x × scale)` over the last dimension. */
  def softmax(x: Tensor, scale: Float, out: Tensor): Unit

  /** Rotary embedding of `x` `[tokens, heads, headDimension]`; `positions` is
    * I32 `[tokens]`, or `[3, tokens]` for mRoPE.
    */
  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit

  /** `out = x · weightᵀ`, a linear layer without bias: `x` is `[M, K]` F32,
    * `weight` `[N, K]` in any storage type, `out` `[M, N]` F32.
    */
  def linear(x: Tensor, weight: Tensor, out: Tensor): Unit

  /** Several linear layers of the same `x` (Q, K and V; gate and up): a backend
    * may prepare `x` once for all of them.
    */
  def linears(x: Tensor, weights: Seq[Tensor], outs: Seq[Tensor]): Unit =
    weights.zip(outs).foreach((weight, out) => linear(x, weight, out))

  /** Sets every byte of a tensor to zero: a fresh recurrent state. */
  def zero(tensor: Tensor): Unit

  /** Each row's last dimension, `2D`, split into its halves: `[.., 2D]` into
    * `first` and `second`, `[.., D]` each. Qwen 3.5's query projection holds a
    * head's query then its output gate.
    */
  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit

  /** The inverse of `splitHalves`: `first` and `second` side by side. */
  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit

  /** A causal depthwise convolution over time, its taps `dilation` tokens
    * apart, then SiLU: `x` and `out` `[tokens, C]`, `weight` `[C, taps]`,
    * `state` `[(taps − 1) × dilation, C]` holding the inputs before `x`'s first
    * token, and left holding its last ones. A `history` (`[≥ tokens, (taps − 1)
    * × dilation, C]`) receives the state after each token.
    */
  def causalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      history: Option[Tensor],
      out: Tensor
  ): Unit

  /** The gated delta rule over `tokens` in order (Gated DeltaNet): `qkv`
    * `[tokens, rule.qkvWidth]` (q, k, v heads), `a` and `b` `[tokens,
    * valueHeads]`, `decay` (`−exp(A_log)`) and `dtBias` `[valueHeads]`, `state`
    * `[valueHeads, D, D]` F32 carried between calls, `out` `[tokens,
    * valueHeads, D]`. A `history` (`[≥ tokens, valueHeads, D, D]`) receives the
    * state after each token.
    */
  def gatedDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit

  /** `causalConv` of `qkv` (dilation 1) into `convolved`, then `gatedDeltaRule`
    * of it: one launch where the backend fuses them (and leaves `convolved`
    * unwritten).
    */
  def convolvedDeltaRule(
      qkv: Tensor,
      convWeight: Tensor,
      convState: Tensor,
      convHistory: Option[Tensor],
      convolved: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    causalConv(qkv, convWeight, 1, convState, convHistory, convolved)
    gatedDeltaRule(convolved, a, b, decay, dtBias, state, rule, history, out)
  }

  /** Copies `from` into `to`, byte for byte (same type and size): a rollback, a
    * kept row. Ordered with the kernels; the host does not wait.
    */
  def copy(from: Tensor, to: Tensor): Unit

  /** Top-k routing: each token's `k` largest `logits` (`[tokens, experts]`)
    * into `ids` (I32 `[tokens, k]`), their softmax among themselves into
    * `weights` (`[tokens, k]`).
    */
  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit

  /** Experts' products: slot `s` multiplies expert `ids[s]` of `weight`
    * (`[experts, N, K]`) with row `s / (slots / rows of x)` of `x`, into row
    * `s` of `out` (`[slots, N]`).
    */
  def expertsLinear(x: Tensor, weight: Tensor, ids: Tensor, out: Tensor): Unit

  /** `out[t] = residual[t] + Σ_j weights[t, j] · expertOut[t·k + j]`, plus the
    * shared expert's output scaled by `sigmoid(input[t] · router)` when there
    * is one. `out` may be the residual (the layer's `x`).
    */
  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit

  /** `moeCombine` into `residual`, then `norm`'s RMS norm of it into its `out`
    * (the next block's input). `norm.out` may be the shared expert's input.
    */
  def moeCombineNorm(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    moeCombine(expertOut, weights, shared, Some(residual), residual)
    rmsNorm(residual, norm.weight, norm.epsilon, norm.weightOffset, norm.out)
  }

  /** `expertOutputs`, then their sum into `residual` and its norm
    * (`moeCombineNorm`, the shared expert weighted by its gate vector over
    * `x`): the whole mixture of experts in one launch where the backend fuses
    * it.
    */
  def mixtureNorm(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    expertOutputs(x, weights, into)
    moeCombineNorm(
      into.expertOut,
      into.routeWeights,
      Some(SharedExpert(into.sharedOut, x, weights.sharedRouter)),
      residual,
      norm
    )
  }

  /** A router's `logits = x · routerᵀ`, then `route` of them. */
  def routeLinear(
      x: Tensor,
      router: Tensor,
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): Unit = {
    linear(x, router, logits)
    route(logits, ids, weights)
  }

  /** SwiGLU of experts: slot `s` computes `silu(gate · x) × (up · x)` with
    * expert `ids[s]` of `gate` and `up` (`[experts, N, K]` each), `x` as in
    * `expertsLinear`.
    */
  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit

  /** `expertsGatedLinear` of the routed experts, and of a shared expert of the
    * same shapes (`sharedGate` and `sharedUp` `[1, N, K]`, `sharedIds` a zero
    * per token) into `sharedOut`: one launch where the backend fuses them.
    */
  def expertsGatedLinearWithShared(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      sharedGate: Tensor,
      sharedUp: Tensor,
      sharedIds: Tensor,
      sharedOut: Tensor
  ): Unit = {
    expertsGatedLinear(x, gate, up, ids, out)
    expertsGatedLinear(x, sharedGate, sharedUp, sharedIds, sharedOut)
  }

  /** A mixture of experts' outputs on `x` before they are summed (see
    * `ExpertActivations`): the routing, every slot's SwiGLU and down
    * projection, and the shared expert's. One launch where the backend fuses
    * them.
    */
  def expertOutputs(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Unit = {
    routeLinear(x, weights.router, into.logits, into.ids, into.routeWeights)
    val slots = into.ids.view(into.ids.shape.elementCount)
    expertsGatedLinearWithShared(
      x,
      weights.gate,
      weights.up,
      slots,
      into.gated,
      weights.sharedGate,
      weights.sharedUp,
      into.sharedIds,
      into.sharedGated
    )
    expertsLinear(into.gated, weights.down, slots, into.expertOut)
    linear(into.sharedGated, weights.sharedDown, into.sharedOut)
  }

  /** Gated DeltaNet's output norm: `out = rmsNorm(x) × weight ×
    * gateActivation(gate)` (SiLU in Qwen 3.5/3.6, sigmoid in 3.8), per row of
    * `x` and `gate` (`[rows, D]`).
    */
  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit

  /** A hyper-connection's read of the residual streams: `out[t] = mean_j
    * sigmoid(gates[t, j]) ⊙ normed[t, j]`, `gates` (logits) and `normed`
    * `[tokens, streams × width]`, `out` `[tokens, width]`.
    */
  def streamsMix(gates: Tensor, normed: Tensor, streams: Int, out: Tensor): Unit

  /** A hyper-connection's write: `x[t, j] += 2 sigmoid(logits[t, j] / streams)
    * × y[t]`, `x` `[tokens, streams × width]`, `y` `[tokens, width]`, `logits`
    * `[tokens, streams]` (zero: a plain residual add).
    */
  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit

  /** The n-gram embedding's gate per stream: `s = ⟨key[t, j], query[t, j]⟩ /
    * √width`, `out[t, j] = sigmoid(sign(s) √max(|s|, 1e−6)) × value[t]`; `key`,
    * `query` and `out` `[tokens, streams × width]`, `value` `[tokens, width]`.
    */
  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit

  /** The n-gram rows of each token (`NgramHash`): `tokens` I32 `[T]`, `rows`
    * I32 `[T, heads]`. `state` (F32 `[kept]`) holds the tokens before the
    * call's first as id + 1, oldest first (0: none, taken as EOS), and is left
    * holding its last ones. A `history` (`[≥ T, kept]`) receives it after each
    * token.
    */
  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit

  /** An empty cache of `pages` pages (`KvCache`). */
  def allocateCache(
      pages: Int,
      pageSize: Int,
      kvHeads: Int,
      headDimension: Int
  ): KvCache

  /** Writes the keys and values of `k`, `v` (F32 `[tokens, kvHeads, D]`) at
    * positions `start` on, through the sequence's `pageTable`.
    */
  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit

  /** A gated attention's inputs from its projections: each query head's half of
    * `queryAndGate` (`[t, heads, 2·D]`: the query, then the output gate into
    * `gate`) and `keys` (`[t, kvHeads, D]`) normed and turned as `norms` says,
    * the queries into `rotatedQueries`, the keys written to `cache` with
    * `values` at `start`. `queries` and `rotatedKeys` hold what lies between,
    * where the steps run apart. One launch where the backend fuses them.
    */
  def attentionInputs(
      queryAndGate: Tensor,
      keys: Tensor,
      values: Tensor,
      norms: QueryKeyNorms,
      positions: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int,
      queries: Tensor,
      gate: Tensor,
      rotatedKeys: Tensor,
      rotatedQueries: Tensor
  ): Unit = {
    val Seq(t, heads, d) = queries.shape.dimensions
    val kvHeads = keys.shape.dimensions(1)
    splitHalves(queryAndGate, queries, gate)
    rmsNorm(
      queries.view(t * heads, d),
      norms.queryNorm,
      norms.epsilon,
      norms.weightOffset,
      queries.view(t * heads, d)
    )
    rmsNorm(
      keys.view(t * kvHeads, d),
      norms.keyNorm,
      norms.epsilon,
      norms.weightOffset,
      keys.view(t * kvHeads, d)
    )
    rope(queries, positions, norms.rope, rotatedQueries)
    rope(keys, positions, norms.rope, rotatedKeys)
    cacheWrite(rotatedKeys, values, cache, pageTable, start)
  }

  /** Attention of `q` (F32 `[tokens, qHeads, D]`, token `t` at position
    * `queryStart + t`) over the sequence's first `keyCount` cached keys,
    * `qHeads / kvHeads` query heads per key head, into `out` (like `q`).
    */
  def attention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): Unit
}

/** A mixture's shared expert in `Ops.moeCombine`: its `output` `[tokens, H]`,
  * weighted by the sigmoid of `input` (`[tokens, H]`) · `router` (F32 `[H]`).
  */
final case class SharedExpert(output: Tensor, input: Tensor, router: Tensor)

/** A mixture of experts' weights: an F32 router `[experts, hidden]`, the routed
  * experts' gate, up (`[experts, N, hidden]`) and down (`[experts, hidden,
  * N]`), and a shared expert's gate and up (`[1, N, hidden]`), down (`[hidden,
  * N]`) and F32 gate vector (`[hidden]`, see `SharedExpert`).
  */
final case class ExpertProjections(
    router: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    sharedGate: Tensor,
    sharedUp: Tensor,
    sharedDown: Tensor,
    sharedRouter: Tensor
)

/** A mixture of experts' activations for `t` tokens of `k` slots each: the
  * router's `logits` `[t, experts]`, the chosen `ids` and `routeWeights`
  * `[t, k]`, each slot's SwiGLU `gated` `[t·k, N]` and output `expertOut`
  * `[t·k, hidden]`; the shared expert's `sharedIds` (a zero per token),
  * `sharedGated` `[t, N]` and `sharedOut` `[t, hidden]`.
  */
final case class ExpertActivations(
    logits: Tensor,
    ids: Tensor,
    routeWeights: Tensor,
    gated: Tensor,
    expertOut: Tensor,
    sharedIds: Tensor,
    sharedGated: Tensor,
    sharedOut: Tensor
)

/** A gated attention's query and key preparation: RMS norms with `queryNorm`
  * and `keyNorm` (`epsilon` and `weightOffset` as `Ops.rmsNorm` takes them),
  * then `rope`.
  */
final case class QueryKeyNorms(
    queryNorm: Tensor,
    keyNorm: Tensor,
    epsilon: Float,
    weightOffset: Float,
    rope: Rope
)

/** An RMS norm taken of a result as it is written: `weight`, `epsilon` and
  * `weightOffset` as `Ops.rmsNorm` takes them, into `out`.
  */
final case class RowNorm(
    weight: Tensor,
    epsilon: Float,
    weightOffset: Float,
    out: Tensor
)

/** A weights file as a backend maps it; `close` unmaps it. */
final class MappedWeights(
    val segment: java.lang.foreign.MemorySegment,
    val storage: drift.runner.tensor.Storage,
    closing: () => Unit
) extends AutoCloseable {
  def close(): Unit = closing()
}

/** The shape and dtype rules both backends enforce, so a mistake fails the same
  * way on either.
  */
object Ops {

  /** The most candidates `candidates` returns per row: one workgroup sorts
    * them.
    */
  val MaximumCandidates = 1024

  def requireCandidates(values: Tensor, count: Int): Unit = {
    requireF32("candidates", values)
    require(
      count >= 1 && count <= MaximumCandidates,
      s"$count candidates, 1 to $MaximumCandidates"
    )
  }

  def requireF32(operation: String, tensors: Tensor*): Unit =
    tensors.foreach { tensor =>
      require(
        tensor.dtype == DType.F32,
        s"$operation takes F32, not ${tensor.dtype}"
      )
    }

  def requireSameShape(operation: String, tensors: Tensor*): Unit =
    require(
      tensors.map(_.shape).distinct.size == 1,
      s"$operation needs equal shapes, got ${tensors.map(_.shape).mkString(", ")}"
    )

  /** Same-shape F32 operands. */
  def checkElementwise(operation: String, tensors: Tensor*): Unit = {
    requireF32(operation, tensors*)
    requireSameShape(operation, tensors*)
  }

  def checkSplitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    requireF32("splitHalves", x, first, second)
    requireSameShape("splitHalves", first, second)
    require(
      x.shape.elementCount == 2 * first.shape.elementCount && x.shape.last == 2 * first.shape.last,
      s"splitHalves: ${x.shape} into ${first.shape}"
    )
  }

  def checkCopy(from: Tensor, to: Tensor): Unit =
    require(
      from.dtype == to.dtype && from.byteSize == to.byteSize,
      s"copy: ${from.dtype} ${from.shape} into ${to.dtype} ${to.shape}"
    )

  /** A state history holds one F32 state per token, room for `tokens`. */
  def checkHistory(history: Tensor, tokens: Int, state: Tensor): Unit =
    require(
      history.dtype == DType.F32 && history.shape.elementCount >= tokens.toLong * state.shape.elementCount,
      s"history ${history.shape} cannot keep $tokens states of ${state.shape}"
    )

  /** Returns `(tokens, channels, taps)`. */
  def checkCausalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    checkElementwise("causalConv", x, out)
    requireF32("causalConv", weight, state)
    require(
      x.shape.rank == 2,
      s"causalConv takes [tokens, channels], not ${x.shape}"
    )
    val Seq(tokens, channels) = x.shape.dimensions.map(_.toInt)
    val taps = weight.shape.last.toInt
    require(
      weight.shape == Shape.of(channels, taps) && taps <= 9,
      s"causalConv: weight ${weight.shape} for $channels channels"
    )
    val span = (taps - 1) * dilation
    require(
      dilation >= 1 && span <= 16 && state.shape == Shape.of(span, channels),
      s"causalConv: state ${state.shape}, expected [$span, $channels]"
    )
    (tokens, channels, taps)
  }

  /** Returns `(H, W, in, out)` of the input. */
  def checkConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int
  ): (Int, Int, Int, Int) = {
    requireF32("conv3x3", x, bias, out)
    require(
      x.shape.rank == 3 && out.shape.rank == 3,
      s"conv3x3 takes [H, W, C], not ${x.shape}"
    )
    val Seq(height, width, in) = x.shape.dimensions.map(_.toInt)
    val Seq(outChannels, columns) = weight.shape.dimensions.map(_.toInt)
    require(
      stride == 1 || stride == 2 && height % 2 == 0 && width % 2 == 0,
      s"conv3x3: stride $stride on ${x.shape}"
    )
    require(
      columns == 9 * in && bias.shape == Shape.of(outChannels) &&
        out.shape == Shape.of(height / stride, width / stride, outChannels),
      s"conv3x3: x ${x.shape}, weight ${weight.shape}, out ${out.shape}, stride $stride"
    )
    (height, width, in, outChannels)
  }

  /** `conv1d`'s `(T, in, out length, out channels)`. */
  def checkConv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("conv1d", (Seq(x, out) ++ bias)*)
    require(
      x.shape.rank == 2 && out.shape.rank == 2,
      s"conv1d takes [T, C], not ${x.shape}"
    )
    val Seq(length, in) = x.shape.dimensions.map(_.toInt)
    val Seq(outChannels, columns) = weight.shape.dimensions.map(_.toInt)
    require(
      taps > 0 && dilation > 0 && stride > 0 && padLeft >= 0 && padRight >= 0,
      s"conv1d: $taps taps, dilation $dilation, stride $stride, pads $padLeft/$padRight"
    )
    val span = length + padLeft + padRight - dilation * (taps - 1) - 1
    require(span >= 0, s"conv1d: $length samples under $taps taps")
    val outLength = span / stride + 1
    require(
      columns == in * taps && bias.forall(_.shape == Shape.of(outChannels)) &&
        out.shape == Shape.of(outLength, outChannels),
      s"conv1d: x ${x.shape}, weight ${weight.shape}, $taps taps, out ${out.shape}"
    )
    (length, in, outLength, outChannels)
  }

  /** `overlapAdd`'s `(T, out length, out channels)`. */
  def checkOverlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("overlapAdd", columns, bias, out)
    val Seq(length, width) = columns.shape.dimensions.map(_.toInt)
    val outChannels = width / taps
    val outLength = (length - 1) * stride - 2 * pad + taps
    require(
      width % taps == 0 && pad >= 0 && outLength > 0 &&
        bias.shape == Shape.of(outChannels) &&
        out.shape == Shape.of(outLength, outChannels),
      s"overlapAdd: columns ${columns.shape}, $taps taps, stride $stride, pad $pad, out ${out.shape}"
    )
    (length, outLength, outChannels)
  }

  /** `antiAliasedSnake`'s `(T, C)`. */
  def checkAntiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): (Int, Int) = {
    requireF32(
      "antiAliasedSnake",
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    requireSameShape("antiAliasedSnake", x, out)
    val Seq(length, channels) = x.shape.dimensions.map(_.toInt)
    require(
      frequency.shape == Shape.of(channels) &&
        inverseMagnitude.shape == Shape.of(channels) &&
        upFilter.shape == Shape.of(SnakeTaps) &&
        downFilter.shape == Shape.of(SnakeTaps),
      s"antiAliasedSnake: x ${x.shape}, filters ${upFilter.shape} and ${downFilter.shape}"
    )
    (length, channels)
  }

  /** `snake`'s `(T, C)`. */
  def checkSnake(x: Tensor, alpha: Tensor, out: Tensor): (Int, Int) = {
    requireF32("snake", x, alpha, out)
    requireSameShape("snake", x, out)
    val Seq(length, channels) = x.shape.dimensions.map(_.toInt)
    require(
      alpha.shape == Shape.of(channels),
      s"snake: x ${x.shape}, alpha ${alpha.shape}"
    )
    (length, channels)
  }

  /** The taps of BigVGAN's anti-aliasing filters. */
  val SnakeTaps = 12

  /** Returns `(tokens, heads, D, pairs)`. */
  def checkRopeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int, Boolean) = {
    checkElementwise("ropeTable", x, out)
    requireF32("ropeTable", cosines, sines)
    require(
      x.shape.rank == 3,
      s"ropeTable takes [tokens, heads, D], not ${x.shape}"
    )
    val Seq(tokens, heads, d) = x.shape.dimensions.map(_.toInt)
    val pairs = cosines.shape.last.toInt
    val perHead = cosines.shape.rank == 3
    require(
      (cosines.shape == Shape.of(tokens, pairs) ||
        cosines.shape == Shape.of(tokens, heads, pairs)) &&
        sines.shape == cosines.shape && 2 * pairs <= d,
      s"ropeTable: tables ${cosines.shape}, ${sines.shape} for ${x.shape}"
    )
    (tokens, heads, d, pairs, perHead)
  }

  /** Returns `(H, W, C)` of the image. */
  def checkPatchPixels(
      image: Tensor,
      rows: Tensor,
      patch: Int
  ): (Int, Int, Int) = {
    requireF32("patch pixels", image, rows)
    val Seq(height, width, channels) = image.shape.dimensions.map(_.toInt)
    require(
      height % patch == 0 && width % patch == 0 &&
        rows.shape == Shape.of(height.toLong * width, channels),
      s"patch pixels: ${image.shape} in patches of $patch as ${rows.shape}"
    )
    (height, width, channels)
  }

  /** Returns `(rows, C, chunks)`. */
  def checkChunks(
      operation: String,
      x: Tensor,
      table: Tensor,
      out: Tensor,
      indices: Int*
  ): (Long, Int, Int) = {
    checkElementwise(operation, x, out)
    requireF32(operation, table)
    require(x.shape.rank == 2, s"$operation takes [rows, C], not ${x.shape}")
    val Seq(rows, channels) = x.shape.dimensions
    val width = table.shape.last
    require(
      table.shape.rank == 2 && table.shape.dimensions.head == rows &&
        width % channels == 0 &&
        indices.forall(i => i >= 0 && i < width / channels),
      s"$operation: table ${table.shape} for ${x.shape}, chunks $indices"
    )
    (rows, channels.toInt, (width / channels).toInt)
  }

  /** Returns `(rows, cols)`. */
  def checkRowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): (Long, Int) = {
    checkElementwise("rowGatedAdd", x, y)
    requireF32("rowGatedAdd", gate)
    require(
      x.shape.rank == 2,
      s"rowGatedAdd takes [rows, cols], not ${x.shape}"
    )
    val Seq(rows, cols) = x.shape.dimensions
    require(
      gate.shape.elementCount == rows,
      s"rowGatedAdd: gate ${gate.shape} for ${x.shape}"
    )
    (rows, cols.toInt)
  }

  /** Returns `(rows, C)`. */
  def checkGroupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): (Long, Int) = {
    checkRowWise("groupNorm", x, out, weight, bias)
    require(x.shape.rank == 2, s"groupNorm takes [rows, C], not ${x.shape}")
    val Seq(rows, channels) = x.shape.dimensions
    require(
      groups > 0 && channels % groups == 0,
      s"groupNorm: $groups groups of $channels channels"
    )
    (rows, channels.toInt)
  }

  /** Returns `(gridH, gridW, C)`. */
  def checkPackPatches(
      image: Tensor,
      patch: Int,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("packPatches", image, out)
    val Seq(height, width, channels) = image.shape.dimensions.map(_.toInt)
    require(
      height % patch == 0 && width % patch == 0 &&
        out.shape == Shape.of(
          (height / patch).toLong * (width / patch),
          channels.toLong * patch * patch
        ),
      s"packPatches: ${image.shape} in patches of $patch into ${out.shape}"
    )
    (height / patch, width / patch, channels)
  }

  /** Returns `(rows, columns of each part)`. */
  def checkConcatColumns(parts: Seq[Tensor], out: Tensor): (Long, Seq[Int]) = {
    requireF32("concatColumns", (parts :+ out)*)
    require(
      (parts :+ out).forall(_.shape.rank == 2),
      "concatColumns takes [rows, columns]"
    )
    val rows = out.shape.dimensions.head
    val columns = parts.map(_.shape.last.toInt)
    require(
      parts.forall(_.shape.dimensions.head == rows) &&
        columns.sum == out.shape.last,
      s"concatColumns: ${parts.map(_.shape).mkString(", ")} into ${out.shape}"
    )
    (rows, columns)
  }

  /** Returns `(H, W, C)` of the input. */
  def checkUpsample2x(x: Tensor, out: Tensor): (Int, Int, Int) = {
    requireF32("upsample2x", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == Shape.of(2L * height, 2L * width, channels),
      s"upsample2x: ${x.shape} into ${out.shape}"
    )
    (height, width, channels)
  }

  /** `duplicateUp`'s `(height, width, channels, out channels)`. */
  def checkDuplicateUp(
      x: Tensor,
      frames: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("duplicateUp", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    val outChannels = out.shape.last.toInt
    require(
      out.shape == Shape.of(2L * height, 2L * width, outChannels) &&
        outChannels * frames * 4 % channels == 0,
      s"duplicateUp: ${x.shape} into ${out.shape} over $frames frames"
    )
    (height, width, channels, outChannels)
  }

  /** `averageDown`'s `(height, width, channels, out channels)`, of `x`. */
  def checkAverageDown(
      x: Tensor,
      frames: Int,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("averageDown", x, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    val outChannels = out.shape.last.toInt
    require(
      height % 2 == 0 && width % 2 == 0 &&
        out.shape == Shape.of(height / 2L, width / 2L, outChannels) &&
        channels * frames * 4 % outChannels == 0,
      s"averageDown: ${x.shape} into ${out.shape} over $frames frames"
    )
    (height, width, channels, outChannels)
  }

  /** `depthwiseConv3x3`'s `(height, width, channels)`. */
  def checkDepthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): (Int, Int, Int) = {
    requireF32("depthwiseConv3x3", x, weight, bias, out)
    val Seq(height, width, channels) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == x.shape && weight.shape == Shape.of(channels, 9) &&
        bias.shape == Shape.of(channels),
      s"depthwiseConv3x3: ${x.shape} by ${weight.shape} and ${bias.shape} into ${out.shape}"
    )
    (height, width, channels)
  }

  /** `columnMean`'s `(rows, columns)`. */
  def checkColumnMean(x: Tensor, out: Tensor): (Long, Int) = {
    requireF32("columnMean", x, out)
    val Seq(rows, columns) = x.shape.dimensions
    require(
      out.shape == Shape.of(columns),
      s"columnMean: ${x.shape} into ${out.shape}"
    )
    (rows, columns.toInt)
  }

  /** `scatterAddRows`' `(slots, columns)`. */
  def checkScatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): (Long, Int) = {
    requireF32("scatterAddRows", rows, weights, out)
    val Seq(slots, columns) = rows.shape.dimensions
    require(
      ids.dtype == DType.I32 && ids.shape == Shape.of(slots) &&
        weights.shape == Shape.of(slots) && out.shape.last == columns,
      s"scatterAddRows: ${rows.shape} by ${ids.shape} and ${weights.shape} into ${out.shape}"
    )
    (slots, columns.toInt)
  }

  def checkTranspose(x: Tensor, out: Tensor): Seq[Int] = {
    require(
      x.dtype == out.dtype &&
        Seq(DType.F32, DType.F16, DType.BF16).contains(x.dtype),
      s"transpose takes F32, F16 or BF16 into the same, not ${x.dtype} into ${out.dtype}"
    )
    val Seq(rows, cols) = x.shape.dimensions.map(_.toInt)
    require(
      out.shape == Shape.of(cols, rows),
      s"transpose: ${x.shape} into ${out.shape}"
    )
    Seq(rows, cols)
  }

  /** Returns `(gridW, C)`. */
  def checkUnpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): (Int, Int) = {
    requireF32("unpackPatches", packed, out)
    val Seq(tokens, features) = packed.shape.dimensions.map(_.toInt)
    val (gridWidth, channels) =
      (tokens / gridHeight, features / (patch * patch))
    require(
      tokens % gridHeight == 0 && features % (patch * patch) == 0 &&
        out.shape == Shape
          .of(gridHeight.toLong * patch, gridWidth.toLong * patch, channels),
      s"unpackPatches: ${packed.shape} on $gridHeight rows into ${out.shape}"
    )
    (gridWidth, channels)
  }

  /** Returns `(S, B, heads, kvHeads, D)`. */
  def checkShortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int, Int) = {
    requireF32("shortAttention", q, k, v, out)
    requireSameShape("shortAttention", q, out)
    requireSameShape("shortAttention", k, v)
    require(
      q.shape.rank == 4 && k.shape.rank == 4,
      s"shortAttention takes [S, B, heads, D], not ${q.shape}"
    )
    val Seq(s, b, heads, d) = q.shape.dimensions.map(_.toInt)
    val Seq(ks, kb, kvHeads, kd) = k.shape.dimensions.map(_.toInt)
    require(
      ks == s && kb == b && kd == d && heads % kvHeads == 0 && d <= 512,
      s"shortAttention: q ${q.shape}, k ${k.shape}"
    )
    (s, b, heads, kvHeads, d)
  }

  def checkAttentionBias(bias: Tensor, heads: Int, s: Int): Unit = {
    requireF32("shortAttention bias", bias)
    require(
      bias.shape == Shape.of(heads, s, s),
      s"shortAttention: bias ${bias.shape} for $heads heads over $s positions"
    )
  }

  /** Returns the token count. */
  def checkNgramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      rows: Tensor
  ): Int = {
    require(
      tokens.dtype == DType.I32 && rows.dtype == DType.I32,
      s"ngramRows: tokens ${tokens.dtype}, rows ${rows.dtype}"
    )
    requireF32("ngramRows", state)
    val count = tokens.shape.elementCount.toInt
    require(
      rows.shape == Shape.of(count, hash.heads) &&
        state.shape == Shape.of(hash.kept),
      s"ngramRows: rows ${rows.shape}, state ${state.shape} for $count tokens"
    )
    count
  }

  /** Returns `(tokens, streams, width)`. */
  def checkStreams(
      operation: String,
      streamsTensors: Seq[Tensor],
      single: Tensor
  ): (Int, Int, Int) = {
    requireF32(operation, (streamsTensors :+ single)*)
    requireSameShape(operation, streamsTensors*)
    val Seq(tokens, width) = single.shape.dimensions.map(_.toInt)
    val all = streamsTensors.head.shape
    require(
      all.rank == 2 && all.dimensions.head == tokens && all.last % width == 0,
      s"$operation: streams $all for rows of ${single.shape}"
    )
    (tokens, (all.last / width).toInt, width)
  }

  /** Returns the token count. */
  def checkDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      out: Tensor
  ): Int = {
    requireF32("gatedDeltaRule", qkv, a, b, decay, dtBias, state, out)
    val tokens = qkv.shape.dimensions.head.toInt
    val (v, d) = (rule.valueHeads.toLong, rule.dimension.toLong)
    require(
      qkv.shape == Shape.of(tokens, rule.qkvWidth),
      s"gatedDeltaRule: qkv ${qkv.shape}, expected [$tokens, ${rule.qkvWidth}]"
    )
    require(
      a.shape == Shape.of(tokens, v) && b.shape == a.shape,
      s"gatedDeltaRule: a ${a.shape}, b ${b.shape}"
    )
    require(
      decay.shape == Shape.of(v) && dtBias.shape == Shape.of(v),
      s"gatedDeltaRule: decay ${decay.shape}, dtBias ${dtBias.shape}"
    )
    require(
      state.shape == Shape.of(v, d, d),
      s"gatedDeltaRule: state ${state.shape}"
    )
    require(
      out.shape == Shape.of(tokens, v, d),
      s"gatedDeltaRule: out ${out.shape}"
    )
    tokens
  }

  /** Returns `(tokens, experts, k)`. */
  def checkRoute(
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): (Int, Int, Int) = {
    requireF32("route", logits, weights)
    require(
      ids.dtype == DType.I32 && ids.shape == weights.shape && logits.shape.rank == 2,
      s"route: ${logits.shape}, ${ids.shape}, ${weights.shape}"
    )
    val Seq(tokens, experts) = logits.shape.dimensions.map(_.toInt)
    val k = weights.shape.last.toInt
    require(
      weights.shape == Shape.of(tokens, k) && k <= 16 && experts <= 512,
      s"route: top-$k of $experts"
    )
    (tokens, experts, k)
  }

  /** Returns `(slots, N, K, divisor)`. */
  def checkExperts(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): (Int, Int, Int, Int) = {
    requireF32("expertsLinear", x, out)
    require(
      weight.shape.rank == 3 && ids.dtype == DType.I32,
      s"expertsLinear: weight ${weight.shape}, ids ${ids.dtype}"
    )
    val Seq(_, n, k) = weight.shape.dimensions.map(_.toInt)
    val slots = ids.shape.elementCount.toInt
    val rows = x.shape.dimensions.head.toInt
    require(
      x.shape == Shape.of(rows, k) && slots % rows == 0,
      s"expertsLinear: x ${x.shape} for $slots slots of $k"
    )
    require(
      out.shape == Shape.of(slots, n),
      s"expertsLinear: out ${out.shape}, expected [$slots, $n]"
    )
    (slots, n, k, slots / rows)
  }

  def checkEmbedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    require(
      table.shape.rank == 2,
      s"an embedding table is [vocabulary, D], not ${table.shape}"
    )
    require(
      ids.dtype == DType.I32 && ids.shape.rank == 1,
      s"embedding ids are I32 [tokens], not ${ids.dtype} ${ids.shape}"
    )
    requireF32("embedding", out)
    require(
      out.shape == Shape.of(ids.shape.dimensions.head, table.shape.last),
      s"embedding: out ${out.shape} for ${ids.shape.dimensions.head} rows of ${table.shape.last}"
    )
  }

  def checkAddRow(x: Tensor, row: Tensor, out: Tensor): Unit = {
    checkElementwise("addRow", x, out)
    requireF32("addRow", row)
    require(
      row.shape == Shape.of(x.shape.last),
      s"addRow: a row of ${row.shape} for rows of ${x.shape.last}"
    )
  }

  /** The conversions `convert` knows, by (from, to). */
  val Conversions: Set[(DType, DType)] = Set(
    DType.F32 -> DType.F16,
    DType.F32 -> DType.BF16,
    DType.F16 -> DType.F32,
    DType.BF16 -> DType.F32,
    DType.F8E4M3 -> DType.F32
  )

  def checkConvert(x: Tensor, out: Tensor): Unit = {
    requireSameShape("convert", x, out)
    require(
      Conversions.contains(x.dtype -> out.dtype),
      s"convert: no conversion from ${x.dtype} to ${out.dtype}"
    )
  }

  /** Returns the row count. */
  def checkRowWise(
      operation: String,
      x: Tensor,
      out: Tensor,
      rowVectors: Tensor*
  ): Long = {
    checkElementwise(operation, x, out)
    require(x.shape.rank >= 1, s"$operation needs at least one dimension")
    rowVectors.foreach { vector =>
      requireF32(operation, vector)
      require(
        vector.shape == Shape.of(x.shape.last),
        s"$operation: ${vector.shape} for rows of ${x.shape.last}"
      )
    }
    x.shape.elementCount / x.shape.last
  }

  /** Returns `(tokens, heads, headDimension)`. */
  def checkRope(
      x: Tensor,
      positions: Tensor,
      rope: Rope,
      out: Tensor
  ): (Int, Int, Int) = {
    checkElementwise("rope", x, out)
    require(
      x.shape.rank == 3,
      s"rope takes [tokens, heads, headDimension], got ${x.shape}"
    )
    val Seq(tokens, heads, dimension) = x.shape.dimensions.map(_.toInt)
    require(
      positions.dtype == DType.I32,
      s"rope positions are I32, not ${positions.dtype}"
    )
    rope.sections match {
      case RopeSections.Axes(pairs) =>
        require(
          pairs.size <= 4 && pairs.forall(_ > 0) &&
            pairs.sum * 2 == rope.rotaryDimensions,
          s"rope: axes of $pairs pairs for ${rope.rotaryDimensions} rotary values"
        )
      case _ => ()
    }
    val expected = rope.sections.positionsPerToken match {
      case 1     => Shape.of(tokens)
      case count => Shape.of(count, tokens)
    }
    require(
      positions.shape == expected,
      s"rope positions ${positions.shape}, expected $expected"
    )
    require(
      rope.rotaryDimensions % 2 == 0 && rope.rotaryDimensions <= dimension,
      s"rope: ${rope.rotaryDimensions} rotary dimensions in heads of $dimension"
    )
    (tokens, heads, dimension)
  }

  def checkCacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Int = {
    checkElementwise("cacheWrite", k, v)
    require(
      k.shape.rank == 3 && k.shape.dimensions
        .drop(1) == Vector(cache.kvHeads.toLong, cache.headDimension.toLong),
      s"cacheWrite: ${k.shape} into a cache of ${cache.kvHeads} heads of ${cache.headDimension}"
    )
    val tokens = k.shape.dimensions.head.toInt
    checkPageTable(pageTable, cache, start + tokens)
    tokens
  }

  def checkPageTable(
      pageTable: Tensor,
      cache: KvCache,
      positions: Int
  ): Unit = {
    require(
      pageTable.dtype == DType.I32 && pageTable.shape.rank == 1,
      s"a page table is I32 [pages], not ${pageTable.dtype} ${pageTable.shape}"
    )
    require(
      pageTable.shape.elementCount * cache.pageSize >= positions,
      s"a page table of ${pageTable.shape.elementCount} pages for $positions positions"
    )
  }

  /** Returns `(tokens, qHeads)`. */
  def checkAttention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): (Int, Int) = {
    checkElementwise("attention", q, out)
    require(
      q.shape.rank == 3 && q.shape.last == cache.headDimension,
      s"attention: queries ${q.shape} against heads of ${cache.headDimension}"
    )
    val Seq(tokens, qHeads, _) = q.shape.dimensions.map(_.toInt)
    require(
      qHeads % cache.kvHeads == 0,
      s"attention: $qHeads query heads over ${cache.kvHeads} key heads"
    )
    // queries see keys by position only when causal or windowed; otherwise
    // they may lie past the keys (padding that attends to a prompt alone)
    require(
      queryStart >= 0 && (queryStart + tokens <= keyCount ||
        !attention.causal && attention.window.isEmpty),
      s"attention: queries at $queryStart..${queryStart + tokens - 1} past $keyCount keys"
    )
    checkPageTable(pageTable, cache, keyCount)
    attention.sinks.foreach { sinks =>
      requireF32("attention sinks", sinks)
      require(
        sinks.shape == Shape.of(qHeads),
        s"attention: sinks ${sinks.shape} for $qHeads heads"
      )
    }
    (tokens, qHeads)
  }

  /** Returns `(M, N, K)`. */
  def checkLinear(
      x: Tensor,
      weight: Tensor,
      out: Tensor
  ): (Long, Long, Long) = {
    requireF32("linear", x, out)
    require(
      x.shape.rank == 2 && weight.shape.rank == 2 && out.shape.rank == 2,
      s"linear takes [M, K] · [N, K]ᵀ → [M, N], got ${x.shape}, ${weight.shape}, ${out.shape}"
    )
    val Seq(m, k) = x.shape.dimensions
    val Seq(n, weightK) = weight.shape.dimensions
    require(
      weightK == k,
      s"linear: x rows of $k against weight rows of $weightK"
    )
    require(
      out.shape == Shape.of(m, n),
      s"linear: out ${out.shape} for [$m, $n]"
    )
    (m, n, k)
  }

  def checkBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Unit =
    require(
      bytes.length == dtype.byteSize(shape.elementCount),
      s"${bytes.length} bytes for $dtype $shape"
    )
}
