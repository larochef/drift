#include <hip/hip_runtime.h>

#include "combine.h"
#include "route.h"

// Gated DeltaNet (Qwen 3.5/3.6's linear-attention layers) and the MoE glue.
// Every recurrent quantity is F32, as transformers computes it.

// Causal depthwise convolution over time, `taps` taps `dilation` apart, then
// SiLU: out[t] = silu(sum_i weight[c][i] * x[t - (taps-1-i) * dilation]), the
// inputs before the call's first token coming from `state` ([span][channels],
// span = (taps-1) * dilation, oldest first), which is left holding the call's
// last span inputs. One thread per channel walks the tokens in order. With a
// `history` ([tokens][span][channels]), the state after each token is kept
// too: speculative decoding rolls back to one.
extern "C" __global__ void causal_conv_silu(const float* x, const float* weight, float* state, float* out,
                                            int tokens, int channels, int taps, int dilation, float* history) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int span = (taps - 1) * dilation;
  float window[16];  // span <= 16
  for (int i = 0; i < span; ++i) window[i] = state[(long long)i * channels + c];
  const float* w = weight + (long long)c * taps;
  for (int t = 0; t < tokens; ++t) {
    const float current = x[(long long)t * channels + c];
    float sum = w[taps - 1] * current;
    for (int i = 0; i < taps - 1; ++i) sum += w[i] * window[i * dilation];
    out[(long long)t * channels + c] = sum / (1.0f + expf(-sum));
    for (int i = 0; i + 1 < span; ++i) window[i] = window[i + 1];
    if (span > 0) window[span - 1] = current;
    if (history)
      for (int i = 0; i < span; ++i) history[((long long)t * span + i) * channels + c] = window[i];
  }
  for (int i = 0; i < span; ++i) state[(long long)i * channels + c] = window[i];
}

// The gated delta rule, token after token (transformers'
// torch_recurrent_gated_delta_rule), for value head h's state S [D][D]:
//   q, k: the convolved q and k of h's key head, L2-normalized, q / sqrt(D)
//   g = exp(decay[h] * softplus(a + dt_bias[h])),  beta = sigmoid(b)
//   S *= g;  delta = (v - k.S) * beta;  S += k (x) delta;  o = q.S
// Each column j of S is independent, so a head's columns spread over D /
// COLUMNS workgroups, and each column over PARTS waves holding D / PARTS of its
// rows in registers (a wave reads COLUMNS adjacent columns of a row at once);
// the parts' sums meet in LDS.
// `qkv` rows are [q (key heads x D) | k (key heads x D) | v (value heads x D)].
// Value head h reads key head h % key_heads when `tiled` (llama.cpp's GGUF
// order), h / (value_heads / key_heads) otherwise (transformers').
// With a `history` ([tokens][value heads][D][D]), the state after each token is
// kept too: speculative decoding rolls back to one.
constexpr int DELTA_THREADS = 128, DELTA_PARTS = 4, DELTA_COLUMNS = DELTA_THREADS / DELTA_PARTS;

__device__ inline float threads_sum(float value, float* partial) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) value += __shfl_xor(value, offset);
  if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = value;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < blockDim.x / warpSize; ++w) total += partial[w];
  __syncthreads();
  return total;
}

__device__ inline float parts_sum(float value, float (*parts)[DELTA_COLUMNS], int part, int c) {
  parts[part][c] = value;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int p = 0; p < DELTA_PARTS; ++p) total += parts[p][c];
  __syncthreads();
  return total;
}

// Where the delta rule's q, k and v come from: `convolved` rows as
// causal_conv_silu wrote them.
struct Convolved {
  const float* convolved;
  __device__ float at(int t, long long width, long long channel, int /* reader: which of its readers */) const {
    return convolved[t * width + channel];
  }
  __device__ void finish() const {}
};

// Or the projections themselves, convolved here as causal_conv_silu does it
// (taps 4, dilation 1): each thread keeps the windows of the channels it
// reads, and one reader of a channel writes its state and history. A key
// head's q and k have several readers (its value heads' blocks), so their
// keeper writes their state only once every other reader has read it (`reads`
// counts them).
template <int D>
struct Convolving {
  static constexpr int SPAN = 3, TAPS = SPAN + 1;
  const float* x;
  const float* weight;
  float* state;
  float* history;
  int channels;
  bool keeper;      // this block keeps its key head's q and k channels
  unsigned* reads;  // this key head's readers that have read their state
  unsigned readers; // how many other blocks read it
  // the windows of the channels this thread reads: q, k and v
  float windows[3][SPAN];
  long long read[3];
  int tokens_done[3];

  __device__ void open(int reader, long long channel) {
    read[reader] = channel;
    tokens_done[reader] = 0;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) windows[reader][i] = state[(long long)i * channels + channel];
  }
  // reader 0 and 1 read q and k (threads < D), 2 reads v
  __device__ float at(int t, long long width, long long channel, int reader) {
    const float current = x[t * width + channel];
    const float* w = weight + channel * TAPS;
    float sum;
    {
#pragma clang fp contract(off)  // the fused products causal_conv_silu compiles to
      sum = w[TAPS - 1] * current;
#pragma unroll
      for (int i = 0; i < TAPS - 1; ++i) sum = fmaf(w[i], windows[reader][i], sum);
    }
    const float out = sum / (1.0f + expf(-sum));
#pragma unroll
    for (int i = 0; i + 1 < SPAN; ++i) windows[reader][i] = windows[reader][i + 1];
    windows[reader][SPAN - 1] = current;
    if (history && writes(reader))
      for (int i = 0; i < SPAN; ++i) history[((long long)t * SPAN + i) * channels + channel] = windows[reader][i];
    tokens_done[reader] = t + 1;
    return out;
  }
  __device__ bool writes(int reader) const {
    return reader < 2 ? keeper : threadIdx.x < DELTA_COLUMNS;  // v: the first part's thread of each column
  }
  // after every reader's open: the other readers say they have read
  __device__ void opened() const {
    __syncthreads();
    if (!keeper && threadIdx.x == 0) __hip_atomic_fetch_add(reads, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __device__ void finish() const {
    if (keeper)
      while (__hip_atomic_load(reads, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < readers)
        __builtin_amdgcn_s_sleep(2);
#pragma unroll
    for (int reader = 0; reader < 3; ++reader)
      if (tokens_done[reader] > 0 && writes(reader))
        for (int i = 0; i < SPAN; ++i) state[(long long)i * channels + read[reader]] = windows[reader][i];
  }
};

template <int D, typename Inputs>
__device__ void delta_rule(Inputs& inputs, int block, const float* a, const float* b, const float* decay,
                           const float* dt_bias, float* state, float* out, int tokens, int key_heads, int value_heads,
                           int tiled, float* history) {
  constexpr int ROWS = D / DELTA_PARTS, SLICES = D / DELTA_COLUMNS;
  __shared__ float q[D], k[D], partial[DELTA_THREADS / 32], parts[DELTA_PARTS][DELTA_COLUMNS];
  const int h = block / SLICES, part = threadIdx.x / DELTA_COLUMNS, c = threadIdx.x % DELTA_COLUMNS;
  const int j = (block % SLICES) * DELTA_COLUMNS + c;
  const int key_head = tiled ? h % key_heads : h / (value_heads / key_heads);
  const long long width = 2LL * key_heads * D + (long long)value_heads * D;
  const float scale = rsqrtf(float(D));
  float* column = state + (long long)h * D * D + (long long)part * ROWS * D + j;
  float s[ROWS];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) s[i] = column[(long long)i * D];
  for (int t = 0; t < tokens; ++t) {
    float q_squares = 0.0f, k_squares = 0.0f;
    for (int i = threadIdx.x; i < D; i += DELTA_THREADS) {
      q[i] = inputs.at(t, width, key_head * D + i, 0);
      k[i] = inputs.at(t, width, (long long)key_heads * D + key_head * D + i, 1);
      q_squares += q[i] * q[i];
      k_squares += k[i] * k[i];
    }
    const float q_factor = rsqrtf(threads_sum(q_squares, partial) + 1e-6f) * scale;
    const float k_factor = rsqrtf(threads_sum(k_squares, partial) + 1e-6f);
    const float vv = inputs.at(t, width, 2LL * key_heads * D + (long long)h * D + j, 2);
    const float x = a[(long long)t * value_heads + h] + dt_bias[h];
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    const float g = expf(decay[h] * softplus);
    const float beta = 1.0f / (1.0f + expf(-b[(long long)t * value_heads + h]));
    const float* my_q = q + part * ROWS;
    const float* my_k = k + part * ROWS;
    float memory = 0.0f;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      s[i] *= g;
      memory += s[i] * my_k[i];
    }
    memory = parts_sum(memory, parts, part, c);
    const float delta = (vv - memory * k_factor) * beta * k_factor;
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      s[i] += my_k[i] * delta;
      o += s[i] * my_q[i];
    }
    o = parts_sum(o, parts, part, c);
    if (part == 0) out[((long long)t * value_heads + h) * D + j] = o * q_factor;
    if (history) {
      float* kept = history + ((long long)t * value_heads + h) * D * D + (long long)part * ROWS * D + j;
#pragma unroll
      for (int i = 0; i < ROWS; ++i) kept[(long long)i * D] = s[i];
    }
    __syncthreads();  // q and k are rewritten for the next token
  }
#pragma unroll
  for (int i = 0; i < ROWS; ++i) column[(long long)i * D] = s[i];
  inputs.finish();
}

#define DELTA_RULE(d)                                                                                           \
  extern "C" __global__ void __launch_bounds__(DELTA_THREADS)                                                  \
      delta_rule_d##d(const float* qkv, const float* a, const float* b, const float* decay,                    \
                      const float* dt_bias, float* state, float* out, int tokens, int key_heads,               \
                      int value_heads, int tiled, float* history) {                                            \
    Convolved inputs{qkv};                                                                                     \
    delta_rule<d>(inputs, blockIdx.x, a, b, decay, dt_bias, state, out, tokens, key_heads, value_heads, tiled, \
                  history);                                                                                    \
  }                                                                                                            \
  /* causal_conv_silu of the projections (taps 4, dilation 1), then the rule, in one launch. The blocks  \
     take their heads in the order they start (a ticket in `sync`), each key head's keeper last, so that  \
     the readers it waits for have started; `sync` ([1 + key heads + 1] counters, zero between launches)    \
     is left at zero by the last block to finish. */                                                        \
  extern "C" __global__ void __launch_bounds__(DELTA_THREADS) conv_delta_rule_d##d(                           \
      const float* qkv, const float* conv_weight, float* conv_state, float* conv_history, const float* a,      \
      const float* b, const float* decay, const float* dt_bias, float* state, float* out, int tokens,          \
      int key_heads, int value_heads, int tiled, float* history, unsigned* sync) {                             \
    constexpr int SLICES = d / DELTA_COLUMNS;                                                                  \
    __shared__ unsigned ticket;                                                                                \
    if (threadIdx.x == 0) ticket = __hip_atomic_fetch_add(sync, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); \
    __syncthreads();                                                                                           \
    const int ratio = value_heads / key_heads, others = gridDim.x - key_heads;                                \
    /* the keepers: slice 0 of each key head's first value head */                                             \
    int h, slice;                                                                                              \
    if ((int)ticket >= others) {                                                                               \
      const int key_head = ticket - others;                                                                    \
      h = tiled ? key_head : key_head * ratio;                                                                 \
      slice = 0;                                                                                               \
    } else {                                                                                                   \
      /* the others in (value head, slice) order, the keepers skipped */                                     \
      int at = ticket;                                                                                         \
      int block = 0;                                                                                           \
      for (;; ++block) {                                                                                       \
        const int bh = block / SLICES, bs = block % SLICES;                                                    \
        const bool keeps = bs == 0 && (tiled ? bh < key_heads : bh % ratio == 0);                              \
        if (!keeps && at-- == 0) break;                                                                        \
      }                                                                                                        \
      h = block / SLICES;                                                                                      \
      slice = block % SLICES;                                                                                  \
    }                                                                                                          \
    const int j = slice * DELTA_COLUMNS + threadIdx.x % DELTA_COLUMNS;                                        \
    const int key_head = tiled ? h % key_heads : h / ratio;                                                    \
    const int channels = 2 * key_heads * d + value_heads * d;                                                  \
    Convolving<d> inputs{qkv,  conv_weight, conv_state,          conv_history,                                 \
                         channels, (int)ticket >= others, sync + 1 + key_head, unsigned(ratio * SLICES - 1)};  \
    for (int i = threadIdx.x; i < d; i += DELTA_THREADS) {                                                     \
      inputs.open(0, key_head * d + i);                                                                        \
      inputs.open(1, key_heads * d + key_head * d + i);                                                        \
    }                                                                                                          \
    inputs.open(2, 2 * key_heads * d + h * d + j);                                                             \
    inputs.opened();                                                                                           \
    delta_rule<d>(inputs, h * SLICES + slice, a, b, decay, dt_bias, state, out, tokens, key_heads,             \
                  value_heads, tiled, history);                                                                \
    __syncthreads();                                                                                           \
    if (threadIdx.x == 0 &&                                                                                    \
        __hip_atomic_fetch_add(sync + 1 + key_heads, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==        \
            gridDim.x - 1)                                                                                     \
      for (int i = 0; i < 2 + key_heads; ++i) sync[i] = 0;                                                     \
  }
DELTA_RULE(32)
DELTA_RULE(64)
DELTA_RULE(128)

// Top-k routing, one wave per token (route.h).
extern "C" __global__ void route_topk(const float* logits, int* ids, float* weights, int tokens, int experts,
                                      int k) {
  const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= tokens) return;
  const float* row = logits + (long long)t * experts;
  route_token([&](int i) { return row[i]; }, ids + (long long)t * k, weights + (long long)t * k, experts, k);
}

// The mixture of experts' output (combine.h), SLICES workgroups per token, each
// over a slice of the hidden size; each reduces the shared expert's gate logit
// itself (a one-row product not worth a launch). `out` may be `residual`.
constexpr int SLICES = 8;
extern "C" __global__ void __launch_bounds__(256)
    moe_combine(const float* expert_out, const float* weights, const float* shared, const float* x,
                const float* shared_router, const float* residual, float* out, int k, int hidden) {
  __shared__ float partial[256];
  const long long t = blockIdx.x / SLICES;
  const int slice = blockIdx.x % SLICES;
  const float gate = shared ? shared_expert_gate(x + t * hidden, shared_router, hidden, partial) : 0.0f;
  const int begin = (long long)hidden * slice / SLICES, end = (long long)hidden * (slice + 1) / SLICES;
  for (int c = begin + threadIdx.x; c < end; c += blockDim.x)
    out[t * hidden + c] = combined(expert_out, weights, shared, gate, residual, t, c, k, hidden);
}

// The two halves of each row's last dimension: [R][2D] into [R][D] and [R][D].
extern "C" __global__ void split_halves(const float* x, float* first, float* second, long long rows, int d) {
  const long long count = rows * d;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / d, c = i % d;
    first[i] = x[r * 2 * d + c];
    second[i] = x[r * 2 * d + d + c];
  }
}

// The inverse: [R][D] and [R][D] side by side into [R][2D].
extern "C" __global__ void join_halves(const float* first, const float* second, float* x, long long rows, int d) {
  const long long count = rows * d;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / d, c = i % d;
    x[r * 2 * d + c] = first[i];
    x[r * 2 * d + d + c] = second[i];
  }
}
