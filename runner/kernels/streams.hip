#include <hip/hip_runtime.h>
#include <stdint.h>

// Qwen 3.8 Flash Next's parts outside the blocks: the residual as `streams`
// copies of the hidden state ([tokens][streams][width]), read and written
// through hyper-connections, and the n-gram embedding (PLE) of layer 1.

// A hyper-connection's read: out[t][c] = mean_j sigmoid(gates[t][j][c]) *
// normed[t][j][c].
extern "C" __global__ void streams_mix(const float* gates, const float* normed, float* out, int tokens,
                                       int streams, int width) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)tokens * width) return;
  const long long t = index / width, c = index % width;
  float sum = 0.0f;
  for (int j = 0; j < streams; ++j) {
    const long long at = (t * streams + j) * width + c;
    sum += normed[at] / (1.0f + expf(-gates[at]));
  }
  out[index] = sum / float(streams);
}

// A hyper-connection's write: x[t][j][c] += 2 sigmoid(logits[t][j] / streams)
// * y[t][c] (zero logits: a plain residual add).
extern "C" __global__ void streams_combine(float* x, const float* y, const float* logits, int tokens,
                                           int streams, int width) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)tokens * streams * width) return;
  const long long c = index % width, tj = index / width, t = tj / streams;
  x[index] += 2.0f / (1.0f + expf(-logits[tj] / float(streams))) * y[t * width + c];
}

// The PLE's gate, one workgroup per (token, stream): s = <key, query> /
// sqrt(width), g = sigmoid(sign(s) sqrt(max(|s|, 1e-6))), out = g * value.
// key and query are [tokens][streams][width], value [tokens][width].
extern "C" __global__ void ple_gate(const float* key, const float* query, const float* value, float* out,
                                    int streams, int width) {
  __shared__ float partial[32];
  const long long tj = blockIdx.x, t = tj / streams;
  const float* k = key + tj * width;
  const float* q = query + tj * width;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) sum += k[c] * q[c];
  for (int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset);
  if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = sum;
  __syncthreads();
  float s = 0.0f;
  for (int w = 0; w < blockDim.x / warpSize; ++w) s += partial[w];
  s /= sqrtf(float(width));
  const float root = sqrtf(fmaxf(fabsf(s), 1e-6f));
  const float g = 1.0f / (1.0f + expf(-(s < 0.0f ? -root : root)));
  const float* v = value + t * width;
  for (int c = threadIdx.x; c < width; c += blockDim.x) out[tj * width + c] = g * v[c];
}

// The n-gram rows of each token, one thread per (token, head): the token and
// its ngram_size - 1 predecessors, a predecessor at or before an EOS replaced
// by EOS (the token itself never is), hashed as
//   mixed_n = (ctx[0] * m[0]) ^ ... ^ (ctx[n-1] * m[n-1])   (uint64, wrapping)
//   row_h = mixed_n % sizes[h] + offsets[h],  n = 2 + h / heads_per_ngram.
// The predecessors before the call's first token come from `state`, the last
// ngram_size - 1 tokens seen, oldest first, stored as token + 1 (0: none, taken
// as EOS). State and history hold floats (token ids below 2^24 are exact).
// Thread 0 of each token also writes the history row after that token; the
// last block to see the last token updates the state.
extern "C" __global__ void ngram_rows(const int* tokens, float* state, const uint64_t* multipliers,
                                      const uint64_t* sizes, const uint64_t* offsets, int* rows, int count,
                                      int ngram_size, int heads_per_ngram, int eos, float* history) {
  const int heads = (ngram_size - 1) * heads_per_ngram;
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)count * heads) return;
  const int t = index / heads, h = index % heads;
  const int kept = ngram_size - 1;
  // token at offset k back (0 = the token itself)
  auto at = [&](int k) -> int {
    const int p = t - k;
    if (p >= 0) return tokens[p];
    const float stored = state[kept + p];  // p in [-kept, -1]
    return stored == 0.0f ? eos : int(stored) - 1;
  };
  int context[8];  // ngram_size <= 8
  context[0] = at(0);
  bool cut = false;
  for (int k = 1; k < ngram_size; ++k) {
    const int previous = at(k);
    cut = cut || previous == eos;
    context[k] = cut ? eos : previous;
  }
  const int n = 2 + h / heads_per_ngram;
  uint64_t mixed = 0;
  for (int j = 0; j < n; ++j) mixed ^= uint64_t(uint32_t(context[j])) * multipliers[j];
  rows[index] = int(mixed % sizes[h] + offsets[h]);
  if (h == 0) {
    float* row = history ? history + (long long)t * kept : nullptr;
    for (int i = 0; i < kept; ++i) {
      const float value = float(at(kept - 1 - i) + 1);
      if (row) row[i] = value;
    }
  }
}

// After ngram_rows: the state becomes the last ngram_size - 1 tokens (one
// thread; the earlier state is read before being overwritten).
extern "C" __global__ void ngram_state(const int* tokens, float* state, int count, int kept) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float next[8];
  for (int i = 0; i < kept; ++i) {
    const int p = count - kept + i;
    next[i] = p >= 0 ? float(tokens[p] + 1) : state[kept + p];
  }
  for (int i = 0; i < kept; ++i) state[i] = next[i];
}
