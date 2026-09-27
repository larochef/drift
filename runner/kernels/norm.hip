#include <hip/hip_runtime.h>

#include "block.h"
#include "combine.h"
#include "quantize.h"

// Row-wise kernels: one block of 256 threads per row of `cols` values.

// out = x / sqrt(mean(x^2) + eps) * (weight + offset); offset 1 for Gemma's
// norms, whose weights are stored minus one. Row r takes weight row r %
// weight_rows: several streams in one row, each normed with its own weights.
// With `y`, x += y first (written back: the residual stream). With `q.xq`,
// out is quantized too (quantize.h), for the int8 products it feeds; cols a
// multiple of 32 then.
__device__ inline void rms_norm_row(float* row, const float* y, const float* weight, float* target, int cols,
                                    float eps, float offset, Quantized q, long long first_block, float* partial) {
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float value = row[c];
    if (y) {
      value = value + y[c];
      row[c] = value;
    }
    sum += value * value;
  }
  const float scale = rsqrtf(block_sum(sum, partial) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float value = row[c] * scale * (weight[c] + offset);
    target[c] = value;
    if (q.xq) quantize_block(value, first_block + c / 32, q.xq, q.xd, q.xs);
  }
}

extern "C" __global__ void rms_norm_f32(float* x, const float* y, const float* weight, float* out, int cols,
                                        float eps, float offset, int weight_rows, int8_t* xq, float* xd, float* xs) {
  __shared__ float partial[256];
  const Quantized q{xq, xd, xs};
  rms_norm_row(x + (long long)blockIdx.x * cols, y ? y + (long long)blockIdx.x * cols : nullptr,
               weight + (long long)(blockIdx.x % weight_rows) * cols, out + (long long)blockIdx.x * cols, cols, eps,
               offset, q, (long long)blockIdx.x * cols / 32, partial);
}

// moe_combine (combine.h) then the RMS norm of its output, one block of 1024
// threads per token: the residual stream's sums as moe_combine and rms_norm_f32
// make them, without two launches between the experts and the next block.
extern "C" __global__ void __launch_bounds__(1024)
    moe_combine_norm(const float* expert_out, const float* weights, const float* shared, const float* x,
                     const float* shared_router, float* residual, int k, int hidden, const float* norm_weight,
                     float* normed, float eps, float offset, int8_t* xq, float* xd, float* xs) {
  __shared__ float partial[256];
  const Quantized q{xq, xd, xs};
  const long long t = blockIdx.x;
  const float gate = shared ? shared_expert_gate(x + t * hidden, shared_router, hidden, partial) : 0.0f;
  __syncthreads();  // partial is reused by the norm
  for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    residual[t * hidden + c] = combined(expert_out, weights, shared, gate, residual, t, c, k, hidden);
  rms_norm_row(residual + t * hidden, nullptr, norm_weight, normed + t * hidden, hidden, eps, offset, q,
               t * hidden / 32, partial);
}

// out = (x - mean) / sqrt(var + eps) * weight + bias; weight and bias may be
// absent (null).
extern "C" __global__ void layer_norm_f32(const float* x, const float* weight, const float* bias, float* out,
                                          int cols, float eps) {
  __shared__ float partial[256];
  const float* row = x + (long long)blockIdx.x * cols;
  float* target = out + (long long)blockIdx.x * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += row[c];
  const float mean = block_sum(sum, partial) / cols;
  float squares = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) squares += (row[c] - mean) * (row[c] - mean);
  const float scale = rsqrtf(block_sum(squares, partial) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float value = (row[c] - mean) * scale;
    if (weight) value *= weight[c];
    if (bias) value += bias[c];
    target[c] = value;
  }
}

// out = softmax(x * scale) over each row.
extern "C" __global__ void softmax_f32(const float* x, float* out, int cols, float scale) {
  __shared__ float partial[256];
  const float* row = x + (long long)blockIdx.x * cols;
  float* target = out + (long long)blockIdx.x * cols;
  float largest = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) largest = fmaxf(largest, row[c] * scale);
  largest = block_max(largest, partial);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(row[c] * scale - largest);
  const float inverse = 1.0f / block_sum(sum, partial);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) target[c] = expf(row[c] * scale - largest) * inverse;
}

// Gated RMS norm (Gated DeltaNet's output): out = x / sqrt(mean(x^2) + eps) *
// weight * act(gate), per row, act silu or (`sigmoid_gate`) sigmoid. `out`
// may be `x`. With `q.xq`, out is quantized too, as rms_norm_f32's.
extern "C" __global__ void gated_rms_norm_f32(const float* x, const float* weight, const float* gate, float* out,
                                              int cols, float eps, int sigmoid_gate, int8_t* xq, float* xd,
                                              float* xs) {
  __shared__ float partial[256];
  const Quantized q{xq, xd, xs};
  const float* row = x + (long long)blockIdx.x * cols;
  const float* g = gate + (long long)blockIdx.x * cols;
  float* target = out + (long long)blockIdx.x * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += row[c] * row[c];
  const float scale = rsqrtf(block_sum(sum, partial) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float value = row[c] * scale * weight[c] * (sigmoid_gate ? 1.0f : g[c]) / (1.0f + expf(-g[c]));
    target[c] = value;
    if (q.xq) quantize_block(value, (long long)blockIdx.x * cols / 32 + c / 32, q.xq, q.xd, q.xs);
  }
}

// The index of the largest of `count` values, the first among equals: one
// block of 1024 threads, into index[0].
extern "C" __global__ void argmax_f32(const float* values, int* index, int count) {
  __shared__ float best_values[32];
  __shared__ int best_indices[32];
  float best = -INFINITY;
  int best_index = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    if (values[i] > best) {
      best = values[i];
      best_index = i;
    }
  auto better = [](float value, int at, float than, int than_at) {
    return value > than || (value == than && at < than_at);
  };
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other = __shfl_xor(best, offset);
    const int other_index = __shfl_xor(best_index, offset);
    if (better(other, other_index, best, best_index)) {
      best = other;
      best_index = other_index;
    }
  }
  if (threadIdx.x % warpSize == 0) {
    best_values[threadIdx.x / warpSize] = best;
    best_indices[threadIdx.x / warpSize] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < blockDim.x / warpSize; ++w)
      if (better(best_values[w], best_indices[w], best, best_index)) {
        best = best_values[w];
        best_index = best_indices[w];
      }
    index[0] = best_index == 0x7FFFFFFF ? 0 : best_index;
  }
}

// A float as an unsigned key in the floats' order, -0 as 0.
__device__ inline unsigned int ordered_key(float value) {
  const unsigned int bits = __float_as_uint(value == 0.0f ? 0.0f : value);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// histogram[bin] += 1 for each active lane of the wave: one atomic per
// distinct bin (logits crowd a few), not one per lane.
__device__ inline void count_bin(unsigned int* histogram, int bin, bool active) {
  const int lane = threadIdx.x % warpSize;
  unsigned long long pending = __ballot(active);
  while (pending) {
    const int leader_bin = __shfl(bin, __ffsll(pending) - 1);
    const unsigned long long same = __ballot(active && bin == leader_bin);
    if (lane == __ffsll(same) - 1) atomicAdd(&histogram[leader_bin], (unsigned int)__popcll(same));
    pending &= ~same;
  }
}

// The LDS of a selection: 1024 threads, up to 1024 kept.
struct Selection {
  unsigned int histogram[256];
  unsigned int keys[1024];
  int ids[1024];
  unsigned int wave_equals[32];
  unsigned int chosen_prefix, chosen_remaining, larger, equals_seen;
};

// The k (at most 1024) largest of n keys, `key(i)` the i-th and `index(i)` its
// index, into s.keys and s.ids [0, k), largest first, the lower index first
// among equals (the keys come in index order). A radix select, a byte at a
// time, finds the k-th largest key; a pass gathers the larger ones and, in
// order, as many equal ones as needed; a bitonic sort orders them.
template <typename Key, typename Index>
__device__ void select_largest(Key key_at, Index index_at, int n, int k, Selection& s) {
  const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  // the k-th largest key, and how many keys equal to it are among the k
  unsigned int prefix = 0, mask = 0, remaining = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) s.histogram[b] = 0;
    __syncthreads();
    for (int base = 0; base < n; base += blockDim.x) {
      const int i = base + threadIdx.x;
      const unsigned int key = i < n ? key_at(i) : 0;
      count_bin(s.histogram, (key >> shift) & 255, i < n && (key & mask) == prefix);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int r = remaining;
      int b = 255;
      while (s.histogram[b] < r) r -= s.histogram[b--];
      s.chosen_prefix = prefix | ((unsigned int)b << shift);
      s.chosen_remaining = r;
    }
    __syncthreads();
    prefix = s.chosen_prefix;
    remaining = s.chosen_remaining;
    mask |= 255u << shift;
  }
  const unsigned int threshold = prefix;
  const int first_equal = k - remaining;

  if (threadIdx.x == 0) s.larger = s.equals_seen = 0;
  __syncthreads();
  for (int base = 0; base < n; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const unsigned int key = i < n ? key_at(i) : 0;
    if (i < n && key > threshold) {
      const unsigned int at = atomicAdd(&s.larger, 1u);
      s.keys[at] = key;
      s.ids[at] = index_at(i);
    }
    const bool equal = i < n && key == threshold;
    const unsigned long long ballot = __ballot(equal);
    if (lane == 0) s.wave_equals[wave] = __popcll(ballot);
    __syncthreads();
    if (equal) {
      unsigned int rank = s.equals_seen + __popcll(ballot & ((1ull << lane) - 1));
      for (int w = 0; w < wave; ++w) rank += s.wave_equals[w];
      if (rank < remaining) {
        s.keys[first_equal + rank] = key;
        s.ids[first_equal + rank] = index_at(i);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
      for (int w = 0; w < blockDim.x / warpSize; ++w) s.equals_seen += s.wave_equals[w];
    __syncthreads();
  }

  int padded = 1;
  while (padded < k) padded *= 2;
  for (int i = k + threadIdx.x; i < padded; i += blockDim.x) {
    s.keys[i] = 0;
    s.ids[i] = 0x7FFFFFFF;
  }
  __syncthreads();
  auto before = [&](int a, int b) {
    return s.keys[a] > s.keys[b] || (s.keys[a] == s.keys[b] && s.ids[a] < s.ids[b]);
  };
  for (int size = 2; size <= padded; size *= 2)
    for (int stride = size / 2; stride > 0; stride /= 2) {
      for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i && ((i & size) == 0 ? before(j, i) : before(i, j))) {
          const unsigned int key = s.keys[i];
          s.keys[i] = s.keys[j];
          s.keys[j] = key;
          const int id = s.ids[i];
          s.ids[i] = s.ids[j];
          s.ids[j] = id;
        }
      }
      __syncthreads();
    }
}

// The largest key and its index (the lower among equals) of n, into s.keys[0]
// and s.ids[0]: one pass, as argmax_f32.
template <typename Key, typename Index>
__device__ void select_largest_one(Key key_at, Index index_at, int n, Selection& s) {
  const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  unsigned int best = 0;
  int best_index = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const unsigned int key = key_at(i);
    const int index = index_at(i);
    if (key > best || (key == best && index < best_index) || best_index == 0x7FFFFFFF) {
      best = key;
      best_index = index;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const unsigned int other = __shfl_xor(best, offset);
    const int other_index = __shfl_xor(best_index, offset);
    if (other > best || (other == best && other_index < best_index)) {
      best = other;
      best_index = other_index;
    }
  }
  if (lane == 0) {
    s.keys[wave] = best;
    s.ids[wave] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < blockDim.x / warpSize; ++w)
      if (s.keys[w] > best || (s.keys[w] == best && s.ids[w] < best_index)) {
        best = s.keys[w];
        best_index = s.ids[w];
      }
    s.keys[0] = best;
    s.ids[0] = best_index;
  }
  __syncthreads();
}

// Each row's `count` (at most 1024) largest of `width` values, largest first,
// the lower index first among equals, into out[row][c] as (value bits, index)
// pairs. Blocks of 1024 threads: each row's CANDIDATE_CHUNK values at a time
// keep their own largest (select_largest) in `kept` ([rows][chunks][count]
// (key, index) pairs, chunk after chunk: all but the last keep `count`); the
// last block of a row to finish (`finished`, one counter per row, left at 0)
// selects among them, in index order still, so the result is a selection over
// the whole row's.
constexpr int CANDIDATE_CHUNK = 4096;

extern "C" __global__ void __launch_bounds__(1024)
    candidates_f32(const float* values, int* out, int width, int count, int* kept, unsigned* finished) {
  __shared__ Selection s;
  __shared__ bool last;
  const int chunks = (width + CANDIDATE_CHUNK - 1) / CANDIDATE_CHUNK;
  const int row_index = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  const float* row = values + (long long)row_index * width;
  const int first = chunk * CANDIDATE_CHUNK, length = min(CANDIDATE_CHUNK, width - first);
  const int k = min(count, width), chunk_k = min(k, length);
  int* row_kept = kept + 2LL * row_index * chunks * count;
  auto key_in_chunk = [&](int i) { return ordered_key(row[first + i]); };
  auto index_in_chunk = [&](int i) { return first + i; };
  if (k == 1) select_largest_one(key_in_chunk, index_in_chunk, length, s);
  else select_largest(key_in_chunk, index_in_chunk, length, chunk_k, s);
  for (int c = threadIdx.x; c < chunk_k; c += blockDim.x) {
    row_kept[2LL * (chunk * count + c)] = int(s.keys[c]);
    row_kept[2LL * (chunk * count + c) + 1] = s.ids[c];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(finished + row_index, 1u) == unsigned(chunks - 1);
    if (last) finished[row_index] = 0;
    __threadfence();
  }
  __syncthreads();
  if (!last) return;
  // every chunk's but the last keeps `count`, and count <= CANDIDATE_CHUNK
  const int n = (chunks - 1) * count + min(k, width - (chunks - 1) * CANDIDATE_CHUNK);
  auto kept_at = [&](int i, int part) {
    return __hip_atomic_load(row_kept + 2LL * i + part, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  };
  auto key_kept = [&](int i) { return (unsigned int)kept_at(i, 0); };
  auto index_kept = [&](int i) { return kept_at(i, 1); };
  if (k == 1) select_largest_one(key_kept, index_kept, n, s);
  else select_largest(key_kept, index_kept, n, k, s);
  int* target = out + 2LL * row_index * count;
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    target[2 * c] = __float_as_int(row[s.ids[c]]);
    target[2 * c + 1] = s.ids[c];
  }
}

// Group norm over x [rows][channels] (channels-last pixels): group g is the
// channels [g c/G, (g + 1) c/G) of every row. Three passes for accuracy over
// millions of values: the means, then the centred squares, then the output.
//
// Pass 1 and 2: block g * blocks + b sums rows [b * rows_per_block, ...) of
// group g, less center[g] and squared when center is given, into partials.
extern "C" __global__ void group_partials_f32(const float* x, const float* center, float* partials, int rows,
                                              int channels, int groups, int blocks, int rows_per_block) {
  __shared__ float partial[256];
  const int g = blockIdx.x / blocks, b = blockIdx.x % blocks;
  const int width = channels / groups;
  const long long first = (long long)b * rows_per_block;
  const long long last = min((long long)rows, first + rows_per_block);
  const long long count = (last - first) * width;
  const float mean = center ? center[g] : 0.0f;
  float sum = 0.0f;
  for (long long i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = x[(first + i / width) * channels + (long long)g * width + i % width] - mean;
    sum += center ? value * value : value;
  }
  sum = block_sum(sum, partial);
  if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

// Block g: stats[g] = the partials' mean (mode 0), or stats[groups + g] =
// 1 / sqrt(their mean + eps) (mode 1), over `count` values per group.
extern "C" __global__ void group_finish_f32(const float* partials, float* stats, int groups, int blocks,
                                            long long count, float eps, int mode) {
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int b = threadIdx.x; b < blocks; b += blockDim.x) sum += partials[(long long)blockIdx.x * blocks + b];
  sum = block_sum(sum, partial);
  if (threadIdx.x == 0) {
    const float mean = sum / (float)count;
    if (mode == 0) stats[blockIdx.x] = mean;
    else stats[groups + blockIdx.x] = rsqrtf(mean + eps);
  }
}

// out = (x - mean[g]) * rstd[g] * weight[c] + bias[c].
extern "C" __global__ void group_apply_f32(const float* x, const float* stats, const float* weight,
                                           const float* bias, float* out, long long values, int channels,
                                           int groups) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const int c = index % channels, g = c / (channels / groups);
  out[index] = (x[index] - stats[g]) * stats[groups + g] * weight[c] + bias[c];
}

// rms_norm_f32 for narrow rows (cols <= 32, a pixel's channels): one thread
// per row, rows past the grid limit of one-block-per-row kernels.
extern "C" __global__ void rms_norm_narrow_f32(const float* x, const float* weight, float* out, long long rows,
                                               int cols, float eps, float offset) {
  const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* row = x + r * cols;
  float* target = out + r * cols;
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) sum += row[c] * row[c];
  const float scale = rsqrtf(sum / cols + eps);
  for (int c = 0; c < cols; ++c) target[c] = row[c] * scale * (weight[c] + offset);
}

// The largest magnitude of x, as its float bits, into *out (zeroed first):
// magnitudes' bits order like the values, so the blocks' maxima merge by
// atomicMax. NaNs are skipped.
extern "C" __global__ void max_abs_f32(const float* x, unsigned* out, long long count) {
  __shared__ float partial[32];
  float m = 0.0f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (long long)gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(x[i]));
  m = block_max(m, partial);
  if (threadIdx.x == 0) atomicMax(out, __float_as_uint(m));
}
